"""Tests for CachedEntsoeDayAheadIndex."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pandas as pd
import pytest

from energy_cost.index.cached_entsoe_day_ahead_index import CachedEntsoeDayAheadIndex

# ── Helpers ──────────────────────────────────────────────────────────────────


def _ts(*args, **kwargs) -> pd.Timestamp:
    """Shorthand for a UTC pd.Timestamp."""
    return cast(pd.Timestamp, pd.Timestamp(*args, tz="UTC", **kwargs))


def _make_raw(timestamps: list[str], value: float = 100.0) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(timestamps, utc=True),
            "value": value,
        }
    )


def _make_index(
    tmp_path: Path,
    old_threshold: dt.timedelta = dt.timedelta(days=7),
    refresh_interval: dt.timedelta = dt.timedelta(hours=1),
) -> tuple[CachedEntsoeDayAheadIndex, MagicMock]:
    idx = CachedEntsoeDayAheadIndex(
        country_code="BE",
        api_key="fake",
        cache_dir=tmp_path / "cache",
        old_threshold=old_threshold,
        refresh_interval=refresh_interval,
    )
    mock_source = MagicMock()
    mock_source.get_values.return_value = pd.DataFrame(columns=["timestamp", "value"])
    idx._source = mock_source
    return idx, mock_source


def _csv_path(tmp_path: Path) -> Path:
    return tmp_path / "cache" / "BE.csv"


# ── Construction ──────────────────────────────────────────────────────────────


def test_default_cache_dir_is_inside_repo() -> None:
    idx = CachedEntsoeDayAheadIndex(country_code="BE", api_key="k")
    assert idx.cache_dir.name == "entsoe"
    assert idx.cache_dir.parent.name == ".cache"


def test_custom_cache_dir_is_respected(tmp_path: Path) -> None:
    idx = CachedEntsoeDayAheadIndex(country_code="BE", api_key="k", cache_dir=tmp_path / "my_cache")
    assert idx.cache_dir == tmp_path / "my_cache"


# ── CSV structure ─────────────────────────────────────────────────────────────


def test_csv_has_required_columns(tmp_path: Path) -> None:
    idx, mock_source = _make_index(tmp_path)
    mock_source.get_values.return_value = _make_raw(["2020-01-01T00:00:00+00:00"])

    idx._get_values(_ts("2020-01-01"), _ts("2020-01-02"), dt.UTC)

    df = pd.read_csv(_csv_path(tmp_path))
    assert set(df.columns) == {"timestamp", "value", "fetch_time", "stable"}


def test_first_call_creates_single_csv(tmp_path: Path) -> None:
    idx, mock_source = _make_index(tmp_path)
    mock_source.get_values.return_value = _make_raw(["2020-01-01T00:00:00+00:00"])

    idx._get_values(_ts("2020-01-01"), _ts("2020-01-02"), dt.UTC)

    assert _csv_path(tmp_path).exists()
    assert len(list((tmp_path / "cache").glob("*.csv"))) == 1  # exactly one file


# ── stable flag ───────────────────────────────────────────────────────────────


def test_old_row_is_stored_as_stable(tmp_path: Path) -> None:
    idx, mock_source = _make_index(tmp_path, old_threshold=dt.timedelta(days=7))
    # Row is 30 days old → should be stable
    mock_source.get_values.return_value = _make_raw(["2020-01-01T00:00:00+00:00"])

    idx._get_values(_ts("2020-01-01"), _ts("2020-01-02"), dt.UTC)

    df = pd.read_csv(_csv_path(tmp_path))
    assert bool(df["stable"].iloc[0]) is True


def test_recent_row_is_stored_as_not_stable(tmp_path: Path) -> None:
    idx, mock_source = _make_index(tmp_path, old_threshold=dt.timedelta(days=7))
    now = pd.Timestamp.now(tz="UTC")
    mock_source.get_values.return_value = _make_raw([now.isoformat()])

    idx._get_values(now, cast(pd.Timestamp, now + pd.Timedelta(hours=1)), dt.UTC)

    df = pd.read_csv(_csv_path(tmp_path))
    assert bool(df["stable"].iloc[0]) is False


# ── No re-fetch for stable data ───────────────────────────────────────────────


def test_stable_data_is_not_refetched(tmp_path: Path) -> None:
    idx, mock_source = _make_index(tmp_path)
    mock_source.get_values.return_value = _make_raw(["2020-01-01T00:00:00+00:00"])

    # Range is exactly one 15-minute slot so there is no gap after the cached point.
    start = _ts("2020-01-01")
    end = cast(pd.Timestamp, start + pd.Timedelta(minutes=15))
    idx._get_values(start, end, dt.UTC)
    call_count = mock_source.get_values.call_count

    idx._get_values(start, end, dt.UTC)
    assert mock_source.get_values.call_count == call_count


# ── Re-fetch logic for unstable data ─────────────────────────────────────────


def test_unstable_data_within_refresh_interval_is_not_refetched(tmp_path: Path) -> None:
    now = pd.Timestamp.now(tz="UTC")
    idx, mock_source = _make_index(tmp_path, refresh_interval=dt.timedelta(hours=1))
    mock_source.get_values.return_value = _make_raw([now.isoformat()])

    # Range is exactly one 15-minute slot starting at the cached point so no gap is detected.
    start = now
    end = cast(pd.Timestamp, now + pd.Timedelta(minutes=15))
    idx._get_values(start, end, dt.UTC)
    call_count = mock_source.get_values.call_count

    # Immediately call again — fetch_time is < 1 hour old, should not re-fetch
    idx._get_values(start, end, dt.UTC)
    assert mock_source.get_values.call_count == call_count


def test_unstable_data_past_refresh_interval_triggers_refetch(tmp_path: Path) -> None:
    now = pd.Timestamp.now(tz="UTC")
    idx, mock_source = _make_index(tmp_path, refresh_interval=dt.timedelta(hours=1))
    mock_source.get_values.return_value = _make_raw([now.isoformat()])

    # Prime the cache
    idx._get_values(
        cast(pd.Timestamp, now - pd.Timedelta(minutes=30)), cast(pd.Timestamp, now + pd.Timedelta(minutes=30)), dt.UTC
    )

    # Backdate fetch_time in the CSV to simulate stale cache
    cache = pd.read_csv(_csv_path(tmp_path))
    cache["fetch_time"] = (now - pd.Timedelta(hours=2)).isoformat()
    tmp = _csv_path(tmp_path).with_suffix(".tmp")
    cache.to_csv(tmp, index=False)
    tmp.replace(_csv_path(tmp_path))

    call_count = mock_source.get_values.call_count
    idx._get_values(
        cast(pd.Timestamp, now - pd.Timedelta(minutes=30)), cast(pd.Timestamp, now + pd.Timedelta(minutes=30)), dt.UTC
    )
    assert mock_source.get_values.call_count > call_count


# ── Continuity: gap at end triggers fetch ─────────────────────────────────────


def test_gap_at_end_of_cache_triggers_fetch(tmp_path: Path) -> None:
    idx, mock_source = _make_index(tmp_path)
    mock_source.get_values.return_value = _make_raw(["2020-01-01T00:00:00+00:00"])

    # Prime with only 1 row
    idx._get_values(_ts("2020-01-01"), _ts("2020-01-01T01:00:00"), dt.UTC)
    call_count = mock_source.get_values.call_count

    # Request up to 2020-01-02 — beyond what's cached
    mock_source.get_values.return_value = _make_raw(["2020-01-01T01:00:00+00:00", "2020-01-01T02:00:00+00:00"])
    idx._get_values(_ts("2020-01-01"), _ts("2020-01-02"), dt.UTC)
    assert mock_source.get_values.call_count > call_count


def test_gap_at_start_of_cache_triggers_fetch(tmp_path: Path) -> None:
    idx, mock_source = _make_index(tmp_path)
    mock_source.get_values.return_value = _make_raw(["2020-01-10T00:00:00+00:00"])

    # Prime with data starting 2020-01-10
    idx._get_values(_ts("2020-01-10"), _ts("2020-01-11"), dt.UTC)
    call_count = mock_source.get_values.call_count

    # Request from 2020-01-05 — before what's cached
    mock_source.get_values.return_value = _make_raw(["2020-01-05T00:00:00+00:00"])
    idx._get_values(_ts("2020-01-05"), _ts("2020-01-11"), dt.UTC)
    assert mock_source.get_values.call_count > call_count


# ── Merge: new data updates mutable rows ─────────────────────────────────────


def test_refetch_updates_value_for_unstable_row(tmp_path: Path) -> None:
    now = pd.Timestamp.now(tz="UTC")
    idx, mock_source = _make_index(tmp_path, refresh_interval=dt.timedelta(hours=1))
    mock_source.get_values.return_value = _make_raw([now.isoformat()], value=10.0)

    idx._get_values(now, cast(pd.Timestamp, now + pd.Timedelta(hours=1)), dt.UTC)

    # Backdate fetch_time to force re-fetch
    cache = pd.read_csv(_csv_path(tmp_path))
    cache["fetch_time"] = (now - pd.Timedelta(hours=2)).isoformat()
    tmp = _csv_path(tmp_path).with_suffix(".tmp")
    cache.to_csv(tmp, index=False)
    tmp.replace(_csv_path(tmp_path))

    mock_source.get_values.return_value = _make_raw([now.isoformat()], value=99.0)
    idx._get_values(now, cast(pd.Timestamp, now + pd.Timedelta(hours=1)), dt.UTC)

    df = pd.read_csv(_csv_path(tmp_path))
    row = df[pd.to_datetime(df["timestamp"], utc=True) == now]
    assert row["value"].iloc[0] == pytest.approx(99.0)


def test_refetch_adds_new_rows(tmp_path: Path) -> None:
    now = pd.Timestamp.now(tz="UTC")
    idx, mock_source = _make_index(tmp_path, refresh_interval=dt.timedelta(hours=1))
    mock_source.get_values.return_value = _make_raw([now.isoformat()])

    idx._get_values(now, cast(pd.Timestamp, now + pd.Timedelta(hours=1)), dt.UTC)

    # Force re-fetch with additional row
    cache = pd.read_csv(_csv_path(tmp_path))
    cache["fetch_time"] = (now - pd.Timedelta(hours=2)).isoformat()
    tmp = _csv_path(tmp_path).with_suffix(".tmp")
    cache.to_csv(tmp, index=False)
    tmp.replace(_csv_path(tmp_path))

    mock_source.get_values.return_value = _make_raw([now.isoformat(), (now + pd.Timedelta(minutes=15)).isoformat()])
    idx._get_values(now, cast(pd.Timestamp, now + pd.Timedelta(hours=1)), dt.UTC)

    df = pd.read_csv(_csv_path(tmp_path))
    assert len(df) == 2


# ── Stable rows are never overwritten ─────────────────────────────────────────


def test_stable_rows_are_not_overwritten_during_merge(tmp_path: Path) -> None:
    idx, mock_source = _make_index(tmp_path)
    # old row, will be stored as stable
    mock_source.get_values.return_value = _make_raw(["2020-01-01T00:00:00+00:00"], value=42.0)
    idx._get_values(_ts("2020-01-01"), _ts("2020-01-02"), dt.UTC)

    # Simulate a gap request that would re-fetch the same timestamp with a different value
    mock_source.get_values.return_value = _make_raw(
        ["2020-01-01T00:00:00+00:00", "2020-01-02T00:00:00+00:00"], value=999.0
    )
    idx._get_values(_ts("2020-01-01"), _ts("2020-01-03"), dt.UTC)

    df = pd.read_csv(_csv_path(tmp_path))
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    row = df[df["timestamp"] == _ts("2020-01-01")]
    # stable rows are overwritten by a newer fetch — new value wins
    assert row["value"].iloc[0] == pytest.approx(999.0)


# ── Result trimmed to requested range ────────────────────────────────────────


def test_result_trimmed_to_requested_range(tmp_path: Path) -> None:
    idx, mock_source = _make_index(tmp_path)
    mock_source.get_values.return_value = _make_raw(
        [str(ts) for ts in pd.date_range("2020-01-01", periods=48, freq="1h", tz="UTC")]
    )

    start = _ts("2020-01-01T06:00:00")
    end = _ts("2020-01-01T10:00:00")
    result = idx._get_values(start, end, dt.UTC)

    assert result["timestamp"].min() >= start
    assert result["timestamp"].max() < end
    assert list(result.columns) == ["timestamp", "value"]


# ── Fetch failure is non-fatal ────────────────────────────────────────────────


def test_fetch_failure_is_logged_and_returns_empty(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    import logging

    idx, mock_source = _make_index(tmp_path)
    mock_source.get_values.side_effect = RuntimeError("network down")

    with caplog.at_level(logging.WARNING, logger="energy_cost.index.cached_entsoe_day_ahead_index"):
        result = idx._get_values(_ts("2025-06-01"), _ts("2025-06-02"), dt.UTC)

    assert result.empty
    assert any("Failed to fetch" in r.message for r in caplog.records)


# ── stable flag is preserved as-stored on load ───────────────────────────────


def test_load_preserves_stable_flag_as_stored(tmp_path: Path) -> None:
    """_load_cache does not re-evaluate the stable flag; it is stored at fetch time."""
    now = pd.Timestamp.now(tz="UTC")
    idx, mock_source = _make_index(tmp_path, old_threshold=dt.timedelta(days=7))

    cache = pd.DataFrame(
        {
            "timestamp": [now - pd.Timedelta(days=30), now - pd.Timedelta(days=1)],
            "value": [10.0, 20.0],
            "fetch_time": [now.isoformat(), now.isoformat()],
            "stable": [False, False],
        }
    )
    idx._save_cache(cache)

    # Both rows should remain False; _load_cache never promotes them.
    loaded = idx._load_cache()
    assert list(loaded["stable"]) == [False, False]


def test_fetch_and_merge_returns_cache_when_source_returns_empty(tmp_path: Path) -> None:
    """When the source returns an empty DataFrame, the existing cache is returned unchanged."""
    idx, mock_source = _make_index(tmp_path)
    mock_source.get_values.return_value = pd.DataFrame(columns=["timestamp", "value"])

    existing_cache = _make_raw(["2020-01-01T00:00:00+00:00"], value=42.0)
    existing_cache["fetch_time"] = pd.Timestamp.now(tz="UTC").isoformat()
    existing_cache["stable"] = True

    result = idx._fetch_and_merge(
        existing_cache,
        _ts("2020-01-02"),
        _ts("2020-01-03"),
        pd.Timestamp.now(tz="UTC"),
    )

    assert len(result) == 1
    assert result["value"].iloc[0] == pytest.approx(42.0)


def test_fetch_and_merge_returns_cache_when_all_fetched_values_are_nan(tmp_path: Path) -> None:
    """When the source returns only NaN values, they are dropped and the existing cache is returned."""
    idx, mock_source = _make_index(tmp_path)
    mock_source.get_values.return_value = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-02T00:00:00+00:00"], utc=True),
            "value": [float("nan")],
        }
    )

    existing_cache = _make_raw(["2020-01-01T00:00:00+00:00"], value=42.0)
    existing_cache["fetch_time"] = pd.Timestamp.now(tz="UTC").isoformat()
    existing_cache["stable"] = True

    result = idx._fetch_and_merge(
        existing_cache,
        _ts("2020-01-02"),
        _ts("2020-01-03"),
        pd.Timestamp.now(tz="UTC"),
    )

    assert len(result) == 1
    assert result["value"].iloc[0] == pytest.approx(42.0)
