from __future__ import annotations

import datetime as dt

import pandas as pd

from energy_cost.formula import IndexFormula
from energy_cost.tariff import Tariff
from energy_cost.tariff_version import TariffVersion
from energy_cost.versioning import Versioned

# ---------------------------------------------------------------------------
# Versioned – end field
# ---------------------------------------------------------------------------


class TestVersionedEnd:
    def test_end_defaults_to_none(self) -> None:
        v = Versioned(start=dt.datetime(2025, 1, 1))
        assert v.end is None

    def test_end_can_be_set(self) -> None:
        v = Versioned(start=dt.datetime(2025, 1, 1), end=dt.datetime(2026, 1, 1))
        assert v.end == dt.datetime(2026, 1, 1)


# ---------------------------------------------------------------------------
# VersionedCollection – gap-aware find_active_versions
# ---------------------------------------------------------------------------


def _tariff_version(start: dt.datetime, end: dt.datetime | None = None, rate: float = 1.0) -> TariffVersion:
    return TariffVersion(
        start=start,
        end=end,
        consumption={"all": {"energy": IndexFormula(constant_cost=rate)}},
    )


_D = dt.datetime


class TestFindActiveVersionsWithEnd:
    def test_explicit_end_limits_version_range(self) -> None:
        """A version with an explicit end should not cover time after that end."""
        tariff = Tariff(
            versions=[
                _tariff_version(
                    dt.datetime(2025, 1, 1, tzinfo=dt.UTC), end=dt.datetime(2025, 6, 1, tzinfo=dt.UTC), rate=10.0
                ),
            ]
        )
        segments = tariff.find_active_versions(
            dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
            dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
        )
        assert len(segments) == 1
        _, seg_start, seg_end = segments[0]
        assert seg_start == dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
        assert seg_end == dt.datetime(2025, 6, 1, tzinfo=dt.UTC)

    def test_gap_between_versions_produces_no_segment(self) -> None:
        """A gap between two versions should not produce any segment."""
        tariff = Tariff(
            versions=[
                _tariff_version(
                    dt.datetime(2025, 1, 1, tzinfo=dt.UTC), end=dt.datetime(2025, 3, 1, tzinfo=dt.UTC), rate=10.0
                ),
                _tariff_version(dt.datetime(2025, 6, 1, tzinfo=dt.UTC), rate=20.0),
            ]
        )
        segments = tariff.find_active_versions(
            dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
            dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
        )
        assert len(segments) == 2
        _, start1, end1 = segments[0]
        _, start2, end2 = segments[1]
        assert start1 == dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
        assert end1 == dt.datetime(2025, 3, 1, tzinfo=dt.UTC)
        assert start2 == dt.datetime(2025, 6, 1, tzinfo=dt.UTC)
        assert end2 == dt.datetime(2026, 1, 1, tzinfo=dt.UTC)

    def test_contiguous_versions_without_end_behave_as_before(self) -> None:
        """Versions without explicit end fall back to next version's start."""
        tariff = Tariff(
            versions=[
                _tariff_version(dt.datetime(2025, 1, 1, tzinfo=dt.UTC), rate=10.0),
                _tariff_version(dt.datetime(2025, 7, 1, tzinfo=dt.UTC), rate=20.0),
            ]
        )
        segments = tariff.find_active_versions(
            dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
            dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
        )
        assert len(segments) == 2
        _, start1, end1 = segments[0]
        _, start2, end2 = segments[1]
        assert end1 == dt.datetime(2025, 7, 1, tzinfo=dt.UTC)
        assert start2 == dt.datetime(2025, 7, 1, tzinfo=dt.UTC)

    def test_query_window_clips_version(self) -> None:
        """Query [start, end) clips a version that extends beyond the window."""
        tariff = Tariff(
            versions=[
                _tariff_version(
                    dt.datetime(2024, 1, 1, tzinfo=dt.UTC), end=dt.datetime(2026, 1, 1, tzinfo=dt.UTC), rate=10.0
                ),
            ]
        )
        segments = tariff.find_active_versions(
            dt.datetime(2025, 3, 1, tzinfo=dt.UTC),
            dt.datetime(2025, 9, 1, tzinfo=dt.UTC),
        )
        assert len(segments) == 1
        _, seg_start, seg_end = segments[0]
        assert seg_start == dt.datetime(2025, 3, 1, tzinfo=dt.UTC)
        assert seg_end == dt.datetime(2025, 9, 1, tzinfo=dt.UTC)

    def test_no_segments_when_query_falls_in_gap(self) -> None:
        """Querying a range that falls entirely within a gap returns nothing."""
        tariff = Tariff(
            versions=[
                _tariff_version(
                    dt.datetime(2025, 1, 1, tzinfo=dt.UTC), end=dt.datetime(2025, 3, 1, tzinfo=dt.UTC), rate=10.0
                ),
                _tariff_version(dt.datetime(2025, 6, 1, tzinfo=dt.UTC), rate=20.0),
            ]
        )
        segments = tariff.find_active_versions(
            dt.datetime(2025, 4, 1, tzinfo=dt.UTC),
            dt.datetime(2025, 5, 1, tzinfo=dt.UTC),
        )
        assert len(segments) == 0


# ---------------------------------------------------------------------------
# collect_version_frames – gaps produce no rows
# ---------------------------------------------------------------------------


class TestCollectVersionFramesWithGaps:
    def test_energy_cost_gap_returns_no_rows_for_gap_period(self) -> None:
        """get_energy_cost across a gap should only return rows for covered periods."""
        tariff = Tariff(
            versions=[
                _tariff_version(
                    dt.datetime(2025, 1, 1, tzinfo=dt.UTC), end=dt.datetime(2025, 2, 1, tzinfo=dt.UTC), rate=10.0
                ),
                _tariff_version(dt.datetime(2025, 4, 1, tzinfo=dt.UTC), rate=20.0),
            ]
        )
        result = tariff.get_energy_cost(
            start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
            end=dt.datetime(2025, 5, 1, tzinfo=dt.UTC),
            resolution=dt.timedelta(hours=1),
        )
        assert result is not None
        # January should have rate 10, April should have rate 20
        jan_rows = result[result["timestamp"] < pd.Timestamp("2025-02-01", tz=dt.UTC)]
        apr_rows = result[
            (result["timestamp"] >= pd.Timestamp("2025-04-01", tz=dt.UTC))
            & (result["timestamp"] < pd.Timestamp("2025-05-01", tz=dt.UTC))
        ]
        gap_rows = result[
            (result["timestamp"] >= pd.Timestamp("2025-02-01", tz=dt.UTC))
            & (result["timestamp"] < pd.Timestamp("2025-04-01", tz=dt.UTC))
        ]
        assert len(jan_rows) > 0
        assert len(apr_rows) > 0
        assert len(gap_rows) == 0
        assert (jan_rows["energy"] == 10.0).all()
        assert (apr_rows["energy"] == 20.0).all()


# ---------------------------------------------------------------------------
# from_yaml / from_dict on VersionedCollection base
# ---------------------------------------------------------------------------


class TestFromYamlFromDict:
    def test_tariff_from_dict(self) -> None:
        data = [
            {"start": "2025-01-01T00:00:00", "consumption": {"constant_cost": 100.0}},
        ]
        tariff = Tariff.from_dict(data)
        assert len(tariff.versions) == 1

    def test_tariff_from_yaml(self, tmp_path) -> None:
        path = tmp_path / "tariff.yml"
        path.write_text(
            "- start: 2025-01-01T00:00:00\n  consumption:\n    constant_cost: 100.0\n",
            encoding="utf-8",
        )
        tariff = Tariff.from_yaml(path)
        assert len(tariff.versions) == 1

    def test_from_dict_with_end(self) -> None:
        data = [
            {
                "start": "2025-01-01T00:00:00",
                "end": "2025-06-01T00:00:00",
                "consumption": {"constant_cost": 100.0},
            },
        ]
        tariff = Tariff.from_dict(data)
        assert tariff.versions[0].end == dt.datetime(2025, 6, 1, 0, 0)
