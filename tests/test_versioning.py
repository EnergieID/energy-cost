from __future__ import annotations

import datetime as dt

import pandas as pd

from energy_cost.versioning import Versioned, VersionedCollection

# ---------------------------------------------------------------------------
# VersionedCollection – gap-aware find_active_versions
# ---------------------------------------------------------------------------


class TestFindActiveVersionsWithEnd:
    def test_explicit_end_limits_version_range(self) -> None:
        """A version with an explicit end should not cover time after that end."""
        col = VersionedCollection(
            [Versioned(start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC), end=dt.datetime(2025, 6, 1, tzinfo=dt.UTC))]
        )
        segments = col.find_active_versions(
            dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
            dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
        )
        assert len(segments) == 1
        _, seg_start, seg_end = segments[0]
        assert seg_start == dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
        assert seg_end == dt.datetime(2025, 6, 1, tzinfo=dt.UTC)

    def test_gap_between_versions_produces_no_segment(self) -> None:
        """A gap between two versions should not produce any segment."""
        col = VersionedCollection(
            [
                Versioned(start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC), end=dt.datetime(2025, 3, 1, tzinfo=dt.UTC)),
                Versioned(start=dt.datetime(2025, 6, 1, tzinfo=dt.UTC)),
            ]
        )
        segments = col.find_active_versions(
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
        col = VersionedCollection(
            [
                Versioned(start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC)),
                Versioned(start=dt.datetime(2025, 7, 1, tzinfo=dt.UTC)),
            ]
        )
        segments = col.find_active_versions(
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
        col = VersionedCollection(
            [Versioned(start=dt.datetime(2024, 1, 1, tzinfo=dt.UTC), end=dt.datetime(2026, 1, 1, tzinfo=dt.UTC))]
        )
        segments = col.find_active_versions(
            dt.datetime(2025, 3, 1, tzinfo=dt.UTC),
            dt.datetime(2025, 9, 1, tzinfo=dt.UTC),
        )
        assert len(segments) == 1
        _, seg_start, seg_end = segments[0]
        assert seg_start == dt.datetime(2025, 3, 1, tzinfo=dt.UTC)
        assert seg_end == dt.datetime(2025, 9, 1, tzinfo=dt.UTC)

    def test_no_segments_when_query_falls_in_gap(self) -> None:
        """Querying a range that falls entirely within a gap returns nothing."""
        col = VersionedCollection(
            [
                Versioned(start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC), end=dt.datetime(2025, 3, 1, tzinfo=dt.UTC)),
                Versioned(start=dt.datetime(2025, 6, 1, tzinfo=dt.UTC)),
            ]
        )
        segments = col.find_active_versions(
            dt.datetime(2025, 4, 1, tzinfo=dt.UTC),
            dt.datetime(2025, 5, 1, tzinfo=dt.UTC),
        )
        assert len(segments) == 0


# ---------------------------------------------------------------------------
# collect_version_frames – gaps produce no rows
# ---------------------------------------------------------------------------


class TestCollectVersionFramesWithGaps:
    def test_gap_produces_no_rows(self) -> None:
        """collect_version_frames across a gap should only return rows for covered periods."""
        col = VersionedCollection(
            [
                Versioned(start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC), end=dt.datetime(2025, 2, 1, tzinfo=dt.UTC)),
                Versioned(start=dt.datetime(2025, 4, 1, tzinfo=dt.UTC)),
            ]
        )

        def get_frame(version: Versioned, seg_start: dt.datetime, seg_end: dt.datetime) -> pd.DataFrame:
            timestamps = pd.date_range(seg_start, seg_end, freq="h", inclusive="left")
            return pd.DataFrame({"timestamp": timestamps})

        result = col.collect_version_frames(
            get_frame,
            start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
            end=dt.datetime(2025, 5, 1, tzinfo=dt.UTC),
        )
        assert result is not None
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


# ---------------------------------------------------------------------------
# from_yaml on VersionedCollection base
# ---------------------------------------------------------------------------


class TestFromYamlFromDict:
    def test_from_list(self) -> None:
        data = [{"start": "2025-01-01T00:00:00"}]
        col = VersionedCollection.model_validate(data)
        assert len(col.root) == 1

    def test_from_yaml(self, tmp_path) -> None:
        path = tmp_path / "versions.yml"
        path.write_text("- start: 2025-01-01T00:00:00\n", encoding="utf-8")
        col = VersionedCollection.from_yaml(path)
        assert len(col.root) == 1

    def test_from_list_with_end(self) -> None:
        data = [{"start": "2025-01-01T00:00:00", "end": "2025-06-01T00:00:00"}]
        col = VersionedCollection.model_validate(data)
        assert col.root[0].end == dt.datetime(2025, 6, 1, 0, 0)
