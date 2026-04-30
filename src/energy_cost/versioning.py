import bisect
import datetime as dt
from collections.abc import Callable
from pathlib import Path
from typing import Any, Self

import pandas as pd
import yaml
from pydantic import BaseModel

from energy_cost.resolution import align_datetime_to_tz


class Versioned(BaseModel):
    start: dt.datetime
    end: dt.datetime | None = None


class VersionedCollection[V: Versioned](BaseModel):
    versions: list[V]

    @classmethod
    def from_dict(cls, data: list[dict[str, Any]]) -> Self:
        """Create from a list of version dicts (the same structure found in YAML files)."""
        return cls.model_validate({"versions": data})

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Load from a YAML file containing a list of version entries."""
        with Path(path).open(encoding="utf-8") as file:
            raw_data = yaml.safe_load(file)
        return cls.from_dict(raw_data)

    def find_active_versions(
        self,
        start: dt.datetime,
        end: dt.datetime,
        timezone: dt.tzinfo = dt.UTC,
    ) -> list[tuple[V, dt.datetime, dt.datetime]]:
        """Return each segment that overlaps ``[start, end)`` together with the effective sub-range."""

        def norm_start(v: V) -> dt.datetime:
            return align_datetime_to_tz(v.start, timezone)

        def norm_end(v: V) -> dt.datetime | None:
            return align_datetime_to_tz(v.end, timezone) if v.end is not None else None

        start_index = max(0, bisect.bisect_right(self.versions, start, key=norm_start) - 1)
        end_index = bisect.bisect_right(self.versions, end, key=norm_start)
        candidates = self.versions[start_index:end_index]
        if not candidates:
            return []

        result: list[tuple[V, dt.datetime, dt.datetime]] = []
        for i, version in enumerate(candidates):
            v_start = norm_start(version)
            v_end = norm_end(version)
            if v_end is None:
                # Fall back to next version's start, then query end
                v_end = norm_start(candidates[i + 1]) if i + 1 < len(candidates) else end

            # Clip to query window
            seg_start = max(v_start, start)
            seg_end = min(v_end, end)

            if seg_start < seg_end:
                result.append((version, seg_start, seg_end))

        return result

    def collect_version_frames(
        self,
        get_frame: Callable[[V, dt.datetime, dt.datetime], pd.DataFrame | None],
        start: dt.datetime,
        end: dt.datetime,
        timezone: dt.tzinfo = dt.UTC,
    ) -> pd.DataFrame | None:
        segments = self.find_active_versions(start, end, timezone)
        frames = [
            df
            for version, seg_start, seg_end in segments
            if (df := get_frame(version, seg_start, seg_end)) is not None and not df.empty
        ]
        if not frames:
            return None
        return (
            pd.concat(frames, ignore_index=True)
            .groupby("timestamp")
            .sum(numeric_only=True)
            .reset_index()
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
