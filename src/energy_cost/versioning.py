import bisect
import datetime as dt
from collections.abc import Callable

import pandas as pd
from pydantic import BaseModel

from energy_cost.resolution import align_datetime_to_tz


class Versioned(BaseModel):
    start: dt.datetime


class VersionedCollection[V: Versioned](BaseModel):
    versions: list[V]

    def find_active_versions(
        self,
        start: dt.datetime,
        end: dt.datetime,
        timezone: dt.tzinfo = dt.UTC,
    ) -> list[tuple[V, dt.datetime, dt.datetime]]:
        """Return each segment that overlaps ``[start, end)`` together with the effective sub-range."""

        def norm(v: V) -> dt.datetime:
            return align_datetime_to_tz(v.start, timezone)

        start_index = max(0, bisect.bisect_right(self.versions, start, key=norm) - 1)
        end_index = bisect.bisect_right(self.versions, end, key=norm)
        segments = self.versions[start_index:end_index]
        if not segments:
            return []

        starts = [max(norm(segment), start) for segment in segments]
        ends = [norm(segment) for segment in segments[1:]] + [end]
        return list(zip(segments, starts, ends, strict=True))

    def collect_version_frames(
        self,
        get_frame: Callable[[V, dt.datetime, dt.datetime], pd.DataFrame | None],
        start: dt.datetime,
        end: dt.datetime,
        timezone: dt.tzinfo = dt.UTC,
    ) -> pd.DataFrame | None:
        segments = self.find_active_versions(start, end, timezone)
        frames = [
            df[(df["timestamp"] >= seg_start) & (df["timestamp"] < seg_end)]
            for version, seg_start, seg_end in segments
            if (df := get_frame(version, seg_start, seg_end)) is not None and not df.empty
        ]
        if not frames:
            return None
        return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
