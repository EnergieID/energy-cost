import datetime as dt
from abc import ABC, abstractmethod
from typing import ClassVar

import pandas as pd

from ..resolution import Resolution, is_divisor, to_pandas_freq, to_pandas_offset


class Index(ABC):
    """An index used for calculating energy costs (€/MWh)."""

    indexes: ClassVar[dict[str, "Index"]] = {}

    @classmethod
    def register(cls, name: str, index: "Index") -> None:
        """Register an index instance by name."""
        cls.indexes[name] = index

    @staticmethod
    def from_name(name: str) -> "Index":
        """Return the named index instance."""
        if name not in Index.indexes:
            raise ValueError(f"Unsupported index: {name}")
        return Index.indexes[name]

    def __init__(self, resolution: Resolution) -> None:
        self.resolution = resolution

    def get_values(
        self,
        start: dt.datetime,
        end: dt.datetime,
        resolution: Resolution,
    ) -> pd.DataFrame:
        if not is_divisor(self.resolution, resolution):
            raise ValueError(
                f"Requested resolution {resolution!r} is not a whole divisor "
                f"of the index resolution {self.resolution!r}."
            )

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        requested_freq = to_pandas_freq(resolution)
        native_resolution_offset = to_pandas_offset(self.resolution)

        target_index = pd.date_range(start=start_ts, end=end_ts, freq=requested_freq, inclusive="left")

        # When forward-filling, we may need the last known value that lies
        # *before* `start`, so we widen the look-back window by one native resolution period.
        fetch_start = start_ts - native_resolution_offset
        raw = self._get_values(fetch_start, end_ts)

        # explicitly add a timestamp one native resolution after the last raw timestamp with value `nan`
        # This way, they are not forward-filled with the last known value, but correctly marked as out-of-range.
        if not raw.empty:
            last_raw_ts = raw["timestamp"].max()
            last_period_end = last_raw_ts + native_resolution_offset
            raw = pd.concat(
                [raw, pd.DataFrame({"timestamp": [last_period_end], "value": [float("nan")]})], ignore_index=True
            )

        merged = pd.merge_asof(
            left=pd.DataFrame({"timestamp": target_index}),
            right=raw,
            on="timestamp",
            direction="backward",
        )

        return merged

    @abstractmethod
    def _get_values(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Return raw index values for the given range.

        Must return a ``pd.DataFrame`` with:
        - a ``"timestamp"`` column (``pd.Timestamp``, UTC-aware or naive,
          consistent with the inputs) marking the *start* of each period,
        - a ``"value"`` column (float) with the index value for that period in €/MWh.
        """
