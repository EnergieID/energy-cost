import datetime as dt
from abc import ABC, abstractmethod
from typing import ClassVar, Literal

import pandas as pd

from ..resolution import Resolution, resolution_divides, to_pandas_freq, to_pandas_offset

FillMode = Literal["ffill", "nan"]


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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_values(
        self,
        start: dt.datetime,
        end: dt.datetime,
        resolution: Resolution,
        out_of_range_fill: FillMode = "nan",
    ) -> pd.DataFrame:
        if not resolution_divides(self.resolution, resolution):
            raise ValueError(
                f"Requested resolution {resolution!r} is not a whole divisor "
                f"of the index resolution {self.resolution!r}."
            )

        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)

        # Build the desired output index
        freq = to_pandas_freq(resolution)
        target_index = pd.date_range(start=start_ts, end=end_ts, freq=freq, inclusive="left")

        # Fetch raw data from the concrete implementation.
        # When forward-filling, we may need the last known value that lies
        # *before* `start`, so we widen the look-back window by one native period.
        fetch_start = start_ts - to_pandas_offset(self.resolution)
        raw = self._get_values(fetch_start, end_ts)

        if raw.empty:
            df = pd.DataFrame({"timestamp": target_index})
            df["value"] = float("nan")
            return df

        # --- merge & fill --------------------------------------------------
        merged = pd.merge_asof(
            left=pd.DataFrame({"timestamp": target_index}),
            right=raw,
            on="timestamp",
            direction="backward",
        )

        if out_of_range_fill == "nan":
            # Slots that fall before the first raw data point get NaN from
            # merge_asof automatically.  For slots *after* the last raw point
            # we need to null them out explicitly, because merge_asof would
            # have forward-filled the last value into them.
            last_raw_ts = raw["timestamp"].max()
            # Determine the end of the last raw period
            last_period_end = last_raw_ts + to_pandas_offset(self.resolution)
            out_of_range = merged["timestamp"] >= last_period_end
            merged.loc[out_of_range, "value"] = float("nan")

        return merged

    @abstractmethod
    def _get_values(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
        """Return raw index values for the given range.

        Must return a ``pd.DataFrame`` with:
        - a ``"timestamp"`` column (``pd.Timestamp``, UTC-aware or naive,
          consistent with the inputs) marking the *start* of each period,
        - a ``"value"`` column (float) with the index value for that period in €/MWh.

        The returned rows should cover the requested range; returning a
        slightly wider range is fine — ``get_values`` will clip via the
        target index.
        """
