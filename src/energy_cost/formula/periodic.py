from __future__ import annotations

import datetime as dt
from datetime import UTC

import pandas as pd

from energy_cost.fractional_periods import Period
from energy_cost.resolution import (
    Resolution,
    align_timestamps_to_tz,
    detect_resolution_and_range,
    to_pandas_freq,
    to_pandas_offset,
)

from .formula import Formula


class PeriodicFormula(Formula):
    kind: str = "periodic"
    period: Period
    constant_cost: float

    def get_values(
        self,
        start: dt.datetime,
        end: dt.datetime,
        resolution: Resolution,
        timezone: dt.tzinfo = UTC,
    ) -> pd.DataFrame:
        raise NotImplementedError("Periodic formulas cannot be represented as time series. Use apply() instead.")

    def get_cost_for_interval(self, start: dt.datetime, end: dt.datetime, timezone: dt.tzinfo = UTC) -> float:
        return self.constant_cost * self.period.fractional_periods(start, end, timezone)

    def apply(
        self,
        data: pd.DataFrame,
        resolution: Resolution | None = None,
        timezone: dt.tzinfo = UTC,
    ) -> pd.DataFrame:
        data = align_timestamps_to_tz(data, timezone)
        start, end, resolution = detect_resolution_and_range(data, resolution)
        timestamps = pd.date_range(start=start, end=end, freq=to_pandas_freq(resolution), inclusive="left")
        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "value": timestamps.to_series().apply(
                    lambda ts: self.get_cost_for_interval(ts, ts + to_pandas_offset(resolution), timezone)
                ),
            }
        )
