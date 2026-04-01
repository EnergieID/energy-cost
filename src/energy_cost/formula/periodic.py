from __future__ import annotations

import datetime as dt

import pandas as pd

from energy_cost.fractional_periods import Period
from energy_cost.resolution import Resolution, detect_resolution_and_range, to_pandas_freq

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
    ) -> pd.DataFrame:
        timestamps = pd.date_range(start=start, end=end, freq=to_pandas_freq(resolution), inclusive="left")
        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "value": timestamps.to_series().apply(lambda ts: self.get_cost_for_interval(ts, ts + resolution)),
            }
        )

    def get_cost_for_interval(self, start: dt.datetime, end: dt.datetime) -> float:
        return self.constant_cost * self.period.fractional_periods(start, end)

    def apply(
        self,
        data: pd.DataFrame,
        resolution: Resolution | None = None,
    ) -> pd.DataFrame:
        start, end, resolution = detect_resolution_and_range(data, resolution)
        return self.get_values(start, end, resolution)
