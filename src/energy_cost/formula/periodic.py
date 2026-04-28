from __future__ import annotations

import datetime as dt
from datetime import UTC

import pandas as pd
from pydantic import ConfigDict

from energy_cost.resolution import (
    Resolution,
    align_timestamps_to_tz,
    detect_resolution_and_range,
    resample_or_distribute,
    snap_billing_period,
    to_pandas_freq,
)

from .formula import Formula


class PeriodicFormula(Formula):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: str = "periodic"
    period: Resolution
    constant_cost: float

    def get_values(
        self,
        start: dt.datetime,
        end: dt.datetime,
        resolution: Resolution,
        timezone: dt.tzinfo = UTC,
    ) -> pd.DataFrame:
        raise NotImplementedError("Periodic formulas cannot be represented as time series. Use apply() instead.")

    def apply(
        self,
        data: pd.DataFrame,
        resolution: Resolution | None = None,
        timezone: dt.tzinfo = UTC,
        *,
        start: dt.datetime | None = None,
        end: dt.datetime | None = None,
    ) -> pd.DataFrame:
        data = align_timestamps_to_tz(data, timezone)
        if start is None or end is None or resolution is None:
            start, end, resolution = detect_resolution_and_range(data, resolution)

        period_freq = to_pandas_freq(self.period)

        # Snap to period boundary so all periods in [start, end) are covered.
        snapped_start, _ = snap_billing_period(start, end, period_freq)
        period_timestamps = pd.date_range(start=snapped_start, end=end, freq=period_freq, inclusive="left")

        coarse_df = pd.DataFrame({"timestamp": period_timestamps, "value": float(self.constant_cost)})
        return resample_or_distribute(coarse_df, self.period, resolution, start, end)
