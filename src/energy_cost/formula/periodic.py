from __future__ import annotations

import datetime as dt
from datetime import UTC

import isodate
import pandas as pd

from energy_cost.fractional_periods import Period
from energy_cost.resolution import (
    Resolution,
    align_timestamps_to_tz,
    detect_resolution_and_range,
    resample_or_distribute,
    snap_billing_period,
    to_pandas_freq,
)

from .formula import Formula

_PERIOD_RESOLUTIONS: dict[Period, Resolution] = {
    Period.HOURLY: dt.timedelta(hours=1),
    Period.DAILY: dt.timedelta(days=1),
    Period.MONTHLY: isodate.Duration(months=1),
    Period.YEARLY: isodate.Duration(years=1),
}


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

        period_resolution = _PERIOD_RESOLUTIONS[self.period]
        period_freq = to_pandas_freq(period_resolution)

        # Snap to period boundary so all periods in [start, end) are covered.
        snapped_start, _ = snap_billing_period(start, end, period_freq)
        period_timestamps = pd.date_range(start=snapped_start, end=end, freq=period_freq, inclusive="left")

        coarse_df = pd.DataFrame({"timestamp": period_timestamps, "value": float(self.constant_cost)})
        return resample_or_distribute(coarse_df, period_resolution, resolution, start, end)
