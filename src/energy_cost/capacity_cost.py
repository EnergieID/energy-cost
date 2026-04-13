from __future__ import annotations

import datetime as dt
from datetime import UTC

import pandas as pd
from pydantic import BaseModel, ConfigDict

from .formula import Formula
from .resolution import Resolution, detect_resolution, is_divisor, to_pandas_freq


class CapacityComponent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    measurement_period: Resolution
    billing_period: Resolution
    window_periods: int | None = None
    formula: Formula

    def apply(
        self,
        capacity_data: pd.DataFrame,
        timezone: dt.tzinfo = UTC,
    ) -> pd.DataFrame:
        # 1. if resolution of capacity data is divisor of measurement_period, resample to measurement_period using sum (else continue, assuming data in higher resolution is already aggregated in a correct way)
        resolution = detect_resolution(capacity_data["timestamp"])
        if is_divisor(self.measurement_period, resolution):
            capacity_data = (
                capacity_data.set_index("timestamp")
                .resample(to_pandas_freq(self.measurement_period))
                .sum()
                .reset_index()
            )
            resolution = self.measurement_period

        # 2. now, if resolution is divisor of billing period, we can resample to billing period using max (else throw error)
        if not is_divisor(self.billing_period, resolution):
            raise ValueError("Capacity data resolution must be a divisor of billing period for aggregation.")

        capacity_data = (
            capacity_data.set_index("timestamp").resample(to_pandas_freq(self.billing_period)).max().reset_index()
        )

        # 3. apply rolling average if window_periods is set
        if self.window_periods is not None:
            capacity_data["value"] = capacity_data["value"].rolling(window=self.window_periods, min_periods=1).mean()

        # 4. apply pricing formula to aggregated data
        return self.formula.apply(capacity_data, resolution=self.billing_period, timezone=timezone)
