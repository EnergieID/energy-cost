from __future__ import annotations

import datetime as dt
from datetime import UTC
from typing import Literal

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
        data: pd.DataFrame,
        timezone: dt.tzinfo = UTC,
        unit: Literal["MW", "MWh"] = "MWh",
    ) -> pd.DataFrame:
        # The data is expected to have a "timestamp" column and a "value" column.
        # It is expected to be consumption data in MWh at a resolution that is a divisor of the measurment_period
        # If data is given in MW, it is expected to be the maximum average capacity used in each measurment_period and should be given in a resolution that is a divisor of the billing_period.

        resolution = detect_resolution(data["timestamp"])

        if unit == "MWh":
            # 1. if resolution of capacity data is divisor of measurement_period, resample to measurement_period using sum
            if is_divisor(self.measurement_period, resolution):
                data = data.set_index("timestamp").resample(to_pandas_freq(self.measurement_period)).sum().reset_index()
                resolution = self.measurement_period
            else:
                raise ValueError(
                    f"Data resolution must be a divisor of measurement period {self.measurement_period} for aggregation."
                )
            # 2. convert MWh to MW by dividing by the number of hours in the resolution (eg divide by 0.25 for 15min data)
            hours_in_resolution = resolution.total_seconds() / 3600
            data["value"] = data["value"] / hours_in_resolution
            unit = "MW"

        # 3. now, if resolution is divisor of billing period, we can resample to billing period using max (else throw error)
        if not is_divisor(self.billing_period, resolution):
            raise ValueError("Capacity data resolution must be a divisor of billing period for aggregation.")

        data = data.set_index("timestamp").resample(to_pandas_freq(self.billing_period)).max().reset_index()

        # 4. apply rolling average if window_periods is set
        if self.window_periods is not None:
            data["value"] = data["value"].rolling(window=self.window_periods, min_periods=1).mean()

        # 5. apply pricing formula to aggregated data
        return self.formula.apply(data, resolution=self.billing_period, timezone=timezone)
