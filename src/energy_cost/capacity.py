from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from energy_cost.meter import Meter, TimeseriesFrame

from .resolution import Resolution, is_divisor, to_pandas_freq


class CapacityRule(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    measurement_period: Resolution
    billing_period: Resolution
    window_periods: int | None = None

    def apply(
        self,
        meter: Meter,
    ) -> Meter:
        data = meter.capacity

        if data is not None:
            resolution = data.resolution
        else:
            # calculate capacity from power data if capacity data is not provided
            data = meter.power
            resolution = data.resolution
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

        # 3. now, if resolution is divisor of billing period, we can resample to billing period using max (else throw error)
        if not is_divisor(self.billing_period, resolution):
            raise ValueError("Capacity data resolution must be a divisor of billing period for aggregation.")

        data = data.set_index("timestamp").resample(to_pandas_freq(self.billing_period)).max().reset_index()

        # 4. apply rolling average if window_periods is set
        if self.window_periods is not None:
            data["value"] = data["value"].rolling(window=self.window_periods, min_periods=1).mean()

        return Meter(
            power=meter.power,
            capacity=TimeseriesFrame(data, resolution=self.billing_period),
            direction=meter.direction,
            type=meter.type,
        )
