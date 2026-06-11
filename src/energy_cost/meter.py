import datetime as dt
from enum import StrEnum

import pandas as pd
from pydantic import BaseModel, ConfigDict

from energy_cost.resolution import Resolution, align_timestamps_to_tz, detect_resolution_and_range


class CostGroup(StrEnum):
    CONSUMPTION = "consumption"
    INJECTION = "injection"
    CAPACITY = "capacity"
    FIXED = "fixed"


class MeterType(StrEnum):
    SINGLE_RATE = "single_rate"
    NIGHT_ONLY = "night_only"


class PowerDirection(StrEnum):
    CONSUMPTION = "consumption"
    INJECTION = "injection"


class TariffCategory(StrEnum):
    SUPPLIER = "supplier"
    DISTRIBUTOR = "distributor"
    FEES = "fees"
    TAXES = "taxes"


class TimeseriesFrame(pd.DataFrame):
    _start: dt.datetime | None = None
    _end: dt.datetime | None = None
    _resolution: Resolution | None = None

    def __init__(self, *args, resolution: Resolution | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._resolution = resolution

    def _calc_properties(self):
        self._start, self._end, self._resolution = detect_resolution_and_range(self, self._resolution)

    @property
    def start(self) -> dt.datetime:
        if self._start is None:
            self._calc_properties()
        assert self._start is not None
        return self._start

    @property
    def end(self) -> dt.datetime:
        if self._end is None:
            self._calc_properties()
        assert self._end is not None
        return self._end

    @property
    def resolution(self) -> Resolution:
        if self._resolution is None:
            self._calc_properties()
        assert self._resolution is not None
        return self._resolution

    def align_to_timezone(self, timezone: dt.tzinfo) -> "TimeseriesFrame":
        return TimeseriesFrame(align_timestamps_to_tz(self, timezone), resolution=self._resolution)


class Meter(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    direction: PowerDirection = PowerDirection.CONSUMPTION
    type: MeterType = MeterType.SINGLE_RATE
    measurements: TimeseriesFrame
    capacity: TimeseriesFrame | None = None

    def align_to_timezone(self, timezone: dt.tzinfo) -> "Meter":
        return Meter(
            direction=self.direction,
            type=self.type,
            measurements=self.measurements.align_to_timezone(timezone),
            capacity=self.capacity.align_to_timezone(timezone) if self.capacity is not None else None,
        )
