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
    TOU_PEAK = "tou_peak"
    TOU_OFFPEAK = "tou_offpeak"
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

    @resolution.setter
    def resolution(self, value: Resolution) -> None:
        self._resolution = value


class Meter(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    direction: PowerDirection = PowerDirection.CONSUMPTION
    type: MeterType = MeterType.SINGLE_RATE
    power: TimeseriesFrame
    capacity: TimeseriesFrame | None = None

    def align_to_timezone(self, timezone: dt.tzinfo) -> "Meter":
        return Meter(
            direction=self.direction,
            type=self.type,
            power=TimeseriesFrame(align_timestamps_to_tz(self.power, timezone)),
            capacity=TimeseriesFrame(align_timestamps_to_tz(self.capacity, timezone))
            if self.capacity is not None
            else None,
        )
