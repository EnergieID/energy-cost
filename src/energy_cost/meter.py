from enum import StrEnum

import pandas as pd
from pydantic import BaseModel, ConfigDict


class CostGroup(StrEnum):
    CONSUMPTION = "consumption"
    INJECTION = "injection"
    CAPACITY = "capacity"
    FIXED = "fixed"
    TOTAL = "total"


class MeterType(StrEnum):
    SINGLE_RATE = "single_rate"
    TOU_PEAK = "tou_peak"
    TOU_OFFPEAK = "tou_offpeak"
    NIGHT_ONLY = "night_only"
    ALL = "all"  # The "all" meter type is used for formulas that apply to all meter types. It should not be used in actual Meter instances.


class PowerDirection(StrEnum):
    CONSUMPTION = "consumption"
    INJECTION = "injection"


class Meter(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    direction: PowerDirection = PowerDirection.CONSUMPTION
    type: MeterType = MeterType.SINGLE_RATE
    data: pd.DataFrame


def as_single_meter(meters: list[Meter], direction: PowerDirection) -> Meter:
    direction_meters = [m for m in meters if m.direction == direction]
    if not direction_meters:
        raise ValueError(f"No meters found for direction {direction}")
    if len(direction_meters) > 1:
        combined_data = pd.concat([m.data for m in direction_meters]).groupby("timestamp", as_index=False).sum()
        return Meter(direction=direction, type=MeterType.SINGLE_RATE, data=combined_data)
    return direction_meters[0]
