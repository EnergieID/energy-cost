import datetime as dt
from enum import StrEnum
from typing import Annotated, Any

import pandas as pd
from pydantic import BaseModel, BeforeValidator, Field

from .capacity_cost import CapacityComponent
from .formula import Formula, PeriodicFormula
from .resolution import Resolution


class MeterType(StrEnum):
    SINGLE_RATE = "single_rate"
    TOU_PEAK = "tou_peak"
    TOU_OFFPEAK = "tou_offpeak"
    NIGHT_ONLY = "night_only"
    ALL = "all"  # The "all" meter type is used for formulas that apply to all meter types.


class PowerDirection(StrEnum):
    CONSUMPTION = "consumption"
    INJECTION = "injection"


class CostType(StrEnum):
    ENERGY = "energy"
    # Combined Heat and Power certificates, a type of subsidy for efficient cogeneration plants
    CHP_CERTIFICATES = "chp_certificates"
    # Renewable Energy certificates, a type of subsidy for renewable energy generation
    RENEWABLE_CERTIFICATES = "renewable_certificates"


_COST_TYPE_VALUES = {e.value for e in CostType}
_METER_KEYS = {e.value for e in MeterType}


def _coerce_cost_type_formulas(value: Any) -> Any:
    """Allow a bare Formula dict to be shorthand for ``{energy: it}``."""
    if isinstance(value, dict) and not value.keys() <= _COST_TYPE_VALUES:
        return {CostType.ENERGY: value}
    return value


CostTypeFormulas = Annotated[
    dict[CostType, Formula],
    BeforeValidator(_coerce_cost_type_formulas),
]


def _coerce_meter_formulas(value: Any) -> Any:
    """Coerce shorthand MeterFormulas values."""
    if not isinstance(value, dict):
        return value
    if value.keys() <= _COST_TYPE_VALUES:
        return {MeterType.ALL: value}
    if not value.keys() <= _METER_KEYS:
        return {MeterType.ALL: {CostType.ENERGY: value}}
    return value


MeterFormulas = Annotated[
    dict[str, CostTypeFormulas],
    BeforeValidator(_coerce_meter_formulas),
]


class TariffVersion(BaseModel):
    start: dt.datetime
    injection: MeterFormulas = Field(default_factory=dict)
    consumption: MeterFormulas = Field(default_factory=dict)
    capacity: CapacityComponent | None = None
    periodic: dict[str, PeriodicFormula] = Field(default_factory=dict)

    def resolve_cost_formulas(
        self,
        meter_type: MeterType,
        direction: PowerDirection,
    ) -> dict[CostType, Formula]:
        direction_formulas: MeterFormulas = getattr(self, direction)
        result: dict[CostType, Formula] = {}
        if MeterType.ALL in direction_formulas:
            result.update(direction_formulas[MeterType.ALL])
        if meter_type in direction_formulas:
            result.update(direction_formulas[meter_type])
        return result

    def get_cost(
        self,
        start: dt.datetime,
        end: dt.datetime,
        resolution: Resolution,
        meter_type: MeterType,
        direction: PowerDirection,
    ) -> pd.DataFrame:
        resolved = self.resolve_cost_formulas(meter_type, direction)
        result: pd.DataFrame | None = None
        for cost_type in resolved:
            df = resolved[cost_type].get_values(start, end, resolution)
            if df.empty:
                continue
            df = df.rename(columns={"value": cost_type.value})
            result = df if result is None else result.merge(df, on="timestamp", how="outer")

        if result is None:
            raise ValueError(f"No formulas for meter type '{meter_type}' and direction '{direction}' found in tariff.")

        cost_columns = [col for col in result.columns if col != "timestamp"]
        result["total"] = result[cost_columns].sum(axis=1)
        return result

    def apply_capacity_cost(self, capacity_data: pd.DataFrame) -> pd.DataFrame:
        if self.capacity is None:
            return pd.DataFrame(columns=["timestamp", "value"])
        return self.capacity.apply(capacity_data)

    def get_periodic_cost(self, start: dt.datetime, end: dt.datetime) -> dict[str, float]:
        return {name: entry.get_cost_for_interval(start, end) for name, entry in self.periodic.items()}
