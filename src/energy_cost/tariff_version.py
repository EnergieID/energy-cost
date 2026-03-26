import datetime as dt
from enum import StrEnum
from typing import Annotated, Any

import pandas as pd
from pydantic import BaseModel, BeforeValidator, Field

from .periodic_cost import PeriodicCost
from .price_formula import PriceFormula
from .scheduled_formula import ScheduledPriceFormulas


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
    """Allow a bare PriceFormula dict to be shorthand for ``{energy: it}``.

    Detected when the dict's keys are not all recognised CostType values.
    """
    if isinstance(value, dict) and not value.keys() <= _COST_TYPE_VALUES:
        return {CostType.ENERGY: value}
    return value


CostTypeFormulas = Annotated[
    dict[CostType, PriceFormula | ScheduledPriceFormulas],
    BeforeValidator(_coerce_cost_type_formulas),
]


def _coerce_meter_formulas(value: Any) -> Any:
    """Coerce shorthand MeterFormulas values.

    - List of scheduled formulas → ``{"all": {energy: it}}``
    - CostType-keyed dict → ``{"all": it}``
    - Dict with keys not recognised as meter keys → bare PriceFormula → ``{"all": {energy: it}}``
    """
    if isinstance(value, list):
        return {MeterType.ALL: {CostType.ENERGY: value}}
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
    periodic: dict[str, PeriodicCost] = Field(default_factory=dict)

    def resolve_cost_formulas(
        self,
        meter_type: MeterType,
        direction: PowerDirection,
    ) -> dict[CostType, PriceFormula | ScheduledPriceFormulas]:
        """Merge the ``all`` defaults with any meter-type-specific overrides for a direction.

        Per-meter-type entries take precedence over ``all`` entries for the same cost type.
        """
        direction_formulas: MeterFormulas = getattr(self, direction)
        result: dict[CostType, PriceFormula | ScheduledPriceFormulas] = {}
        if MeterType.ALL in direction_formulas:
            result.update(direction_formulas[MeterType.ALL])
        if meter_type in direction_formulas:
            result.update(direction_formulas[meter_type])
        return result

    def get_cost(
        self,
        start: dt.datetime,
        end: dt.datetime,
        resolution: dt.timedelta,
        meter_type: MeterType,
        direction: PowerDirection,
    ) -> pd.DataFrame:
        """Compute cost time series per cost type for the given interval.

        Returns a mapping of cost type to a DataFrame with ``timestamp`` and ``value`` columns.
        """
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

    def get_periodic_cost(self, start: dt.datetime, end: dt.datetime) -> dict[str, float]:
        """Get the prorated periodic costs for the given interval."""
        return {name: entry.get_cost_for_interval(start, end) for name, entry in self.periodic.items()}
