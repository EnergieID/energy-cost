import datetime as dt
from collections.abc import Callable
from datetime import UTC
from typing import Annotated, Any, Literal

import pandas as pd
from pydantic import BeforeValidator, Field

from energy_cost.versioning import Versioned

from .capacity_cost import CapacityComponent
from .formula import Formula, PeriodicFormula
from .meter import MeterType, PowerDirection
from .resolution import Resolution

_METER_KEYS = {e.value for e in MeterType}


def _coerce_named_formulas(value: Any) -> Any:
    """Allow a bare Formula dict to be shorthand for ``{total: it}``."""
    try:
        return {"total": Formula.model_validate(value)}
    except ValueError:
        return value


NamedFormulas = Annotated[
    dict[str, Formula],
    BeforeValidator(_coerce_named_formulas),
]


def _coerce_meter_formulas(value: Any) -> Any:
    """Coerce shorthand MeterFormulas values."""
    if not isinstance(value, dict):
        return value
    if not any(key in value for key in _METER_KEYS):
        # No meter keys, assume it's a shorthand for ALL meters
        return {MeterType.ALL.value: _coerce_named_formulas(value)}
    return value


MeterFormulas = Annotated[
    dict[str, NamedFormulas],
    BeforeValidator(_coerce_meter_formulas),
]


class TariffVersion(Versioned):
    injection: MeterFormulas = Field(default_factory=dict)
    consumption: MeterFormulas = Field(default_factory=dict)
    capacity: CapacityComponent | None = None
    periodic: dict[str, PeriodicFormula] = Field(default_factory=dict)

    def _resolve_energy_formulas(
        self,
        meter_type: MeterType,
        direction: PowerDirection,
    ) -> dict[str, Formula]:
        direction_formulas: MeterFormulas = getattr(self, direction)
        result: dict[str, Formula] = {}
        if MeterType.ALL in direction_formulas:
            result.update(direction_formulas[MeterType.ALL])
        if meter_type in direction_formulas:
            result.update(direction_formulas[meter_type])
        return result

    def _combine_energy_formulas(
        self,
        meter_type: MeterType,
        direction: PowerDirection,
        get_df: Callable[[Formula], pd.DataFrame],
    ) -> pd.DataFrame | None:
        resolved = self._resolve_energy_formulas(meter_type, direction)
        result: pd.DataFrame | None = None
        for cost_type, formula in resolved.items():
            df = get_df(formula)
            if df.empty:
                continue
            df = df.rename(columns={"value": cost_type})
            result = df if result is None else result.merge(df, on="timestamp", how="outer")

        if result is None:
            return None

        cost_columns = [col for col in result.columns if col not in ("timestamp", "total")]
        if cost_columns:
            result["total"] = result[cost_columns].sum(axis=1)
        return result

    def get_energy_cost(
        self,
        start: dt.datetime,
        end: dt.datetime,
        resolution: Resolution,
        meter_type: MeterType,
        direction: PowerDirection,
        timezone: dt.tzinfo = UTC,
    ) -> pd.DataFrame | None:
        """Get energy cost rates in €/MWh. Returns None if no formulas are configured."""
        return self._combine_energy_formulas(
            meter_type,
            direction,
            lambda formula: formula.get_values(start, end, resolution, timezone),
        )

    def apply_energy_cost(
        self,
        data: pd.DataFrame,
        meter_type: MeterType,
        direction: PowerDirection,
        timezone: dt.tzinfo = UTC,
    ) -> pd.DataFrame | None:
        """Apply energy cost formulas to quantity data, returning costs in €."""
        return self._combine_energy_formulas(
            meter_type,
            direction,
            lambda formula: formula.apply(data, timezone=timezone),
        )

    def apply_capacity_cost(
        self,
        capacity_data: pd.DataFrame,
        timezone: dt.tzinfo = UTC,
        unit: Literal["MW", "MWh"] = "MWh",
    ) -> pd.DataFrame | None:
        if self.capacity is None:
            return None
        return self.capacity.apply(capacity_data, timezone=timezone, unit=unit)

    def get_periodic_cost(self, start: dt.datetime, end: dt.datetime, timezone: dt.tzinfo = UTC) -> dict[str, float]:
        return {name: entry.get_cost_for_interval(start, end, timezone) for name, entry in self.periodic.items()}
