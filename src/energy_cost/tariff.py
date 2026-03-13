import bisect
import datetime as dt
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import yaml
from pydantic import BaseModel, BeforeValidator, Field

from .price_formula import PriceFormula


class MeterType(StrEnum):
    SINGLE_RATE = "single_rate"
    TOU_PEAK = "tou_peak"
    TOU_OFF_PEAK = "tou_off_peak"
    NIGHT_ONLY = "night_only"


class PowerDirection(StrEnum):
    CONSUMPTION = "consumption"
    INJECTION = "injection"


class CostType(StrEnum):
    ENERGY = "energy"
    WKK = "wkk"
    GREEN = "green"


class TimedPriceFormula(BaseModel):
    start: dt.datetime
    formula: PriceFormula

    def get_values(self, start: dt.datetime, end: dt.datetime, resolution: dt.timedelta) -> pd.DataFrame:
        """Get the cost values for the given time range and resolution in €/MWh."""
        actual_start = max(self.start, start)
        if actual_start >= end:
            return pd.DataFrame(columns=["timestamp", "value"])
        return self.formula.get_values(actual_start, end, resolution)


def _coerce_cost_type_dict(value: Any) -> Any:
    """Allow a bare list of TimedPriceFormula to be shorthand for ``{energy: <list>}``."""
    if isinstance(value, list):
        return {CostType.ENERGY: value}
    return value


CostTypeFormulas = Annotated[
    dict[CostType, list[TimedPriceFormula]],
    BeforeValidator(_coerce_cost_type_dict),
]


def _coerce_direction_dict(value: Any) -> Any:
    """Allow a bare CostTypeFormulas value to be shorthand for ``{consumption: <value>}``."""
    if isinstance(value, list):
        return {PowerDirection.CONSUMPTION: value}
    if isinstance(value, dict) and value.keys() <= {e.value for e in CostType}:
        return {PowerDirection.CONSUMPTION: value}
    return value


DirectionFormulas = Annotated[
    dict[PowerDirection, CostTypeFormulas],
    BeforeValidator(_coerce_direction_dict),
]


class Tariff(BaseModel):
    supplier: str
    product: str
    defaults: DirectionFormulas = Field(default_factory=dict)
    by_meter_type: dict[MeterType, DirectionFormulas] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Tariff":
        """Load a tariff definition from YAML."""
        with Path(path).open(encoding="utf-8") as file:
            raw_data = yaml.safe_load(file)
        return cls.model_validate(raw_data)

    def resolve_cost_formulas(
        self,
        meter_type: MeterType,
        direction: PowerDirection,
    ) -> CostTypeFormulas:
        """Merge defaults with meter-type-specific overrides for a given meter type and direction.

        Per-meter-type entries take precedence over defaults for the same cost type.
        """
        result: CostTypeFormulas = {}
        if direction in self.defaults:
            result.update(self.defaults[direction])
        if meter_type in self.by_meter_type and direction in self.by_meter_type[meter_type]:
            result.update(self.by_meter_type[meter_type][direction])
        return result

    def get_formulas(
        self,
        meter_type: MeterType,
        direction: PowerDirection,
        start: dt.datetime,
        end: dt.datetime,
        cost_type: CostType = CostType.ENERGY,
    ) -> list[TimedPriceFormula]:
        """Get the price formulas of the given type that are active during the given time range."""
        resolved = self.resolve_cost_formulas(meter_type, direction)
        timed_formulas = resolved.get(cost_type, [])
        start_index = max(0, bisect.bisect_right(timed_formulas, start, key=lambda c: c.start) - 1)
        end_index = bisect.bisect_right(timed_formulas, end, key=lambda c: c.start)
        return timed_formulas[start_index:end_index]

    @staticmethod
    def _compute_cost_series(
        formulas: list[TimedPriceFormula],
        start: dt.datetime,
        end: dt.datetime,
        resolution: dt.timedelta,
    ) -> pd.DataFrame:
        """Compute the cost time series for a list of consecutive TimedPriceFormulas."""
        ends = [f.start for f in formulas[1:]] + [end]
        df = formulas[0].get_values(start, ends[0], resolution)
        for formula, end_time in zip(formulas[1:], ends[1:], strict=True):
            formula_values = formula.get_values(formula.start, end_time, resolution)
            df = pd.concat([df, formula_values], ignore_index=True)
        return df

    def get_cost(
        self,
        meter_type: MeterType,
        direction: PowerDirection,
        start: dt.datetime,
        end: dt.datetime,
        resolution: dt.timedelta,
    ) -> pd.DataFrame:
        """Get the cost values for the given meter type and time range at the given resolution in €/MWh.

        Returns a DataFrame with a column per active cost type and a ``total`` column.
        """
        resolved = self.resolve_cost_formulas(meter_type, direction)
        if not resolved:
            raise ValueError(f"No formulas for meter type '{meter_type}' and direction '{direction}' found in tariff.")

        result: pd.DataFrame | None = None
        for cost_type in resolved:
            active = self.get_formulas(meter_type, direction, start, end, cost_type)
            if not active:
                continue
            df = self._compute_cost_series(active, start, end, resolution)
            df = df.rename(columns={"value": cost_type.value})
            result = df if result is None else result.merge(df, on="timestamp", how="outer")

        if result is None:
            raise ValueError(f"No formulas for meter type '{meter_type}' and direction '{direction}' found in tariff.")

        cost_columns = [col for col in result.columns if col != "timestamp"]
        result["total"] = result[cost_columns].sum(axis=1)
        return result
