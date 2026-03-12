import bisect
import datetime as dt
from collections import defaultdict
from enum import StrEnum
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel, Field

from .price_formula import PriceFormula


class MeterType(StrEnum):
    SINGLE_RATE = "single_rate"
    TOU_PEAK = "tou_peak"
    TOU_OFF_PEAK = "tou_off_peak"
    NIGHT_ONLY = "night_only"


class PowerDirection(StrEnum):
    CONSUMPTION = "consumption"
    INJECTION = "injection"


class TimedPriceFormula(BaseModel):
    start: dt.datetime
    formula: PriceFormula

    def get_values(self, start: dt.datetime, end: dt.datetime, resolution: dt.timedelta) -> pd.DataFrame:
        """Get the cost values for the given time range and resolution in €/MWh."""
        actual_start = max(self.start, start)
        if actual_start >= end:
            return pd.DataFrame(columns=["timestamp", "value"])
        return self.formula.get_values(actual_start, end, resolution)


class Tariff(BaseModel):
    supplier: str
    product: str
    by_meter_type: dict[MeterType, dict[PowerDirection, list[TimedPriceFormula]]] = Field(
        default_factory=lambda: defaultdict(lambda: defaultdict(list))
    )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Tariff":
        """Load a tariff definition from YAML."""
        with Path(path).open(encoding="utf-8") as file:
            raw_data = yaml.safe_load(file)
        tariff = cls.model_validate(raw_data)
        return tariff

    def get_formulas(
        self,
        meter_type: MeterType,
        direction: PowerDirection,
        start: dt.datetime,
        end: dt.datetime,
    ) -> list[TimedPriceFormula]:
        """Get the price formulas of the given type that are active during the given time range."""
        timed_formulas = self.by_meter_type[meter_type][direction]
        start_index = max(0, bisect.bisect_right(timed_formulas, start, key=lambda c: c.start) - 1)
        end_index = bisect.bisect_right(timed_formulas, end, key=lambda c: c.start)
        return timed_formulas[start_index:end_index]

    def get_cost(
        self,
        meter_type: MeterType,
        direction: PowerDirection,
        start: dt.datetime,
        end: dt.datetime,
        resolution: dt.timedelta,
    ) -> pd.DataFrame:
        """Get the cost values for the given meter type and time range at the given resolution in €/MWh."""
        formulas = self.get_formulas(meter_type, direction, start, end)
        if not formulas:
            raise ValueError(f"No meters of type '{meter_type}' and direction '{direction}' found in tariff.")

        ends = [formula.start for formula in formulas[1:]] + [end]
        df = formulas[0].get_values(start, ends[0], resolution)

        for formula, end_time in zip(formulas[1:], ends[1:], strict=True):
            formula_values = formula.get_values(formula.start, end_time, resolution)
            df = pd.concat([df, formula_values], ignore_index=True)

        return df
