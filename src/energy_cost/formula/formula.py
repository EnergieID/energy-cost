from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from typing import Any

import pandas as pd
from pydantic import BaseModel
from pydantic_core import core_schema

from energy_cost.resolution import Resolution, detect_resolution_and_range


class Formula(ABC, BaseModel):
    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler):
        # Subclasses get normal Pydantic schema — only the base class dispatches
        if cls.__name__ != "Formula":
            return handler(source_type)
        return core_schema.no_info_plain_validator_function(cls._coerce)

    @classmethod
    def _coerce(cls, value: Any) -> Formula:
        # only import here to avoid circular imports
        from . import IndexFormula, PeriodicFormula, ScheduledFormulas, TieredFormula

        if isinstance(value, Formula):
            return value
        if isinstance(value, dict):
            if value.get("kind") == "tiered" or "bands" in value:
                return TieredFormula.model_validate(value)
            if value.get("kind") == "periodic" or "period" in value:
                return PeriodicFormula.model_validate(value)
            if value.get("kind") == "scheduled" or "schedule" in value:
                return ScheduledFormulas.model_validate(value)
            return IndexFormula.model_validate(value)
        raise ValueError(f"Cannot coerce {value!r} to Formula")

    @abstractmethod
    def get_values(
        self,
        start: dt.datetime,
        end: dt.datetime,
        resolution: Resolution,
    ) -> pd.DataFrame:
        """Return time-indexed values for the formula."""

    def apply(
        self,
        data: pd.DataFrame,
        resolution: Resolution | None = None,
    ) -> pd.DataFrame:
        """Apply formula values to a dataframe of quantities and return a single value column."""
        if data.empty:
            return data.copy()
        start, end, resolution = detect_resolution_and_range(data, resolution)
        formula_values = self.get_values(start, end, resolution)

        result = data.copy()
        merged = result.merge(formula_values, on="timestamp", how="left", suffixes=("", "_formula"))
        result["value"] = merged["value_formula"].mul(merged["value"])
        return result
