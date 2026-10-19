from __future__ import annotations

import datetime as dt
from datetime import UTC
from typing import Literal, Self

import pandas as pd
from pydantic import BaseModel, Field, PrivateAttr, model_validator

from energy_cost.index.index import Index
from energy_cost.resolution import Resolution, align_datetime_to_tz, to_pandas_freq

from .base import FormulaBase


class IndexAdder(BaseModel):
    index: str
    scalar: float
    _resolved: Index = PrivateAttr()

    @model_validator(mode="after")
    def _resolve_index(self) -> Self:
        """Look up the index in the current catalog, once."""
        try:
            self._resolved = Index.get(self.index)
        except KeyError:
            raise ValueError(f"unknown index '{self.index}'") from None
        return self

    def get_values(
        self, start: dt.datetime, end: dt.datetime, output_resolution: Resolution, timezone: dt.tzinfo = UTC
    ) -> pd.DataFrame:
        index_values = self._resolved.get_values(start, end, output_resolution, timezone)
        index_values = index_values.copy()
        index_values["value"] = index_values["value"] * self.scalar
        return index_values


class IndexFormula(FormulaBase):
    kind: Literal["index"] = "index"
    constant_cost: float = 0.0
    variable_costs: list[IndexAdder] = Field(default_factory=list)

    def get_values(
        self,
        start: dt.datetime,
        end: dt.datetime,
        output_resolution: Resolution,
        timezone: dt.tzinfo = UTC,
    ) -> pd.DataFrame:
        # Align start/end to the target timezone so all generated timestamps share one tz
        start = align_datetime_to_tz(start, timezone)
        end = align_datetime_to_tz(end, timezone)
        timestamps = pd.date_range(start=start, end=end, freq=to_pandas_freq(output_resolution), inclusive="left")
        values = pd.Series(self.constant_cost, index=timestamps, dtype=float)

        for variable_cost in self.variable_costs:
            vc = variable_cost.get_values(start, end, output_resolution, timezone)
            values = values.add(vc.set_index("timestamp")["value"])

        return pd.DataFrame({"timestamp": values.index, "value": values.to_numpy()})
