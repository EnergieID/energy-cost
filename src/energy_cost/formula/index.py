from __future__ import annotations

import datetime as dt

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from energy_cost.index.index import Index
from energy_cost.resolution import Resolution, to_pandas_freq

from .formula import Formula


class IndexAdder(BaseModel):
    index: str
    scalar: float

    def get_values(self, start: dt.datetime, end: dt.datetime, resolution: Resolution) -> pd.DataFrame:
        index = Index.from_name(self.index)
        index_values = index.get_values(start, end, resolution)
        index_values = index_values.copy()
        index_values["value"] = index_values["value"] * self.scalar
        return index_values


class IndexFormula(Formula):
    model_config = ConfigDict(extra="forbid")

    kind: str = "index"
    constant_cost: float = 0.0
    variable_costs: list[IndexAdder] = Field(default_factory=list)

    def get_values(
        self,
        start: dt.datetime,
        end: dt.datetime,
        resolution: Resolution,
    ) -> pd.DataFrame:
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(start=start, end=end, freq=to_pandas_freq(resolution), inclusive="left"),
                "value": self.constant_cost,
            }
        )

        for variable_cost in self.variable_costs:
            variable_cost_values = variable_cost.get_values(start, end, resolution)
            df = df.merge(variable_cost_values, on="timestamp", how="left", suffixes=("", "_right"))
            df["value"] = df["value"] + df["value_right"]
            df = df.drop(columns=["value_right"])

        return df
