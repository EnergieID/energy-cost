import datetime as dt
from enum import StrEnum

import pandas as pd
from pydantic import BaseModel, Field

from energy_cost.index.index import Index


class ComponentType(StrEnum):
    SINGLE_RATE = "single_rate"
    TOU_PEAK = "tou_peak"
    TOU_OFF_PEAK = "tou_off_peak"
    TOU_SHOULDER = "tou_shoulder"
    NIGHT_ONLY = "night_only"
    INJECTION = "injection"


class IndexBasedCost(BaseModel):
    index: str
    scalar: float

    def get_values(self, start: dt.datetime, end: dt.datetime, resolution: dt.timedelta) -> pd.DataFrame:
        """Get the cost values for the given time range and resolution."""
        index = Index.from_name(self.index)
        index_values = index.get_values(start, end, resolution)
        index_values = index_values.copy()
        index_values["value"] = index_values["value"] * self.scalar
        return index_values


class PriceComponent(BaseModel):
    start: dt.datetime
    constant_cost: float
    variable_costs: list[IndexBasedCost] = Field(default_factory=list)

    def get_values(self, start: dt.datetime, end: dt.datetime, resolution: dt.timedelta) -> pd.DataFrame:
        """Get the cost values for the given time range and resolution."""
        actual_start = max(self.start, start)

        df = pd.DataFrame(
            {
                "timestamp": pd.date_range(start=actual_start, end=end, freq=resolution, inclusive="left"),
                "value": self.constant_cost,
            }
        )

        for variable_cost in self.variable_costs:
            variable_cost_values = variable_cost.get_values(actual_start, end, resolution)
            df = df.merge(variable_cost_values, on="timestamp", how="left", suffixes=("", "_right"))
            df["value"] = df["value"] + df["value_right"].fillna(0)
            df = df.drop(columns=["value_right"])

        return df
