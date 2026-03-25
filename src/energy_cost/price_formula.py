import datetime as dt

import pandas as pd
from pydantic import BaseModel, Field

from energy_cost.index.index import Index
from energy_cost.resolution import Resolution, to_pandas_freq


class IndexAdder(BaseModel):
    index: str
    # The scalar to multiply the index values by before adding to the cost in €/MWh.
    scalar: float

    def get_values(self, start: dt.datetime, end: dt.datetime, resolution: Resolution) -> pd.DataFrame:
        """Get the cost values for the given time range and resolution in €/MWh."""
        index = Index.from_name(self.index)
        index_values = index.get_values(start, end, resolution)
        index_values = index_values.copy()
        index_values["value"] = index_values["value"] * self.scalar
        return index_values


class PriceFormula(BaseModel):
    # The constant cost component of the price formula in €/MWh.
    constant_cost: float
    variable_costs: list[IndexAdder] = Field(default_factory=list)

    def get_values(self, start: dt.datetime, end: dt.datetime, resolution: Resolution) -> pd.DataFrame:
        """Get the cost values for the given time range and resolution in €/MWh."""
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
