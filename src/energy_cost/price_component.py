import datetime as dt
from enum import StrEnum

import narwhals as nw
import polars as pl

from energy_cost.index.index import Index


class ComponentType(StrEnum):
    SINGLE_RATE = "single_rate"
    TOU_PEAK = "tou_peak"
    TOU_OFF_PEAK = "tou_off_peak"
    TOU_SHOULDER = "tou_shoulder"
    NIGHT_ONLY = "night_only"
    INJECTION = "injection"


class IndexBasedCost:
    index: str
    scalar: float

    def get_values(self, start: dt.datetime, end: dt.datetime, resolution: dt.timedelta) -> nw.DataFrame:
        """Get the cost values for the given time range and resolution."""
        index = Index.from_name(self.index)
        index_values = index.get_values(start, end, resolution)
        return index_values.with_columns(value=index_values["value"] * self.scalar)


class PriceComponent:
    type: ComponentType
    start: dt.datetime
    constant_cost: float
    variable_costs: list[IndexBasedCost]

    def get_values(self, start: dt.datetime, end: dt.datetime, resolution: dt.timedelta) -> nw.DataFrame:
        """Get the cost values for the given time range and resolution."""
        actual_start = max(self.start, start)

        df = nw.from_native(
            pl.DataFrame(
                {
                    "timestamp": pl.date_range(actual_start, end, resolution, closed="left"),
                    "value": self.constant_cost,
                }
            )
        )

        for variable_cost in self.variable_costs:
            variable_cost_values = variable_cost.get_values(actual_start, end, resolution)
            df = (
                df.join(variable_cost_values, on="timestamp", how="left")
                .with_columns(value=nw.col("value") + nw.col("value_right").fill_null(0))
                .drop("value_right")
            )

        return df
