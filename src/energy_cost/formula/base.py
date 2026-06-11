from __future__ import annotations

import datetime as dt
from abc import ABC
from datetime import UTC

import pandas as pd
from pydantic import BaseModel

from energy_cost.meter import Meter, TimeseriesFrame
from energy_cost.resolution import Resolution, redistribute_to_resolution


class FormulaBase(ABC, BaseModel):
    capacity_based: bool = False

    def get_values(
        self,
        start: dt.datetime,
        end: dt.datetime,
        output_resolution: Resolution,
        timezone: dt.tzinfo = UTC,
    ) -> pd.DataFrame:
        """Return time-indexed values for the formula."""
        raise NotImplementedError("This formula does not support get_values(). Use apply() instead.")

    def apply(
        self,
        meter: Meter,
        start: dt.datetime,
        end: dt.datetime,
        output_resolution: Resolution,
        timezone: dt.tzinfo = UTC,
        binning_anchor: dt.datetime | None = None,
    ) -> pd.DataFrame:
        """Apply formula values to a dataframe of quantities and return a single value column."""
        data = meter.measurements
        if self.capacity_based:
            if meter.capacity is None:
                raise ValueError("Capacity is required for capacity-based formulas.")
            data = meter.capacity
        return self._apply(data, start, end, output_resolution, timezone=timezone, binning_anchor=binning_anchor)

    def _apply(
        self,
        data: TimeseriesFrame,
        start: dt.datetime,
        end: dt.datetime,
        output_resolution: Resolution,
        timezone: dt.tzinfo = UTC,
        binning_anchor: dt.datetime | None = None,
    ) -> pd.DataFrame:
        """Apply formula values to a dataframe of quantities and return a single value column."""
        formula_values = self.get_values(start, end, data.resolution, timezone)

        result = data.reset_index(drop=True)
        formula_series = formula_values.set_index("timestamp")["value"].reindex(result["timestamp"])
        result["value"] = formula_series.values * result["value"].values
        return redistribute_to_resolution(
            result, data.resolution, output_resolution, start, end, binning_anchor=binning_anchor
        )
