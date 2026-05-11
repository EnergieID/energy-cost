from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from datetime import UTC

import pandas as pd
from pydantic import BaseModel

from energy_cost.resolution import Resolution, align_timestamps_to_tz, detect_resolution_and_range


class FormulaBase(ABC, BaseModel):
    @abstractmethod
    def get_values(
        self,
        start: dt.datetime,
        end: dt.datetime,
        resolution: Resolution,
        timezone: dt.tzinfo = UTC,
    ) -> pd.DataFrame:
        """Return time-indexed values for the formula."""

    def apply(
        self,
        data: pd.DataFrame,
        resolution: Resolution | None = None,
        timezone: dt.tzinfo = UTC,
        *,
        start: dt.datetime | None = None,
        end: dt.datetime | None = None,
    ) -> pd.DataFrame:
        """Apply formula values to a dataframe of quantities and return a single value column."""
        if data.empty:
            return data.copy()
        data = align_timestamps_to_tz(data, timezone)
        if start is None or end is None or resolution is None:
            start, end, resolution = detect_resolution_and_range(data, resolution)
        formula_values = self.get_values(start, end, resolution, timezone)

        result = data.reset_index(drop=True)
        formula_series = formula_values.set_index("timestamp")["value"].reindex(result["timestamp"])
        result["value"] = formula_series.values * result["value"].values
        return result
