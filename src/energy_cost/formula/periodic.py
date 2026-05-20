from __future__ import annotations

import datetime as dt
from datetime import UTC
from typing import Literal

import pandas as pd
from pydantic import ConfigDict

from energy_cost.resolution import (
    Resolution,
    redistribute_to_resolution,
    snap_billing_period,
    to_pandas_freq,
)

from .base import FormulaBase


class PeriodicFormula(FormulaBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: Literal["periodic"] = "periodic"
    period: Resolution
    constant_cost: float

    def get_values(
        self,
        start: dt.datetime,
        end: dt.datetime,
        output_resolution: Resolution,
        timezone: dt.tzinfo = UTC,
    ) -> pd.DataFrame:
        raise NotImplementedError("Periodic formulas cannot be represented as time series. Use apply() instead.")

    def _apply(
        self,
        data: pd.DataFrame,
        start: dt.datetime,
        end: dt.datetime,
        output_resolution: Resolution,
        timezone: dt.tzinfo = dt.UTC,
        binning_anchor: dt.datetime | None = None,
    ) -> pd.DataFrame:
        period_freq = to_pandas_freq(self.period)

        # Snap to period boundary so all periods in [start, end) are covered.
        snapped_start, _ = snap_billing_period(start, end, period_freq, anchor=binning_anchor)
        period_timestamps = pd.date_range(start=snapped_start, end=end, freq=period_freq, inclusive="left")

        coarse_df = pd.DataFrame({"timestamp": period_timestamps, "value": float(self.constant_cost)})
        return redistribute_to_resolution(
            coarse_df, self.period, output_resolution, start, end, binning_anchor=binning_anchor
        )
