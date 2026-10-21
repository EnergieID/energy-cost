from __future__ import annotations

import datetime as dt
from typing import Literal

import pandas as pd
from pydantic import ConfigDict

from energy_cost.meter import TimeseriesFrame
from energy_cost.resolution import (
    Resolution,
    redistribute_to_resolution,
    snap_billing_period,
    to_pandas_freq,
)

from .base import FormulaBase


class PeriodicFormula(FormulaBase):
    """A fixed amount charged every period, independent of usage."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: Literal["periodic"] = "periodic"
    """Selects this formula type."""
    period: Resolution
    """How often the amount is charged."""
    constant_cost: float
    """Amount charged per period, e.g. €/month."""

    def _apply(
        self,
        data: TimeseriesFrame,
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


class UnitPeriodicFormula(FormulaBase):
    """A price per unit per period, e.g. €/MW/year."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: Literal["unit_periodic"] = "unit_periodic"
    """Selects this formula type."""
    period: Resolution
    """Period the price is expressed in."""
    constant_cost: float
    """Price per unit per period."""

    def get_values(
        self,
        start: dt.datetime,
        end: dt.datetime,
        output_resolution: Resolution,
        timezone: dt.tzinfo = dt.UTC,
    ) -> pd.DataFrame:
        period_freq = to_pandas_freq(self.period)

        snapped_start, _ = snap_billing_period(start, end, period_freq)
        period_timestamps = pd.date_range(start=snapped_start, end=end, freq=period_freq, inclusive="left")

        coarse_df = pd.DataFrame({"timestamp": period_timestamps, "value": float(self.constant_cost)})
        return redistribute_to_resolution(coarse_df, self.period, output_resolution, start, end)
