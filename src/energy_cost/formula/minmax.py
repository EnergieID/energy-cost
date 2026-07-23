from __future__ import annotations

import datetime as dt
from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal

import pandas as pd
from pydantic import ConfigDict, Field

from energy_cost.meter import Meter
from energy_cost.resolution import (
    Resolution,
    find_common_divisor,
    redistribute_to_resolution,
    snap_billing_period,
    to_pandas_freq,
)

from .base import FormulaBase

if TYPE_CHECKING:
    from .formula import Formula


def add_period_sum(df: pd.DataFrame, period_bins: pd.DatetimeIndex) -> pd.DataFrame:
    """Helper function to add a column with the sum of values within each period."""
    df = df.copy()
    df["period"] = pd.cut(df["timestamp"], bins=period_bins, right=False)
    period_sums = df.groupby("period")["value"].sum().reset_index(name="period_sum")
    return df.merge(period_sums, on="period", how="left")


class ExtremeFormulaBase(FormulaBase):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    period: Resolution

    @property
    @abstractmethod
    def formulas(self) -> list[Formula]:
        """Return child formulas to evaluate."""

    @property
    @abstractmethod
    def extreme_func(self) -> Callable[[pd.Series], float]:
        """Return the row-wise or period-wise selector function (min/max)."""

    def get_values(
        self,
        start: dt.datetime,
        end: dt.datetime,
        output_resolution: Resolution,
        timezone: dt.tzinfo = dt.UTC,
    ) -> pd.DataFrame:
        if self.period != output_resolution:
            raise NotImplementedError(
                "Minimum/maximum get_values() only supports output_resolution equal to formula period. "
                "Use apply() instead."
            )

        value_frames = [
            formula.get_values(
                start=start,
                end=end,
                output_resolution=output_resolution,
                timezone=timezone,
            )
            for formula in self.formulas
        ]

        merged = value_frames[0][["timestamp", "value"]].rename(columns={"value": "value_0"})
        for i, frame in enumerate(value_frames[1:], start=1):
            merged = merged.merge(
                frame[["timestamp", "value"]].rename(columns={"value": f"value_{i}"}),
                on="timestamp",
                how="outer",
            )

        value_columns = [c for c in merged.columns if c.startswith("value_")]
        merged["value"] = merged[value_columns].agg(self.extreme_func, axis=1)

        result = merged[["timestamp", "value"]].sort_values("timestamp")
        return result.reset_index(drop=True)

    def apply(
        self,
        meter: Meter,
        start: dt.datetime,
        end: dt.datetime,
        output_resolution: Resolution,
        timezone: dt.tzinfo = dt.UTC,
        binning_anchor: dt.datetime | None = None,
    ) -> pd.DataFrame:
        sub_resolution = find_common_divisor(output_resolution, self.period)

        applied_dfs = [
            formula.apply(
                meter,
                start=start,
                end=end,
                output_resolution=sub_resolution,
                timezone=timezone,
                binning_anchor=binning_anchor,
            )
            for formula in self.formulas
        ]

        freq = to_pandas_freq(self.period)
        snapped_start, snapped_end = snap_billing_period(start, end, freq, anchor=binning_anchor)
        period_bins = pd.date_range(start=snapped_start, end=snapped_end, freq=freq)

        summed_by_period = [add_period_sum(df, period_bins) for df in applied_dfs]

        # Add formula ID to resolve ties deterministically by declaration order.
        for i, df in enumerate(summed_by_period):
            df["formula_id"] = i

        all_data = pd.concat(summed_by_period, ignore_index=True)
        extreme_period_sums = (
            all_data.groupby("period")["period_sum"].agg(self.extreme_func).reset_index(name="extreme_period_sum")
        )
        winners = all_data.merge(extreme_period_sums, on="period")
        winners = winners[winners["period_sum"] == winners["extreme_period_sum"]]
        winners = winners.sort_values(["timestamp", "formula_id"]).drop_duplicates(subset="timestamp", keep="first")
        result = winners[["timestamp", "value"]].reset_index(drop=True)

        return redistribute_to_resolution(
            result, sub_resolution, output_resolution, start=start, end=end, binning_anchor=binning_anchor
        )


class MinimumFormula(ExtremeFormulaBase):
    """A formula returns the minimum formula result of multiple formulas in a given period."""

    kind: Literal["minimum"] = "minimum"
    minimum: list[Formula] = Field(default_factory=list)

    @property
    def formulas(self) -> list[Formula]:
        return self.minimum

    @property
    def extreme_func(self) -> Callable[[pd.Series], float]:
        return min


class MaximumFormula(ExtremeFormulaBase):
    """A formula returns the maximum formula result of multiple formulas in a given period."""

    kind: Literal["maximum"] = "maximum"
    maximum: list[Formula] = Field(default_factory=list)

    @property
    def formulas(self) -> list[Formula]:
        return self.maximum

    @property
    def extreme_func(self) -> Callable[[pd.Series], float]:
        return max
