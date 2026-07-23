from __future__ import annotations

import datetime as dt
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


def apply_extreme_formula(
    formulas: list[Formula],
    extreme_func: Callable[[pd.Series], float],
    period: Resolution,
    meter: Meter,
    start: dt.datetime,
    end: dt.datetime,
    output_resolution: Resolution,
    timezone: dt.tzinfo = dt.UTC,
    binning_anchor: dt.datetime | None = None,
) -> pd.DataFrame:
    sub_resolution = find_common_divisor(output_resolution, period)

    applied_dfs = [
        formula.apply(
            meter,
            start=start,
            end=end,
            output_resolution=sub_resolution,
            timezone=timezone,
            binning_anchor=binning_anchor,
        )
        for formula in formulas
    ]

    freq = to_pandas_freq(period)
    snapped_start, snapped_end = snap_billing_period(start, end, freq, anchor=binning_anchor)
    period_bins = pd.date_range(start=snapped_start, end=snapped_end, freq=freq)

    summed_by_period = [add_period_sum(df, period_bins) for df in applied_dfs]

    # add formula ID to handle ties later (arbitrary choice to take the first one in case of ties, but we want it to be deterministic)
    for i, df in enumerate(summed_by_period):
        df["formula_id"] = i

    # Now we have a list of DataFrames with 'timestamp', 'value', 'period', and 'period_sum' columns.
    # We want to find the extreme 'period_sum' across the formulas for each period, and keep the corresponding timestamps and values from the winning formula.
    # First, concatenate all the DataFrames and find the extreme period_sum for each period.
    all_data = pd.concat(summed_by_period, ignore_index=True)
    extreme_period_sums = (
        all_data.groupby("period")["period_sum"].agg(extreme_func).reset_index(name="extreme_period_sum")
    )
    # Merge back to find the rows where period_sum equals extreme_period_sum
    winners = all_data.merge(extreme_period_sums, on="period")
    winners = winners[winners["period_sum"] == winners["extreme_period_sum"]]
    # In case of ties, we take the first formula (lowest formula_id) for each period
    winners = winners.sort_values(["timestamp", "formula_id"]).drop_duplicates(subset="timestamp", keep="first")
    result = winners[["timestamp", "value"]].reset_index(drop=True)

    return redistribute_to_resolution(
        result, sub_resolution, output_resolution, start=start, end=end, binning_anchor=binning_anchor
    )


def get_extreme_values(
    formulas: list[Formula],
    extreme_func: Callable[[pd.Series], float],
    period: Resolution,
    start: dt.datetime,
    end: dt.datetime,
    output_resolution: Resolution,
    timezone: dt.tzinfo = dt.UTC,
) -> pd.DataFrame:
    if period != output_resolution:
        raise NotImplementedError(
            "Minimum/maximum get_values() only supports output_resolution equal to formula period. Use apply() instead."
        )

    value_frames = [
        formula.get_values(
            start=start,
            end=end,
            output_resolution=output_resolution,
            timezone=timezone,
        )
        for formula in formulas
    ]

    merged = value_frames[0][["timestamp", "value"]].rename(columns={"value": "value_0"})
    for i, frame in enumerate(value_frames[1:], start=1):
        merged = merged.merge(
            frame[["timestamp", "value"]].rename(columns={"value": f"value_{i}"}),
            on="timestamp",
            how="outer",
        )

    value_columns = [c for c in merged.columns if c.startswith("value_")]
    merged["value"] = merged[value_columns].agg(extreme_func, axis=1)

    result = merged[["timestamp", "value"]].sort_values("timestamp")
    return result.reset_index(drop=True)


class MinimumFormula(FormulaBase):
    """A formula returns the minimum formula result of multiple formulas in a given period."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    kind: Literal["minimum"] = "minimum"
    period: Resolution
    minimum: list[Formula] = Field(default_factory=list)

    def get_values(
        self,
        start: dt.datetime,
        end: dt.datetime,
        output_resolution: Resolution,
        timezone: dt.tzinfo = dt.UTC,
    ) -> pd.DataFrame:
        return get_extreme_values(
            formulas=self.minimum,
            extreme_func=min,
            period=self.period,
            start=start,
            end=end,
            output_resolution=output_resolution,
            timezone=timezone,
        )

    def apply(
        self,
        meter: Meter,
        start: dt.datetime,
        end: dt.datetime,
        output_resolution: Resolution,
        timezone: dt.tzinfo = dt.UTC,
        binning_anchor: dt.datetime | None = None,
    ) -> pd.DataFrame:
        return apply_extreme_formula(
            formulas=self.minimum,
            extreme_func=min,
            period=self.period,
            meter=meter,
            start=start,
            end=end,
            output_resolution=output_resolution,
            timezone=timezone,
            binning_anchor=binning_anchor,
        )


class MaximumFormula(FormulaBase):
    """A formula returns the maximum formula result of multiple formulas in a given period."""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    kind: Literal["maximum"] = "maximum"
    period: Resolution
    maximum: list[Formula] = Field(default_factory=list)

    def get_values(
        self,
        start: dt.datetime,
        end: dt.datetime,
        output_resolution: Resolution,
        timezone: dt.tzinfo = dt.UTC,
    ) -> pd.DataFrame:
        return get_extreme_values(
            formulas=self.maximum,
            extreme_func=max,
            period=self.period,
            start=start,
            end=end,
            output_resolution=output_resolution,
            timezone=timezone,
        )

    def apply(
        self,
        meter: Meter,
        start: dt.datetime,
        end: dt.datetime,
        output_resolution: Resolution,
        timezone: dt.tzinfo = dt.UTC,
        binning_anchor: dt.datetime | None = None,
    ) -> pd.DataFrame:
        return apply_extreme_formula(
            formulas=self.maximum,
            extreme_func=max,
            period=self.period,
            meter=meter,
            start=start,
            end=end,
            output_resolution=output_resolution,
            timezone=timezone,
            binning_anchor=binning_anchor,
        )
