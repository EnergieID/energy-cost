from __future__ import annotations

import datetime as dt
from datetime import UTC
from enum import StrEnum

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from energy_cost.resolution import (
    Resolution,
    align_timestamps_to_tz,
    detect_resolution_and_range,
    snap_billing_period,
    to_pandas_freq,
)

from .formula import Formula


class TieringMode(StrEnum):
    BANDED = "banded"
    PROGRESSIVE = "progressive"


class TierBand(BaseModel):
    up_to: float | None = None
    formula: Formula


class TieredFormula(Formula):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: str = "tiered"
    bands: list[TierBand] = Field(default_factory=list)
    band_period: Resolution | None = None
    mode: TieringMode = TieringMode.PROGRESSIVE

    def get_values(
        self,
        start: dt.datetime,
        end: dt.datetime,
        resolution: Resolution,
        timezone: dt.tzinfo = UTC,
    ) -> pd.DataFrame:
        raise NotImplementedError("Tiered formulas cannot be represented as time series. Use apply() instead.")

    def apply(
        self,
        data: pd.DataFrame,
        resolution: Resolution | None = None,
        timezone: dt.tzinfo = UTC,
        *,
        start: dt.datetime | None = None,
        end: dt.datetime | None = None,
    ) -> pd.DataFrame:
        data = align_timestamps_to_tz(data, timezone)
        if start is None or end is None or resolution is None:
            start, end, resolution = detect_resolution_and_range(data, resolution)

        # ── Step 1: estimate the full-period total for every data row ──
        indexed = data.set_index("timestamp")
        if self.band_period is not None:
            period_freq = to_pandas_freq(self.band_period)
            resolution_freq = to_pandas_freq(resolution)

            range_start, range_end = snap_billing_period(start, end, period_freq)

            # Build a lookup: period_start → number of resolution slots in a complete period.
            # (Varies per period, e.g. months have different day counts.)
            period_starts = pd.DataFrame(
                {"period_start": pd.date_range(range_start, range_end, freq=period_freq, inclusive="left")}
            )
            full_grid = pd.DataFrame(
                {"timestamp": pd.date_range(range_start, range_end, freq=resolution_freq, inclusive="left")}
            )
            full_grid_labelled = pd.merge_asof(
                full_grid, period_starts, left_on="timestamp", right_on="period_start", direction="backward"
            )
            full_slot_count = full_grid_labelled.groupby("period_start")["timestamp"].count()

            # Label every data row with its billing period, then sum and count per period.
            data_labelled = pd.merge_asof(
                indexed.reset_index().sort_values("timestamp"),
                period_starts,
                left_on="timestamp",
                right_on="period_start",
                direction="backward",
            )
            actual_sum = data_labelled.groupby("period_start")["value"].sum()
            actual_slot_count = data_labelled.groupby("period_start")["value"].count()

            # Scale actual sum to a full-period estimate:
            #   estimated_total = actual_sum * full_slot_count / actual_slot_count
            estimated_total_per_period: pd.Series = actual_sum * full_slot_count / actual_slot_count.clip(lower=1)  # type: ignore[operator]
            period_total: pd.Series = data_labelled.set_index("timestamp")["period_start"].map(
                estimated_total_per_period
            )
        else:
            # No band_period: each resolution slot is its own "period".
            period_total = indexed["value"].copy()

        # ── Step 2: apply band formulas, skipping bands with no contribution ──
        input_data = data[["timestamp", "value"]]
        result_values = pd.Series(0.0, index=indexed.index)

        if self.mode == TieringMode.BANDED:
            unmatched = pd.Series(True, index=indexed.index)
            for band in self.bands:
                mask = unmatched if band.up_to is None else unmatched & (period_total <= band.up_to)
                if mask.any():
                    band_values = band.formula.apply(
                        input_data.copy(), resolution=resolution, timezone=timezone, start=start, end=end
                    )
                    result_values = result_values.where(~mask, band_values.set_index("timestamp")["value"])
                unmatched = unmatched & ~mask

            if unmatched.any():
                raise ValueError("No tier band matches the estimated total for the period.")

        else:  # PROGRESSIVE
            prev_up_to = 0.0
            for band in self.bands:
                band_ceiling = period_total.clip(upper=band.up_to) if band.up_to is not None else period_total
                contribution = (band_ceiling - prev_up_to).clip(lower=0.0)
                fraction = contribution.div(period_total.where(period_total > 0, other=1.0)).where(
                    period_total > 0, other=0.0
                )
                if fraction.any():
                    band_values = band.formula.apply(
                        input_data.copy(), resolution=resolution, timezone=timezone, start=start, end=end
                    )
                    result_values = result_values + band_values.set_index("timestamp")["value"] * fraction
                if band.up_to is not None:
                    prev_up_to = band.up_to
                else:
                    break

        result_ts = pd.date_range(start=start, end=end, freq=to_pandas_freq(resolution), inclusive="left")
        result = pd.DataFrame({"timestamp": result_ts})
        result["value"] = result["timestamp"].map(result_values).astype(float)
        return result
