from __future__ import annotations

import datetime as dt
from enum import StrEnum

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from energy_cost.resolution import Resolution, detect_resolution_and_range, to_pandas_freq, to_pandas_offset

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
    ) -> pd.DataFrame:
        raise NotImplementedError("Tiered formulas cannot be represented as time series. Use apply() instead.")

    def apply(
        self,
        data: pd.DataFrame,
        resolution: Resolution | None = None,
    ) -> pd.DataFrame:
        start, end, resolution = detect_resolution_and_range(data, resolution)

        result = pd.DataFrame(
            {"timestamp": pd.date_range(start=start, end=end, freq=to_pandas_freq(resolution), inclusive="left")}
        )
        result["value"] = pd.Series(float("nan"), index=result.index, dtype=float)

        indexed = data.set_index("timestamp")
        groups = indexed.groupby(pd.Grouper(freq=to_pandas_freq(self.band_period or resolution)))

        for period_start_key, group in groups:
            period_start = pd.Timestamp(str(period_start_key))
            estimated_total = self._estimate_total_for_period(group, period_start, resolution)
            group_frame = pd.DataFrame({"timestamp": group.index, "value": group["value"].values})

            if self.mode == TieringMode.PROGRESSIVE:
                applied = self._apply_progressive_group(group_frame, estimated_total, resolution)
            else:
                applied = self._apply_banded_group(group_frame, estimated_total, resolution)

            result = result.merge(applied, on="timestamp", how="left", suffixes=("", "_band"))
            result["value"] = result["value_band"].combine_first(result["value"])
            result = result.drop(columns=["value_band"])

        return result

    def _apply_banded_group(
        self,
        group_frame: pd.DataFrame,
        estimated_total: float,
        resolution: Resolution,
    ) -> pd.DataFrame:
        """Select the first band whose threshold covers estimated_total and apply it to all rows."""
        matched_band = next(
            (band for band in self.bands if band.up_to is None or estimated_total <= band.up_to),
            None,
        )
        if matched_band is None:
            raise ValueError("No tier band matches the estimated total for the period.")
        return matched_band.formula.apply(group_frame.copy(), resolution=resolution)

    def _apply_progressive_group(
        self,
        group_frame: pd.DataFrame,
        estimated_total: float,
        resolution: Resolution,
    ) -> pd.DataFrame:
        """Apply all bands proportionally based on their share of estimated_total."""
        accumulated = pd.Series(0.0, index=range(len(group_frame)))
        prev_up_to = 0.0

        for band in self.bands:
            band_up_to = min(band.up_to, estimated_total) if band.up_to is not None else estimated_total
            contribution = max(0.0, band_up_to - prev_up_to)
            fraction = contribution / estimated_total if estimated_total > 0 else 0.0

            if fraction > 0:
                applied = band.formula.apply(group_frame.copy(), resolution=resolution)
                accumulated = accumulated + applied["value"].to_numpy() * fraction

            if band.up_to is not None:
                prev_up_to = band.up_to
            else:
                break

        result = group_frame.copy()
        result["value"] = accumulated.to_numpy()
        return result

    def _estimate_total_for_period(
        self, group: pd.DataFrame, period_start: pd.Timestamp, resolution: Resolution
    ) -> float:
        if self.band_period is None:
            return group["value"].sum()

        period_end = period_start + to_pandas_offset(self.band_period)
        actual_start = group.index[0]
        actual_end = group.index[-1] + to_pandas_offset(resolution)
        full_seconds = (period_end - period_start).total_seconds()
        actual_seconds = (actual_end - actual_start).total_seconds()

        group_sum = group["value"].sum()

        return group_sum * full_seconds / actual_seconds
