from __future__ import annotations

import datetime as dt

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from energy_cost.resolution import Resolution, detect_resolution_and_range, to_pandas_freq, to_pandas_offset

from .formula import Formula


class TierBand(BaseModel):
    up_to: float | None = None
    formula: Formula

    def matches(self, values: pd.Series) -> pd.Series:
        if self.up_to is None:
            return pd.Series([True] * len(values), index=values.index)
        return values <= self.up_to


class TieredFormula(Formula):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    kind: str = "tiered"
    bands: list[TierBand] = Field(default_factory=list)
    band_period: Resolution | None = None

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

        if self.band_period is not None:
            return self._apply_with_period(data, start, end, resolution)

        result = pd.DataFrame(
            {"timestamp": pd.date_range(start=start, end=end, freq=to_pandas_freq(resolution), inclusive="left")}
        )
        result["value"] = pd.Series(float("nan"), index=result.index, dtype=float)

        # create series of data values with timestamps as index for easier calculations
        data_series = data.set_index("timestamp")["value"]

        for band in self.bands:
            mask = band.matches(data_series)
            banded_series = data_series[mask]

            if banded_series.empty:
                data_series = data_series[~mask]
                continue

            banded_frame = pd.DataFrame({"timestamp": banded_series.index, "value": banded_series.values})
            applied_band_values = band.formula.apply(banded_frame, resolution=resolution)
            result = result.merge(applied_band_values, on="timestamp", how="left", suffixes=("", "_band"))
            result["value"] = result["value_band"].combine_first(result["value"])
            result = result.drop(columns=["value_band"])

            # remove the banded values from the data series so they don't get processed by subsequent bands
            data_series = data_series[~mask]

        return result

    def _apply_with_period(
        self,
        data: pd.DataFrame,
        start: dt.datetime,
        end: dt.datetime,
        resolution: Resolution,
    ) -> pd.DataFrame:
        assert self.band_period is not None  # guaranteed by the call site

        result = pd.DataFrame(
            {"timestamp": pd.date_range(start=start, end=end, freq=to_pandas_freq(resolution), inclusive="left")}
        )
        result["value"] = pd.Series(float("nan"), index=result.index, dtype=float)

        indexed = data.set_index("timestamp")
        groups = indexed.groupby(pd.Grouper(freq=to_pandas_freq(self.band_period)))

        for period_start_key, group in groups:
            if group.empty:
                continue

            period_start = pd.Timestamp(period_start_key)  # type: ignore[arg-type]
            estimated_total = self._estimate_total_for_period(group, period_start, resolution)

            # Select the first band whose threshold covers the estimated total.
            matched_band = next(
                (band for band in self.bands if band.up_to is None or estimated_total <= band.up_to),
                None,
            )
            if matched_band is None:
                continue

            group_frame = pd.DataFrame({"timestamp": group.index, "value": group["value"].values})
            applied = matched_band.formula.apply(group_frame, resolution=resolution)

            result = result.merge(applied, on="timestamp", how="left", suffixes=("", "_band"))
            result["value"] = result["value_band"].combine_first(result["value"])
            result = result.drop(columns=["value_band"])

        return result

    def _estimate_total_for_period(
        self, group: pd.DataFrame, period_start: pd.Timestamp, resolution: Resolution
    ) -> float:
        assert self.band_period is not None  # guaranteed by the call site

        period_end = period_start + to_pandas_offset(self.band_period)
        actual_start = group.index[0]
        actual_end = group.index[-1] + to_pandas_offset(resolution)
        full_seconds = (period_end - period_start).total_seconds()
        actual_seconds = (actual_end - actual_start).total_seconds()

        group_sum = group["value"].sum()

        return group_sum * full_seconds / actual_seconds
