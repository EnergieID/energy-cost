from __future__ import annotations

import datetime as dt

import pandas as pd
from pydantic import BaseModel, Field

from energy_cost.resolution import Resolution, detect_resolution_and_range, to_pandas_freq

from .formula import Formula


class TierBand(BaseModel):
    up_to: float | None = None
    formula: Formula

    def matches(self, values: pd.Series) -> pd.Series:
        if self.up_to is None:
            return pd.Series([True] * len(values), index=values.index)
        return values <= self.up_to


class TieredFormula(Formula):
    kind: str = "tiered"
    bands: list[TierBand] = Field(default_factory=list)

    def get_values(
        self,
        start: dt.datetime,
        end: dt.datetime,
        resolution: Resolution,
    ) -> pd.DataFrame:
        result: pd.DataFrame | None = None
        for index, band in enumerate(self.bands, start=1):
            band_values = band.formula.get_values(start, end, resolution).copy()
            rename_map = {
                column: (f"tier_{index}" if column == "value" else f"tier_{index}_{column}")
                for column in band_values.columns
                if column != "timestamp"
            }
            band_values = band_values.rename(columns=rename_map)
            result = band_values if result is None else result.merge(band_values, on="timestamp", how="outer")

        if result is None:
            return pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns]")})
        return result

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

        # create series of data values with timestamps as index for easier calculations
        data_series = data.set_index("timestamp")["value"]

        for band in self.bands:
            mask = band.matches(data_series)
            banded_series = data_series[mask]

            banded_frame = pd.DataFrame({"timestamp": banded_series.index, "value": banded_series.values})
            applied_band_values = band.formula.apply(banded_frame, resolution=resolution)
            result = result.merge(applied_band_values, on="timestamp", how="left", suffixes=("", "_band"))
            result["value"] = result["value_band"].combine_first(result["value"])
            result = result.drop(columns=["value_band"])

            # remove the banded values from the data series so they don't get processed by subsequent bands
            data_series = data_series[~mask]

        return result
