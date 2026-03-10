"""Monthly capacity peaks for electricity (kW)."""

from __future__ import annotations

import narwhals as nw
from narwhals.typing import IntoDataFrame
from pydantic import BaseModel, ConfigDict, field_validator


class MonthlyPeaks(BaseModel):
    """Monthly capacity peaks for electricity (kW).

    Accepted input is ``IntoDataFrame`` with ``month`` and ``peak_kw``.

    Internal normalized schema is a Narwhals dataframe with:
    - ``month`` as YYYY-MM
    - ``peak_kw``
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    unit: str = "kW"
    data: nw.DataFrame

    @field_validator("data", mode="before")
    @classmethod
    def validate_peaks_data(cls, v: IntoDataFrame) -> nw.DataFrame:
        if v is None:
            raise ValueError("Monthly peaks cannot be None")

        df = nw.from_native(v, eager_only=True)
        if "month" in df.columns and "peak_kw" in df.columns:
            return df

        raise ValueError("MonthlyPeaks requires a dataframe with 'month' and 'peak_kw' columns")
