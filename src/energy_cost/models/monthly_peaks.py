"""Monthly capacity peaks for electricity (kW)."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

import narwhals as nw
from pydantic import BaseModel, ConfigDict, field_validator


class MonthlyPeaks(BaseModel):
    """Monthly capacity peaks for electricity (kW).

    Accepted input is either:
    - mapping-like ``{month -> peak_kw}``
    - dataframe with ``month`` and ``peak_kw``.

    Internal normalized schema is a Narwhals dataframe with:
    - ``month`` as YYYY-MM
    - ``peak_kw``
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    unit: str = "kW"
    data: nw.DataFrame

    @field_validator("data", mode="before")
    @classmethod
    def validate_peaks_data(cls, v: Any) -> nw.DataFrame:
        if v is None:
            raise ValueError("Monthly peaks cannot be None")

        rows: list[tuple[str, float]] = []

        if isinstance(v, Mapping) or hasattr(v, "items"):
            for key, value in v.items():
                ts = _to_datetime(key)
                rows.append((f"{ts.year:04d}-{ts.month:02d}", float(value)))
            return nw.from_dict(
                {"month": [m for m, _ in rows], "peak_kw": [p for _, p in rows]},
                backend="pandas",
            )

        df = nw.from_native(v, eager_only=True)
        if "month" in df.columns and "peak_kw" in df.columns:
            return df

        raise ValueError("MonthlyPeaks requires a mapping-like object or a dataframe with 'month' and 'peak_kw'")


def _to_datetime(value: object) -> datetime:
    """Convert various timestamp representations to a datetime."""
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()  # type: ignore[no-any-return]
    year = getattr(value, "year", None)
    month = getattr(value, "month", None)
    if year is not None and month is not None:
        return datetime(int(year), int(month), 1)
    return datetime.fromisoformat(str(value))
