"""Market price series (e.g., EPEX Spot day-ahead 15-min prices)."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import narwhals as nw
from pydantic import BaseModel, ConfigDict, field_validator


class PriceUnit(StrEnum):
    EUR_PER_MWH = "EUR/MWh"
    EUR_PER_KWH = "EUR/kWh"


class Market(StrEnum):
    EPEX_DA_BE_15MIN = "EPEX_DA_BE_15MIN"
    CUSTOM_MONTHLY = "CUSTOM_MONTHLY"
    CUSTOM_OTHER = "CUSTOM_OTHER"


class MarketPriceSeries(BaseModel):
    """Market price series (e.g., EPEX Spot day-ahead 15-min prices).

    Accepted input is either:
    - a native eager series with DatetimeIndex, or
    - a native eager dataframe with ``price`` and optional ``timestamp``.

    Internal normalized schema is a Narwhals dataframe with:
    - ``timestamp``
    - ``price``
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    market: Market
    unit: PriceUnit = PriceUnit.EUR_PER_MWH
    data: nw.DataFrame

    @field_validator("data", mode="before")
    @classmethod
    def validate_data(cls, v: Any) -> nw.DataFrame:
        if v is None:
            raise ValueError("Data cannot be None")

        if hasattr(v, "index") and not hasattr(v, "columns"):
            series = nw.from_native(v, eager_only=True, allow_series=True)
            return nw.from_dict(
                {
                    "timestamp": list(v.index),
                    "price": series.to_list(),
                },
                backend=series.implementation,
            )

        df = nw.from_native(v, eager_only=True)
        if "price" not in df.columns:
            raise ValueError("MarketPriceSeries requires a 'price' column or a native series")

        if "timestamp" in df.columns:
            return df

        if not hasattr(v, "index"):
            raise ValueError("MarketPriceSeries requires a DatetimeIndex or a 'timestamp' column")

        return nw.from_dict(
            {
                "timestamp": list(v.index),
                "price": df["price"].to_list(),
            },
            backend=df.implementation,
        )

    def factor_to_eur_per_kwh(self) -> float:
        """Convert prices to EUR/kWh."""
        return 1 / 1000.0 if self.unit == PriceUnit.EUR_PER_MWH else 1.0

    def factor_to_eur_per_mwh(self) -> float:
        """Convert prices to EUR/MWh."""
        return 1000.0 if self.unit == PriceUnit.EUR_PER_KWH else 1.0
