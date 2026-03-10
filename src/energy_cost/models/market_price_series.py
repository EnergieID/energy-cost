"""Market price series (e.g., EPEX Spot day-ahead 15-min prices)."""

from __future__ import annotations

from enum import StrEnum

import narwhals as nw
from narwhals.typing import IntoDataFrame
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

    Accepted input is ``IntoDataFrame`` with ``timestamp`` and ``price``.

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
    def validate_data(cls, v: IntoDataFrame) -> nw.DataFrame:
        if v is None:
            raise ValueError("Data cannot be None")

        df = nw.from_native(v, eager_only=True)
        if "timestamp" not in df.columns or "price" not in df.columns:
            raise ValueError("MarketPriceSeries requires a dataframe with 'timestamp' and 'price' columns")

        return df

    def factor_to_eur_per_kwh(self) -> float:
        """Convert prices to EUR/kWh."""
        return 1 / 1000.0 if self.unit == PriceUnit.EUR_PER_MWH else 1.0

    def factor_to_eur_per_mwh(self) -> float:
        """Convert prices to EUR/MWh."""
        return 1000.0 if self.unit == PriceUnit.EUR_PER_KWH else 1.0
