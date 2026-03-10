"""Core data models for energy cost calculation."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

import narwhals as nw
from pydantic import BaseModel, ConfigDict, field_validator

from energy_cost.enums import (
    Carrier,
    CustomerType,
    Direction,
    EnergyUnit,
    GasTariffClass,
    Market,
    MeterRegister,
    MeterType,
    PriceUnit,
    Region,
    VoltageLevel,
)


class EnergySeries(BaseModel):
    """Energy consumption or injection time series.

    Accepted input is a native eager dataframe with either:
    - a DatetimeIndex and one or more energy columns, or
    - explicit ``timestamp`` column and one or more energy columns.

    Internal normalized schema is a Narwhals dataframe that always includes
    ``timestamp`` and preserves provided energy columns.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    carrier: Carrier
    direction: Direction
    timezone: str = "Europe/Brussels"
    unit: EnergyUnit = EnergyUnit.KWH
    data: nw.DataFrame

    @field_validator("data", mode="before")
    @classmethod
    def validate_data(cls, v: Any) -> nw.DataFrame:
        if v is None:
            raise ValueError("Data cannot be None")

        df = nw.from_native(v, eager_only=True)
        if "timestamp" in df.columns:
            return df

        if not hasattr(v, "index"):
            raise ValueError("EnergySeries requires a DatetimeIndex or a 'timestamp' column")

        return nw.from_dict(
            {
                "timestamp": list(v.index),
                **{column: df[column].to_list() for column in df.columns},
            },
            backend=df.implementation,
        )


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


class ElectricityConnection(BaseModel):
    """Electricity-specific connection details."""

    meter_register: MeterRegister = MeterRegister.SINGLE
    meter_type: MeterType = MeterType.DIGITAL
    voltage_level: VoltageLevel = VoltageLevel.LV
    has_separate_injection_point: bool = False


class GasConnection(BaseModel):
    """Gas-specific connection details."""

    gas_tariff_class: GasTariffClass | None = None
    telemetered: bool | None = None


class ConnectionInfo(BaseModel):
    """Connection and customer metadata."""

    region: Region
    dso: str  # e.g. "Fluvius Antwerpen"
    customer_type: CustomerType
    vat_rate: float | None = None  # inferred from customer_type + region rules if None

    electricity: ElectricityConnection = ElectricityConnection()
    gas: GasConnection = GasConnection()

    def get_vat_rate(self) -> float:
        """Return explicit VAT rate or infer from region/customer_type."""
        if self.vat_rate is not None:
            return self.vat_rate
        # Belgium standard VAT rate for energy
        return 0.21


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
            return nw.from_dict({"month": [m for m, _ in rows], "peak_kw": [p for _, p in rows]}, backend="pandas")

        df = nw.from_native(v, eager_only=True)
        if "month" in df.columns and "peak_kw" in df.columns:
            return df

        raise ValueError("MonthlyPeaks requires a mapping-like object or a dataframe with 'month' and 'peak_kw'")


def _to_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()  # type: ignore[no-any-return]
    year = getattr(value, "year", None)
    month = getattr(value, "month", None)
    if year is not None and month is not None:
        return datetime(int(year), int(month), 1)
    return datetime.fromisoformat(str(value))


class CostResult(BaseModel):
    """Output of a cost calculation: itemized breakdown + totals."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    breakdown: Any
    totals: dict[str, float]
    assumptions: dict
