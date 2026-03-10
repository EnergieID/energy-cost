"""Shared test fixtures for energy-cost tests."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from energy_cost.core.models import ConnectionInfo, ElectricityConnection, EnergySeries, MarketPriceSeries, MonthlyPeaks
from energy_cost.enums import (
    Carrier,
    Channel,
    ContractType,
    CustomerType,
    Direction,
    Market,
    MeterRegister,
    MeterType,
    PricingKind,
    Region,
    TariffComponentType,
    VoltageLevel,
)
from energy_cost.regions.be_flanders.grid import resolve_grid_tariffs
from energy_cost.regions.be_flanders.taxes import resolve_tax_rules
from energy_cost.tariffs.models import PricingRule, TariffComponent, TariffDefinition


@pytest.fixture()
def one_day_index() -> pd.DatetimeIndex:
    """96 quarter-hourly timestamps for 2026-01-15 (Brussels timezone)."""
    return pd.date_range("2026-01-15", periods=96, freq="15min", tz="Europe/Brussels")


@pytest.fixture()
def one_month_index() -> pd.DatetimeIndex:
    """Full January 2026 at 15-min resolution (31 days × 96 = 2976 timestamps)."""
    return pd.date_range("2026-01-01", periods=31 * 96, freq="15min", tz="Europe/Brussels")


@pytest.fixture()
def constant_energy(one_day_index: pd.DatetimeIndex) -> EnergySeries:
    """Constant 0.25 kWh per quarter-hour (≈1 kW average, 24 kWh/day)."""
    return EnergySeries(
        carrier=Carrier.ELECTRICITY,
        direction=Direction.OFFTAKE,
        data=pd.DataFrame({"kwh": 0.25}, index=one_day_index),
    )


@pytest.fixture()
def constant_prices(one_day_index: pd.DatetimeIndex) -> MarketPriceSeries:
    """Constant 100 EUR/MWh day-ahead price."""
    return MarketPriceSeries(
        market=Market.EPEX_DA_BE_15MIN,
        data=pd.Series(100.0, index=one_day_index),
    )


@pytest.fixture()
def dynamic_offtake_tariff() -> TariffDefinition:
    """EBEM-style dynamic electricity offtake tariff.

    Energy: c€/kWh = 0.108 × Belpex15′ + 1.625
    Fixed fee: 70.75 EUR/year
    """
    return TariffDefinition(
        name="Test Dynamic Offtake",
        carrier=Carrier.ELECTRICITY,
        direction=Direction.OFFTAKE,
        contract_type=ContractType.DYNAMIC,
        components=[
            TariffComponent(
                component_type=TariffComponentType.ENERGY,
                channel=Channel.KWH,
                pricing=PricingRule(
                    kind=PricingKind.INDEXED_LINEAR,
                    index_name="Belpex15'",
                    coef=0.108,
                    add_cents_per_kwh=1.625,
                ),
                vat_applicable=True,
            ),
            TariffComponent(
                component_type=TariffComponentType.FIXED_FEE,
                pricing=PricingRule(
                    kind=PricingKind.FIXED_PER_YEAR,
                    eur_per_year=70.75,
                ),
                vat_applicable=True,
            ),
        ],
        valid_from=date(2026, 1, 1),
        valid_to=date(2026, 12, 31),
    )


@pytest.fixture()
def residential_connection() -> ConnectionInfo:
    """Residential LV digital connection at Fluvius Antwerpen."""
    return ConnectionInfo(
        region=Region.BE_VLG,
        dso="Fluvius Antwerpen",
        customer_type=CustomerType.RESIDENTIAL,
        electricity=ElectricityConnection(
            meter_register=MeterRegister.SINGLE,
            meter_type=MeterType.DIGITAL,
            voltage_level=VoltageLevel.LV,
        ),
    )


@pytest.fixture()
def monthly_peaks_january() -> MonthlyPeaks:
    """Monthly peak of 2.5 kW for January 2026."""
    return MonthlyPeaks(
        data=pd.Series(
            [2.5],
            index=pd.PeriodIndex(["2026-01"], freq="M"),
        ),
    )
