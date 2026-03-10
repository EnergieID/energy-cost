"""Energy-cost models — one-stop import for all model classes and enums.

Usage::

    from energy_cost.models import EnergySeries, Carrier, TariffDefinition, ...
"""

from energy_cost.models._enums import Carrier, CustomerType, Direction
from energy_cost.models.connection import (
    ConnectionInfo,
    ElectricityConnection,
    GasConnection,
    GasTariffClass,
    MeterRegister,
    MeterType,
    Region,
    VoltageLevel,
)
from energy_cost.models.cost_result import CostResult
from energy_cost.models.energy_series import EnergySeries, EnergyUnit
from energy_cost.models.grid_tariff import GridComponentType, GridTariffComponent, GridTariffSet, GridUnit
from energy_cost.models.market_price_series import Market, MarketPriceSeries, PriceUnit
from energy_cost.models.monthly_peaks import MonthlyPeaks
from energy_cost.models.tariff import (
    Channel,
    ContractType,
    PricingKind,
    PricingRule,
    TariffComponent,
    TariffComponentType,
    TariffDefinition,
    TimeOfUse,
)
from energy_cost.models.tax_rule import TaxKind, TaxRule, TaxTier, TaxUnit

__all__ = [
    # Shared enums
    "Carrier",
    "CustomerType",
    "Direction",
    # Connection
    "ConnectionInfo",
    "ElectricityConnection",
    "GasConnection",
    "GasTariffClass",
    "MeterRegister",
    "MeterType",
    "Region",
    "VoltageLevel",
    # Energy series
    "EnergySeries",
    "EnergyUnit",
    # Market prices
    "Market",
    "MarketPriceSeries",
    "PriceUnit",
    # Monthly peaks
    "MonthlyPeaks",
    # Cost result
    "CostResult",
    # Tariff
    "Channel",
    "ContractType",
    "PricingKind",
    "PricingRule",
    "TariffComponent",
    "TariffComponentType",
    "TariffDefinition",
    "TimeOfUse",
    # Grid tariff
    "GridComponentType",
    "GridTariffComponent",
    "GridTariffSet",
    "GridUnit",
    # Tax
    "TaxKind",
    "TaxRule",
    "TaxTier",
    "TaxUnit",
]
