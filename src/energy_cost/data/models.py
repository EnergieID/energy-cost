from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from energy_cost.capacity import CapacityRule
from energy_cost.registry import RegistryMixin
from energy_cost.tariff import Tariff
from energy_cost.tax import Tax
from energy_cost.types import TzInfo


class CustomerType(StrEnum):
    RESIDENTIAL = "residential"
    NON_RESIDENTIAL = "non_residential"
    PROTECTED = "protected"


class ConnectionType(StrEnum):
    ELECTRICITY = "electricity"
    GAS = "gas"


class Supplier(BaseModel, RegistryMixin[str, "Supplier"]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    products: dict[str, Tariff]


class RegionalData(BaseModel, RegistryMixin[tuple[str, ConnectionType], "RegionalData"]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    fees: dict[CustomerType, Tariff | list[Tariff]]
    distributors: dict[str, Tariff | list[Tariff]]
    taxes: Tax | list[Tax]
    timezone: TzInfo
    capacity_rule: CapacityRule | None = None
