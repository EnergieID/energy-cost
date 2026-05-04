import datetime as dt
from enum import StrEnum

from pydantic import BaseModel, ConfigDict

from energy_cost.registry import RegistryMixin
from energy_cost.tariff import Tariff
from energy_cost.tax import Tax


class CustomerType(StrEnum):
    RESIDENTIAL = "residential"
    NON_RESIDENTIAL = "non_residential"
    PROTECTED = "protected"


class ConnectionType(StrEnum):
    ELECTRICITY = "electricity"
    GAS = "gas"


class Supplier(RegistryMixin[str, "Supplier"], BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    products: dict[str, Tariff]


class RegionalData(RegistryMixin[tuple[str, ConnectionType], "RegionalData"], BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    fees: dict[CustomerType, Tariff | list[Tariff]]
    distributors: dict[str, Tariff | list[Tariff]]
    taxes: Tax | list[Tax]
    timezone: dt.tzinfo
