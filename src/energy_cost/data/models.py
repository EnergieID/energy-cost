from enum import StrEnum

from pydantic import BaseModel

from energy_cost.tariff import Tariff
from energy_cost.tax import Tax


class CustomerType(StrEnum):
    RESIDENTIAL = "residential"
    NON_RESIDENTIAL = "non_residential"
    PROTECTED = "protected"


class ConnectionType(StrEnum):
    ELECTRICITY = "electricity"
    GAS = "gas"


class RegionalData(BaseModel):
    fees: dict[CustomerType, Tariff | list[Tariff]]
    distributors: dict[str, Tariff | list[Tariff]]
    taxes: Tax | list[Tax]
