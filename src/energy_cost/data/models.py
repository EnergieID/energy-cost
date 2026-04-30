import datetime as dt
from enum import StrEnum
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

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
    model_config = ConfigDict(arbitrary_types_allowed=True)

    fees: dict[CustomerType, Tariff | list[Tariff]]
    distributors: dict[str, Tariff | list[Tariff]]
    taxes: Tax | list[Tax]
    timezone: dt.tzinfo

    _registry: ClassVar[dict[str, dict[ConnectionType, "RegionalData"]]] = {}

    @classmethod
    def register(cls, region_key: str, connection_type: ConnectionType, data: "RegionalData") -> None:
        cls._registry.setdefault(region_key, {})[connection_type] = data

    @classmethod
    def get(cls, region_key: str, connection_type: ConnectionType) -> "RegionalData":
        return cls._registry[region_key][connection_type]
