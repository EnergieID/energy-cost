from enum import StrEnum

from pydantic import ConfigDict

from energy_cost.capacity import CapacityRule
from energy_cost.catalog import Catalog
from energy_cost.model import EnergyCostModel
from energy_cost.registry import RegistryMixin
from energy_cost.tariff import Tariff
from energy_cost.tax import Tax
from energy_cost.types import TzInfo


class CustomerType(StrEnum):
    """The kind of customer, as fees differ per customer type."""

    RESIDENTIAL = "residential"
    NON_RESIDENTIAL = "non_residential"
    PROTECTED = "protected"


class ConnectionType(StrEnum):
    """The energy carrier of a grid connection."""

    ELECTRICITY = "electricity"
    GAS = "gas"


class Supplier(EnergyCostModel, RegistryMixin[str, "Supplier"]):
    """An energy supplier and the products it offers."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    products: dict[str, Tariff]
    """Supplier tariff per product name."""

    @classmethod
    def _registry(cls) -> dict[str, "Supplier"]:
        return Catalog.current().suppliers


class RegionalData(EnergyCostModel, RegistryMixin[tuple[str, ConnectionType], "RegionalData"]):
    """The fees, distributors and taxes that apply in a region for one connection type."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    fees: dict[CustomerType, Tariff | list[Tariff]]
    """Government fees per customer type."""
    distributors: dict[str, Tariff | list[Tariff]]
    """Grid tariff per distributor name."""
    taxes: Tax | list[Tax]
    """Taxes that apply in the region."""
    timezone: TzInfo
    """Timezone in which the region's tariffs are defined."""
    capacity_rule: CapacityRule | None = None
    """How peak capacity is billed in the region, if it is."""

    @classmethod
    def _registry(cls) -> dict[tuple[str, ConnectionType], "RegionalData"]:
        return Catalog.current().regions
