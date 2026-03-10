"""Regulated grid tariff models with YAML loading."""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from functools import cache
from importlib.resources import files
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, Field

from energy_cost.models._enums import Carrier, Direction

if TYPE_CHECKING:
    from energy_cost.models.connection import ConnectionInfo


class GridComponentType(StrEnum):
    DISTRIBUTION_VARIABLE = "distribution_variable"
    DISTRIBUTION_FIXED = "distribution_fixed"
    DATABEHEER = "databeheer"
    PUBLIC_SERVICE = "public_service"
    SYSTEM_MANAGEMENT = "system_management"
    CAPACITY_TARIFF = "capacity_tariff"
    OTHER = "other"


class GridUnit(StrEnum):
    EUR_PER_KWH = "EUR/kWh"
    EUR_PER_YEAR = "EUR/year"
    EUR_PER_KW_MONTH = "EUR/kW/month"
    EUR_PER_KW_YEAR = "EUR/kW/year"
    EUR_PER_MONTH = "EUR/month"


class GridTariffComponent(BaseModel):
    """A single regulated grid/network tariff component."""

    name: str
    component_type: GridComponentType
    unit: GridUnit
    value: float
    conditions: dict = Field(default_factory=dict)
    vat_applicable: bool = True


class GridTariffSet(BaseModel):
    """Full set of grid tariff components for a region/DSO/carrier/direction."""

    region: str
    dso: str
    carrier: Carrier
    direction: Direction
    valid_from: date
    valid_to: date
    components: list[GridTariffComponent]

    @classmethod
    def resolve(
        cls,
        package: str,
        connection: ConnectionInfo,
        carrier: Carrier,
        direction: Direction,
    ) -> GridTariffSet:
        """Look up the applicable grid tariff set from a region YAML package.

        Parameters
        ----------
        package:
            Dotted Python package path containing ``grid.yaml``.
        connection:
            Customer connection metadata.
        carrier:
            Energy carrier to filter on.
        direction:
            Flow direction to filter on.
        """
        meter_type = connection.electricity.meter_type.value
        for item in _load_grid_config(package).tariff_sets:
            if (
                item.dso == connection.dso
                and item.carrier == carrier
                and item.direction == direction
                and item.meter_type == meter_type
            ):
                return cls(
                    region=item.region,
                    dso=item.dso,
                    carrier=item.carrier,
                    direction=item.direction,
                    valid_from=item.valid_from,
                    valid_to=item.valid_to,
                    components=item.components,
                )

        raise KeyError(f"No grid tariff set found for {(connection.dso, carrier, direction, meter_type)}")


# ---------------------------------------------------------------------------
# Private YAML parsing helpers
# ---------------------------------------------------------------------------


class _GridTariffRecord(BaseModel):
    region: str
    dso: str
    carrier: Carrier
    direction: Direction
    meter_type: str
    valid_from: date
    valid_to: date
    components: list[GridTariffComponent]


class _GridTariffConfig(BaseModel):
    tariff_sets: list[_GridTariffRecord]


@cache
def _load_grid_config(package: str, filename: str = "grid.yaml") -> _GridTariffConfig:
    config_path = files(package).joinpath(filename)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return _GridTariffConfig.model_validate(payload)
