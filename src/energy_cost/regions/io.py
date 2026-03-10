"""Generic region YAML loaders and resolvers."""

from __future__ import annotations

from datetime import date
from functools import cache
from importlib.resources import files

import yaml
from pydantic import BaseModel

from energy_cost.core.models import ConnectionInfo
from energy_cost.enums import Carrier, Direction
from energy_cost.regions.models import GridTariffComponent, GridTariffSet, TaxRule


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


class _TaxConfig(BaseModel):
    tax_rules: list[TaxRule]


@cache
def _load_grid_config(package: str, filename: str = "grid.yaml") -> _GridTariffConfig:
    config_path = files(package).joinpath(filename)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return _GridTariffConfig.model_validate(payload)


@cache
def _load_tax_config(package: str, filename: str = "taxes.yaml") -> _TaxConfig:
    config_path = files(package).joinpath(filename)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return _TaxConfig.model_validate(payload)


def resolve_grid_tariffs_from_package(
    package: str,
    connection: ConnectionInfo,
    carrier: Carrier,
    direction: Direction,
) -> GridTariffSet:
    """Look up the applicable grid tariff set from a region package."""
    meter_type = connection.electricity.meter_type.value
    for item in _load_grid_config(package).tariff_sets:
        if (
            item.dso == connection.dso
            and item.carrier == carrier
            and item.direction == direction
            and item.meter_type == meter_type
        ):
            return GridTariffSet(
                region=item.region,
                dso=item.dso,
                carrier=item.carrier,
                direction=item.direction,
                valid_from=item.valid_from,
                valid_to=item.valid_to,
                components=item.components,
            )

    raise KeyError(f"No grid tariff set found for {(connection.dso, carrier, direction, meter_type)}")


def resolve_tax_rules_from_package(
    package: str,
    connection: ConnectionInfo,
    carrier: Carrier,
    direction: Direction,
) -> list[TaxRule]:
    """Return tax rules applicable for a given connection, carrier and direction."""
    return [
        rule
        for rule in _load_tax_config(package).tax_rules
        if rule.carrier == carrier
        and rule.direction == direction
        and (rule.customer_type is None or rule.customer_type == connection.customer_type)
    ]
