"""Regulated grid tariffs for Flanders (Fluvius)."""

from __future__ import annotations

from energy_cost.core.models import ConnectionInfo
from energy_cost.enums import Carrier, Direction
from energy_cost.regions.io import resolve_grid_tariffs_from_package
from energy_cost.regions.models import GridTariffSet


def resolve_grid_tariffs(connection: ConnectionInfo, carrier: Carrier, direction: Direction) -> GridTariffSet:
    """Look up the applicable grid tariff set for a connection."""
    return resolve_grid_tariffs_from_package("energy_cost.regions.be_flanders", connection, carrier, direction)
