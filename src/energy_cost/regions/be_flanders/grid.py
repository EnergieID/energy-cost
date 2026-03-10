"""Regulated grid tariffs for Flanders (Fluvius)."""

from __future__ import annotations

from energy_cost.models import Carrier, ConnectionInfo, Direction, GridTariffSet


def resolve_grid_tariffs(connection: ConnectionInfo, carrier: Carrier, direction: Direction) -> GridTariffSet:
    """Look up the applicable grid tariff set for a connection."""
    return GridTariffSet.resolve("energy_cost.regions.be_flanders", connection, carrier, direction)
