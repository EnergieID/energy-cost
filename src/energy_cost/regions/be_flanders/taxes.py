"""Tax and levy rules for Flanders."""

from __future__ import annotations

from energy_cost.models import Carrier, ConnectionInfo, Direction, TaxRule


def resolve_tax_rules(connection: ConnectionInfo, carrier: Carrier, direction: Direction) -> list[TaxRule]:
    """Return tax rules applicable for a given connection, carrier and direction."""
    return TaxRule.resolve("energy_cost.regions.be_flanders", connection, carrier, direction)
