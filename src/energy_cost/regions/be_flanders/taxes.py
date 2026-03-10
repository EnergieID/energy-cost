"""Tax and levy rules for Flanders."""

from __future__ import annotations

from energy_cost.core.models import ConnectionInfo
from energy_cost.enums import Carrier, Direction
from energy_cost.regions.io import resolve_tax_rules_from_package
from energy_cost.regions.models import TaxRule


def resolve_tax_rules(connection: ConnectionInfo, carrier: Carrier, direction: Direction) -> list[TaxRule]:
    """Return tax rules applicable for a given connection, carrier and direction."""
    return resolve_tax_rules_from_package("energy_cost.regions.be_flanders", connection, carrier, direction)
