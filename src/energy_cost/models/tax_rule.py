"""Tax and levy rule models with YAML loading."""

from __future__ import annotations

from enum import StrEnum
from functools import cache
from importlib.resources import files
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel

from energy_cost.models._enums import Carrier, CustomerType, Direction

if TYPE_CHECKING:
    from energy_cost.models.connection import ConnectionInfo


class TaxKind(StrEnum):
    VARIABLE = "variable"
    FIXED_MONTHLY = "fixed_monthly"
    FIXED_YEARLY = "fixed_yearly"
    TIERED = "tiered"


class TaxUnit(StrEnum):
    EUR_PER_KWH = "EUR/kWh"
    EUR_PER_MONTH = "EUR/month"
    EUR_PER_YEAR = "EUR/year"


class TaxTier(BaseModel):
    """A single tier within a tiered tax rule."""

    from_kwh: float
    to_kwh: float
    eur_per_kwh: float


class TaxRule(BaseModel):
    """A tax or levy rule applicable to energy consumption."""

    name: str
    carrier: Carrier
    direction: Direction
    customer_type: CustomerType | None = None

    kind: TaxKind
    unit: TaxUnit | None = None
    value: float | None = None
    tiers: list[TaxTier] | None = None

    vat_applicable: bool = True

    @classmethod
    def resolve(
        cls,
        package: str,
        connection: ConnectionInfo,
        carrier: Carrier,
        direction: Direction,
    ) -> list[TaxRule]:
        """Return tax rules applicable for a given connection from a region YAML package.

        Parameters
        ----------
        package:
            Dotted Python package path containing ``taxes.yaml``.
        connection:
            Customer connection metadata.
        carrier:
            Energy carrier to filter on.
        direction:
            Flow direction to filter on.
        """
        return [
            rule
            for rule in _load_tax_config(package).tax_rules
            if rule.carrier == carrier
            and rule.direction == direction
            and (rule.customer_type is None or rule.customer_type == connection.customer_type)
        ]


# ---------------------------------------------------------------------------
# Private YAML parsing helpers
# ---------------------------------------------------------------------------


class _TaxConfig(BaseModel):
    tax_rules: list[TaxRule]


@cache
def _load_tax_config(package: str, filename: str = "taxes.yaml") -> _TaxConfig:
    config_path = files(package).joinpath(filename)
    with config_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    return _TaxConfig.model_validate(payload)
