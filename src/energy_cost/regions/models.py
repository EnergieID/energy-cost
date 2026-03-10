"""Models for regulated grid tariffs and tax rules.

These models represent the normalised internal dataset format.  Region-specific
modules (e.g. ``be_flanders``) provide *instances* of these models loaded from
published DSO/regulator data.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from energy_cost.enums import Carrier, CustomerType, Direction, GridComponentType, GridUnit, TaxKind, TaxUnit


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
