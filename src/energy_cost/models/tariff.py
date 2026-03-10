"""Supplier-agnostic tariff definition models.

Tariff definitions describe a supplier's product as data — no supplier-specific
logic lives in the library.  Values use tariff-card conventions:

- ``indexed_linear``: price in c€/kWh = coef × index(EUR/MWh) + add_cents_per_kwh
- ``fixed_per_kwh``: flat rate in EUR/kWh
- ``fixed_per_year`` / ``fixed_per_month``: standing charges
"""

from __future__ import annotations

from datetime import date
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from energy_cost.models._enums import Carrier, Direction
from energy_cost.models.connection import MeterRegister


class ContractType(StrEnum):
    DYNAMIC = "dynamic"
    VARIABLE = "variable"
    FIXED = "fixed"


class Channel(StrEnum):
    KWH = "kwh"
    KWH_DAY = "kwh_day"
    KWH_NIGHT = "kwh_night"


class TimeOfUse(StrEnum):
    SINGLE = "single"
    DAY_NIGHT = "day_night"


class PricingKind(StrEnum):
    FIXED_PER_KWH = "fixed_per_kwh"
    INDEXED_LINEAR = "indexed_linear"
    FIXED_PER_YEAR = "fixed_per_year"
    FIXED_PER_MONTH = "fixed_per_month"


class TariffComponentType(StrEnum):
    ENERGY = "energy"
    SURCHARGE_GREEN = "surcharge_green"
    SURCHARGE_WKK = "surcharge_wkk"
    FIXED_FEE = "fixed_fee"
    INJECTION_REMUNERATION = "injection_remuneration"
    OTHER = "other"


class PricingRule(BaseModel):
    """How a single tariff component is priced."""

    kind: PricingKind

    # indexed_linear: c€/kWh = coef * market_price(EUR/MWh) + add_cents_per_kwh
    index_name: str | None = None
    coef: float | None = None
    add_cents_per_kwh: float | None = None

    # fixed_per_kwh
    eur_per_kwh: float | None = None

    # fixed_per_year
    eur_per_year: float | None = None

    # fixed_per_month
    eur_per_month: float | None = None

    def compute_eur_per_kwh(self, market_price_eur_per_mwh: float) -> float:
        """Evaluate the pricing rule and return EUR/kWh.

        Only valid for ``indexed_linear`` and ``fixed_per_kwh`` kinds.
        """
        if self.kind == PricingKind.INDEXED_LINEAR:
            assert self.coef is not None and self.add_cents_per_kwh is not None
            return (self.coef * market_price_eur_per_mwh + self.add_cents_per_kwh) / 100.0
        if self.kind == PricingKind.FIXED_PER_KWH:
            assert self.eur_per_kwh is not None
            return self.eur_per_kwh
        raise ValueError(f"compute_eur_per_kwh not applicable for kind={self.kind}")


class TariffComponent(BaseModel):
    """A single billable component within a tariff definition."""

    component_type: TariffComponentType
    channel: Channel | None = None
    pricing: PricingRule
    vat_applicable: bool = True


class TariffDefinition(BaseModel):
    """A complete supplier-agnostic tariff product definition."""

    name: str
    carrier: Carrier
    direction: Direction
    contract_type: ContractType
    settlement_interval_minutes: int = 15
    meter_register_required: MeterRegister | None = None
    time_of_use: TimeOfUse | None = None
    components: list[TariffComponent]
    valid_from: date
    valid_to: date
    notes: dict = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, path: str | Path) -> TariffDefinition:
        """Load a ``TariffDefinition`` from a YAML file."""
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        return cls.model_validate(data)
