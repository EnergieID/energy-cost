"""Tariff definition I/O helpers."""

from __future__ import annotations

from pathlib import Path

import yaml

from energy_cost.tariffs.models import TariffDefinition


def load_tariff_definition_yaml(path: str | Path) -> TariffDefinition:
    """Load a `TariffDefinition` from a YAML file."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return TariffDefinition.model_validate(data)
