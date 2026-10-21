from collections.abc import Iterator
from typing import Any, Literal

import pytest
from pydantic import TypeAdapter

from energy_cost import CapacityRule, Contract, ContractHistory, Tariff, Tax
from energy_cost.data.models import RegionalData, Supplier
from energy_cost.formula import Formula

ROOTS = {
    "Contract": TypeAdapter(Contract),
    "ContractHistory": TypeAdapter(ContractHistory),
    "Tariff": TypeAdapter(Tariff),
    "Tax": TypeAdapter(Tax),
    "CapacityRule": TypeAdapter(CapacityRule),
    "Supplier": TypeAdapter(Supplier),
    "RegionalData": TypeAdapter(RegionalData),
    "Formula": TypeAdapter(Formula),
}


def _models(schema: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    """Yield every named definition in *schema*, including the root itself when it is not just a union."""
    if "oneOf" not in schema and "anyOf" not in schema:
        yield schema.get("title", "<root>"), schema
    yield from schema.get("$defs", {}).items()


@pytest.mark.parametrize("mode", ["validation", "serialization"])
@pytest.mark.parametrize("root", ROOTS)
def test_every_model_and_field_has_a_description(root: str, mode: Literal["validation", "serialization"]) -> None:
    schema = ROOTS[root].json_schema(mode=mode)
    missing = []
    for name, model in _models(schema):
        if "description" not in model:
            missing.append(name)
        for field, prop in model.get("properties", {}).items():
            if "description" not in prop:
                missing.append(f"{name}.{field}")
    assert not missing, f"missing descriptions: {missing}"


def test_catalog_references_have_a_format() -> None:
    contract = Contract.model_json_schema()["properties"]
    formats = {
        name: next(option["format"] for option in contract[name]["anyOf"] if "format" in option)
        for name in ["region", "distributor_key", "supplier_key", "product_key"]
    }
    assert formats == {
        "region": "region-key",
        "distributor_key": "distributor-key",
        "supplier_key": "supplier-key",
        "product_key": "product-key",
    }
    assert contract["timezone"]["format"] == "timezone"

    defs = TypeAdapter(Formula).json_schema()["$defs"]
    assert defs["IndexAdder"]["properties"]["index"]["format"] == "index-name"
    assert defs["PeriodicFormula"]["properties"]["period"]["format"] == "duration"


def test_advanced_fields_are_marked() -> None:
    defs = TypeAdapter(Formula).json_schema()["$defs"]
    assert defs["IndexFormula"]["properties"]["capacity_based"]["x-advanced"] is True
    assert defs["TieredFormula"]["properties"]["mode"]["x-advanced"] is True
    assert "x-advanced" not in defs["IndexFormula"]["properties"]["constant_cost"]

    contract = Contract.model_json_schema()["properties"]
    assert contract["timezone"]["x-advanced"] is True
    assert "x-advanced" not in contract["region"]


def test_tariff_version_end_is_not_in_schema() -> None:
    tariff_version = Tariff.model_json_schema()["$defs"]["TariffVersion"]
    assert "end" not in tariff_version["properties"]
