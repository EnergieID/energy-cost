"""
Integration tests for serialisation and deserialisation of contracts.

Verifies that all example contract YAML files can be round-tripped through
model_dump() and model_validate(), producing the same contract as the original
(after from_yaml), and that the dump contains all keys present in the original YAML.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from energy_cost.contract import Contract, ContractHistory
from energy_cost.data.models import Supplier
from energy_cost.index import DataFrameIndex, Index
from energy_cost.tariff import Tariff

EXAMPLES = Path(__file__).resolve().parent.parent.parent / "examples"
CET = dt.timezone(dt.timedelta(hours=1))

_START = dt.datetime(2024, 1, 1, tzinfo=CET)
_END = dt.datetime(2026, 7, 1, tzinfo=CET)


@pytest.fixture
def fake_index() -> None:
    """Register a fake spot index covering the test period."""
    timestamps = pd.date_range(_START, _END, freq="15min", inclusive="left")
    Index.register(
        "spot",
        DataFrameIndex(pd.DataFrame({"timestamp": timestamps, "value": 50.0})),
    )


@pytest.fixture
def fake_supplier() -> None:
    """Register a test supplier with products matching the example configs."""
    Supplier.register(
        "my_supplier",
        Supplier(
            products={
                "fixed": Tariff.from_yaml(EXAMPLES / "tariffs" / "fixed.yml"),
                "dynamic": Tariff.from_yaml(EXAMPLES / "tariffs" / "dynamic.yml"),
                "injection": Tariff.from_yaml(EXAMPLES / "tariffs" / "injection.yml"),
            }
        ),
    )


def _assert_yaml_keys_in_dump(yaml_data: Any, dump: Any, path: str = "") -> None:
    """Recursively assert every key/index from yaml_data is present in dump.

    The one permitted structural transformation is the formula shorthand coercion:
    a bare formula dict (e.g. ``{constant_cost: 90.0}``) is expanded by the
    MeterFormulas + NamedFormulas validators into ``{"all": {"total": <Formula>}}``.
    When that pattern is detected we assert the specific coercion structure and
    then recurse into the innermost formula dict.
    """
    if isinstance(yaml_data, dict):
        assert isinstance(dump, dict), f"{path}: expected dict, got {type(dump).__name__}"
        if yaml_data and dump and not set(yaml_data.keys()) & set(dump.keys()):
            # The only expected cause: a bare formula dict was coerced by
            # _coerce_named_formulas (wraps in {"total": ...}).
            assert isinstance(dump, dict) and "total" in dump, (
                f"{path}.all: expected NamedFormulas coercion to produce a 'total' key, "
                f"got {set(dump.keys()) if isinstance(dump, dict) else type(dump).__name__}"
            )
            _assert_yaml_keys_in_dump(yaml_data, dump["total"], f"{path}.all.total")
            return
        for key in yaml_data:
            assert key in dump, f"{path}.{key}: key found in YAML but missing from dump"
            _assert_yaml_keys_in_dump(yaml_data[key], dump[key], f"{path}.{key}")
    elif isinstance(yaml_data, list):
        assert isinstance(dump, (list, tuple)), f"{path}: expected list, got {type(dump).__name__}"
        for i, item in enumerate(yaml_data):
            assert i < len(dump), f"{path}[{i}]: index found in YAML but missing from dump"
            _assert_yaml_keys_in_dump(item, dump[i], f"{path}[{i}]")


# ---------------------------------------------------------------------------
# Single contract configs
# ---------------------------------------------------------------------------


class TestContractSerialisation:
    @pytest.mark.parametrize(
        "config_name",
        ["simple.yml", "dynamic.yml", "injection.yml", "gas.yml", "inline.yml", "inline_full.yml"],
    )
    def test_roundtrip(
        self,
        config_name: str,
        fake_index: None,
        fake_supplier: None,
    ) -> None:
        config_path = EXAMPLES / "contracts" / config_name
        original = Contract.from_yaml(config_path)

        roundtripped = Contract.model_validate(original.model_dump())
        assert roundtripped == original

    @pytest.mark.parametrize(
        "config_name",
        ["simple.yml", "dynamic.yml", "injection.yml", "gas.yml", "inline.yml", "inline_full.yml"],
    )
    def test_dump_contains_yaml_keys(
        self,
        config_name: str,
        fake_index: None,
        fake_supplier: None,
    ) -> None:
        config_path = EXAMPLES / "contracts" / config_name
        original = Contract.from_yaml(config_path)

        with config_path.open(encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f)

        _assert_yaml_keys_in_dump(yaml_data, original.model_dump())


# ---------------------------------------------------------------------------
# Contract history configs
# ---------------------------------------------------------------------------


class TestContractHistorySerialisation:
    @pytest.mark.parametrize(
        "config_name",
        ["history.yml", "history_with_gap.yml"],
    )
    def test_roundtrip(
        self,
        config_name: str,
        fake_index: None,
        fake_supplier: None,
    ) -> None:
        config_path = EXAMPLES / "contracts" / config_name
        original = ContractHistory.from_yaml(config_path)

        roundtripped = ContractHistory.model_validate(original.model_dump())
        assert roundtripped == original

    @pytest.mark.parametrize(
        "config_name",
        ["history.yml", "history_with_gap.yml"],
    )
    def test_dump_contains_yaml_keys(
        self,
        config_name: str,
        fake_index: None,
        fake_supplier: None,
    ) -> None:
        config_path = EXAMPLES / "contracts" / config_name
        original = ContractHistory.from_yaml(config_path)

        with config_path.open(encoding="utf-8") as f:
            yaml_list = yaml.safe_load(f)

        _assert_yaml_keys_in_dump(yaml_list, original.model_dump())
