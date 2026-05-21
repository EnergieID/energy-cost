"""
Integration tests for example contract and contract-history YAML configs.

Verifies that the example configs can be loaded, resolved, and applied
against a simple constant-consumption meter without crashing.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from energy_cost.contract import Contract, ContractHistory
from energy_cost.data.models import Supplier
from energy_cost.index import DataFrameIndex, Index
from energy_cost.meter import Meter, TimeseriesFrame
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


@pytest.fixture(scope="module")
def consumption_meter() -> Meter:
    """Constant-consumption meter at 4 kW (1 kWh / 15 min)."""
    timestamps = pd.date_range(_START, _END, freq="15min", inclusive="left")
    return Meter(power=TimeseriesFrame(pd.DataFrame({"timestamp": timestamps, "value": 0.001})))


# ---------------------------------------------------------------------------
# Single contract configs
# ---------------------------------------------------------------------------


class TestContractConfigs:
    @pytest.mark.parametrize(
        "config_name",
        ["simple.yml", "dynamic.yml", "injection.yml", "gas.yml", "inline.yml", "inline_full.yml"],
    )
    def test_load_and_apply(
        self,
        config_name: str,
        fake_index: None,
        fake_supplier: None,
        consumption_meter: Meter,
    ) -> None:
        contract = Contract.from_yaml(EXAMPLES / "contracts" / config_name)
        result = contract.apply(consumption_meter)

        assert isinstance(result, pd.DataFrame)
        assert not result.empty


# ---------------------------------------------------------------------------
# Contract history configs
# ---------------------------------------------------------------------------


class TestContractHistoryConfigs:
    def test_history(self, fake_index: None, fake_supplier: None, consumption_meter: Meter) -> None:
        history = ContractHistory.from_yaml(EXAMPLES / "contracts" / "history.yml")

        assert len(history.root) == 2
        result = history.apply(
            consumption_meter,
            start=dt.datetime(2024, 6, 1, tzinfo=CET),
            end=dt.datetime(2026, 1, 1, tzinfo=CET),
        )

        assert result is not None
        assert not result.empty

    def test_history_with_gap(self, fake_index: None, fake_supplier: None, consumption_meter: Meter) -> None:
        history = ContractHistory.from_yaml(EXAMPLES / "contracts" / "history_with_gap.yml")

        assert len(history.root) == 2
        result = history.apply(
            consumption_meter,
            start=dt.datetime(2024, 6, 1, tzinfo=CET),
            end=dt.datetime(2026, 1, 1, tzinfo=CET),
        )

        assert result is not None
        assert not result.empty
