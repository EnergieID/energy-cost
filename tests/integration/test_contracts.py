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
from energy_cost.data.models import ConnectionType, CustomerType, Supplier
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
    return Meter(measurements=TimeseriesFrame(pd.DataFrame({"timestamp": timestamps, "value": 0.001})))


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


def test_capacity_tariff():
    contract = Contract(
        region="be_flanders",
        distributor_key="fluvius_kempen",
        connection_type=ConnectionType.ELECTRICITY,
        customer_type=CustomerType.RESIDENTIAL,
    )
    capacity = [
        8.374,
        9.424,
        8.805,
        8.973,
        8.12,
        8.63,
        8.645,
        8.457,
        9.413,
        7.215,
        8.53,
        8.412,
        6.382,
        8.204,
        6.371,
        7.966,
        7.983,
        8.504,
        8.975,
        9.674,
        7.28,
        8.407,
        6.095,
        6.503,
        5.74,
        5.53,
        8.227,
        7.712,
        7.027,
        6.747,
        7.601,
        7.829,
        6.109,
        6.514,
        5.955,
        6.176,
        4.22,
    ]

    capacity = [c / 1000 for c in capacity]  # Convert from W to MW
    power = [
        c * 24 * 30 for c in capacity
    ]  # a power consumption equal to full capacity for 30 days, to avoid hitting max capacity limits in the tariff rules

    dates = [
        "2023-06-01T00:00:00+02:00",
        "2023-07-01T00:00:00+02:00",
        "2023-08-01T00:00:00+02:00",
        "2023-09-01T00:00:00+02:00",
        "2023-10-01T00:00:00+02:00",
        "2023-11-01T00:00:00+01:00",
        "2023-12-01T00:00:00+01:00",
        "2024-01-01T00:00:00+01:00",
        "2024-02-01T00:00:00+01:00",
        "2024-03-01T00:00:00+01:00",
        "2024-04-01T00:00:00+02:00",
        "2024-05-01T00:00:00+02:00",
        "2024-06-01T00:00:00+02:00",
        "2024-07-01T00:00:00+02:00",
        "2024-08-01T00:00:00+02:00",
        "2024-09-01T00:00:00+02:00",
        "2024-10-01T00:00:00+02:00",
        "2024-11-01T00:00:00+01:00",
        "2024-12-01T00:00:00+01:00",
        "2025-01-01T00:00:00+01:00",
        "2025-02-01T00:00:00+01:00",
        "2025-03-01T00:00:00+01:00",
        "2025-04-01T00:00:00+02:00",
        "2025-05-01T00:00:00+02:00",
        "2025-06-01T00:00:00+02:00",
        "2025-07-01T00:00:00+02:00",
        "2025-08-01T00:00:00+02:00",
        "2025-09-01T00:00:00+02:00",
        "2025-10-01T00:00:00+02:00",
        "2025-11-01T00:00:00+01:00",
        "2025-12-01T00:00:00+01:00",
        "2026-01-01T00:00:00+01:00",
        "2026-02-01T00:00:00+01:00",
        "2026-03-01T00:00:00+01:00",
        "2026-04-01T00:00:00+02:00",
        "2026-05-01T00:00:00+02:00",
        "2026-06-01T00:00:00+02:00",
    ]

    meter = Meter(
        measurements=TimeseriesFrame(
            pd.DataFrame({"timestamp": pd.to_datetime(dates, format="ISO8601", utc=True), "value": power})
        ),
        capacity=TimeseriesFrame(
            pd.DataFrame({"timestamp": pd.to_datetime(dates, format="ISO8601", utc=True), "value": capacity})
        ),
    )

    result = contract.apply(meter, start=dt.datetime(2025, 1, 1, tzinfo=CET), end=dt.datetime(2026, 7, 1, tzinfo=CET))

    expected = [
        36.70,
        32.42,
        36.35,
        34.29,
        34.71,
        33.36,
        33.47,
        34.17,
        32.97,
        33.71,
        31.98,
        32.53,
        33.69,
        30.01,
        32.48,
        31.37,
        32.29,
        30.66,
    ]

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    assert ("distributor", "capacity", "total") in result.columns
    for idx, expected_value in enumerate(expected):
        actual_value = result[("distributor", "capacity", "total")].iloc[idx]
        assert actual_value == pytest.approx(expected_value, rel=1e-2)
