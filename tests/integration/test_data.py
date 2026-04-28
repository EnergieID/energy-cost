"""
Smoke tests for the data module.

For every region, connection type, distributor and customer type, verify that a
Contract built from the real distributor + fees + taxes data can be evaluated
against a simple constant-consumption meter without crashing and returns a
non-empty DataFrame.

No exact output values are checked – these tests guard against import errors,
YAML-structure mismatches and formula evaluation failures.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from energy_cost.contract import Contract
from energy_cost.data import ConnectionType, CustomerType, regionalData
from energy_cost.meter import Meter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CET = dt.timezone(dt.timedelta(hours=1))

_START = dt.datetime(2025, 1, 1, tzinfo=CET)
_END = dt.datetime(2026, 7, 1, tzinfo=CET)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def consumption_meter_15min() -> Meter:
    """Constant-consumption meter at 4 kW (1 kWh / 15 min) for 1.5 years."""
    timestamps = pd.date_range(_START, _END, freq="15min", inclusive="left")
    data = pd.DataFrame({"timestamp": timestamps, "value": 0.001})
    return Meter(data=data)


@pytest.fixture(scope="module")
def consumption_meter_monthly() -> Meter:
    """Constant-consumption meter at ~1 MWh/month for 1.5 years."""
    timestamps = pd.date_range(_START, _END, freq="MS", inclusive="left")
    data = pd.DataFrame({"timestamp": timestamps, "value": 1.0})
    return Meter(data=data)


# ---------------------------------------------------------------------------
# Test parametrisation
# ---------------------------------------------------------------------------

_CASES: list[tuple[str, ConnectionType, CustomerType, str]] = [
    (region, connection_type, customer_type, distributor_name)
    for region, ct_map in regionalData.items()
    for connection_type, regional_data in ct_map.items()
    for distributor_name in regional_data.distributors
    for customer_type in regional_data.fees
]


@pytest.mark.parametrize(
    "region,connection_type,customer_type,distributor_name",
    [pytest.param(r, ct, cust, dist, id=f"{r}/{ct}/{cust}/{dist}") for r, ct, cust, dist in _CASES],
)
def test_contract_produces_valid_dataframe(
    region: str,
    connection_type: ConnectionType,
    customer_type: CustomerType,
    distributor_name: str,
    consumption_meter_15min: Meter,
    consumption_meter_monthly: Meter,
) -> None:
    """Contract with fees + distributor + taxes returns a non-empty DataFrame."""
    regional_data = regionalData[region][connection_type]

    # Gas banded formulas group by year: monthly data is sufficient and much faster.
    meter = consumption_meter_monthly if connection_type == ConnectionType.GAS else consumption_meter_15min

    contract = Contract(
        fees=regional_data.fees[customer_type],
        distributor=regional_data.distributors[distributor_name],
        taxes=regional_data.taxes,
        timezone=CET,
    )

    result = contract.calculate_cost(meters=[meter])

    assert isinstance(result, pd.DataFrame), "calculate_cost must return a DataFrame"
    assert not result.empty, "result DataFrame must not be empty"
