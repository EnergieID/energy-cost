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

import isodate
import pandas as pd
import pytest

from energy_cost.contract import Contract
from energy_cost.data import ConnectionType, CustomerType, RegionalData
from energy_cost.meter import Meter, TimeseriesFrame

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
    return Meter(measurements=TimeseriesFrame(data))


# ---------------------------------------------------------------------------
# Test parametrisation
# ---------------------------------------------------------------------------

_CASES: list[tuple[str, ConnectionType, CustomerType, str]] = [
    (region, connection_type, customer_type, distributor_name)
    for (region, connection_type), regional_data in RegionalData.items()
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
) -> None:
    """Contract with fees + distributor + taxes returns a non-empty DataFrame."""
    meter = consumption_meter_15min

    contract = Contract(
        region=region,
        connection_type=connection_type,
        customer_type=customer_type,
        distributor_key=distributor_name,
    )

    result = contract.apply(meter)

    assert isinstance(result, pd.DataFrame), "apply must return a DataFrame"
    assert not result.empty, "result DataFrame must not be empty"


def test_p7d_resolution_produces_no_nan_values() -> None:
    """Regression test: weekly (P7D) resolution must not produce NaN values.

    P1M capacity and P1Y periodic fees are bridged to P7D via their common divisor
    P1D: costs are first distributed from the coarse calendar period to daily bins,
    then daily bins are summed into weekly bins.
    """
    CET = dt.timezone(dt.timedelta(hours=1))
    start = dt.datetime(2025, 1, 1, tzinfo=CET)
    end = dt.datetime(2025, 2, 1, tzinfo=CET)

    timestamps = [
        pd.Timestamp("2025-01-01T00:00:00+01:00"),
        pd.Timestamp("2025-01-01T00:15:00+01:00"),
    ]
    meter = Meter(measurements=TimeseriesFrame(pd.DataFrame({"timestamp": timestamps, "value": [150.5, 75.3]})))

    contract = Contract(
        region="be_flanders",
        connection_type=ConnectionType.ELECTRICITY,
        customer_type=CustomerType.RESIDENTIAL,
        distributor_key="fluvius_antwerpen",
    )

    result = contract.apply(meter, start=start, end=end, output_resolution=isodate.parse_duration("P7D"))

    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    # Five weekly buckets expected in January 2025
    assert len(result) == 5, f"Expected 5 weekly rows, got {len(result)}"
    # No row may contain any NaN — all weekly bins must carry distributed costs
    nan_rows = result[result.isna().any(axis=1)]
    assert nan_rows.empty, f"Unexpected NaN values in rows:\n{nan_rows}"


def test_regression_gas_gives_correct_fees_in_october() -> None:
    """Regression test: gas fees in BE Flanders should be correct in October."""
    contract = Contract(
        region="be_flanders", connection_type=ConnectionType.GAS, customer_type=CustomerType.RESIDENTIAL
    )

    meter = Meter(
        measurements=TimeseriesFrame(
            {
                "timestamp": pd.to_datetime(
                    [
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
                    ],
                    format="ISO8601",
                    utc=True,
                ),
                "value": 1,
            }
        )
    )

    result = contract.apply(meter, output_resolution=isodate.parse_duration("P1M"))
    assert isinstance(result, pd.DataFrame)
    assert not result.empty

    # for october, november and december 2025, the fees should be:
    # (fees, consumption, energy_contribution) = 0.9978
    # (fees, consumption, excise) = 8.23

    for month in ["2025-10", "2025-11", "2025-12"]:
        row = result[result["timestamp"].dt.strftime("%Y-%m") == month]
        assert not row.empty, f"No data for month {month}"
        assert row[("fees", "consumption", "energy_contribution")].iloc[0] == pytest.approx(0.9978)
        assert row[("fees", "consumption", "excise")].iloc[0] == pytest.approx(8.23)
