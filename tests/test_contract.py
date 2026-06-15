from __future__ import annotations

import datetime as dt
from typing import Literal
from zoneinfo import ZoneInfo

import isodate
import pandas as pd
import pytest
from pydantic import ValidationError

from energy_cost.contract import Contract, ContractHistory
from energy_cost.data import ConnectionType, CustomerType
from energy_cost.data.models import Supplier
from energy_cost.formula import IndexFormula, PeriodicFormula
from energy_cost.meter import CostGroup, Meter, MeterType, PowerDirection, TariffCategory, TimeseriesFrame
from energy_cost.tariff import Tariff
from energy_cost.tariff_version import TariffVersion
from energy_cost.tax import Tax, TaxVersion

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tariff(
    *,
    energy_rate: float = 100.0,
    injection_rate: float | None = None,
    daily_fixed: float | None = None,
    start: dt.datetime | None = None,
) -> Tariff:
    """Build a simple constant-cost tariff."""
    version_start = start or dt.datetime(2025, 1, 1, 0, 0)
    consumption: dict = {"energy": IndexFormula(constant_cost=energy_rate)}
    injection: dict = {"energy": IndexFormula(constant_cost=injection_rate)} if injection_rate is not None else {}
    fixed: dict = (
        {"fixed_fee": PeriodicFormula(period=isodate.parse_duration("P1D"), constant_cost=daily_fixed)}
        if daily_fixed is not None
        else {}
    )
    return Tariff(
        [
            TariffVersion(
                start=version_start,
                consumption=consumption,
                injection=injection,
                fixed=fixed,
            )
        ]
    )


def _consumption(timestamps: pd.DatetimeIndex, value: float = 1.0) -> pd.DataFrame:
    return pd.DataFrame({"timestamp": timestamps, "value": value})


# ---------------------------------------------------------------------------
# Tariff.apply – consumption only
# ---------------------------------------------------------------------------


def test_apply_consumption_multiplies_rate_and_sums_to_month() -> None:
    """Cost = quantity × rate; rows are aggregated to calendar-monthly buckets."""
    tariff = _tariff(energy_rate=10.0)
    # Full month of January 2025 at 15min resolution
    timestamps = pd.date_range("2025-01-01", "2025-02-01", freq="15min", inclusive="left", tz=dt.UTC)
    consumption = _consumption(timestamps, value=2.0)

    result = tariff.apply(Meter(measurements=TimeseriesFrame(consumption)))

    assert result is not None
    assert len(result) == 1
    # 2976 intervals × 2 MWh × 10 €/MWh (31 days × 96 intervals/day)
    assert result[(CostGroup.CONSUMPTION, "energy")].iloc[0] == pytest.approx(2976 * 2.0 * 10.0)
    assert result[("total", "total")].iloc[0] == pytest.approx(2976 * 2.0 * 10.0)


def test_apply_two_months_produces_two_rows() -> None:
    """One output row per calendar month when data spans two months."""
    tariff = _tariff(energy_rate=5.0)
    # Full January + February 2025
    timestamps = pd.date_range("2025-01-01", "2025-03-01", freq="15min", inclusive="left", tz=dt.UTC)
    consumption = _consumption(timestamps, value=1.0)

    result = tariff.apply(Meter(measurements=TimeseriesFrame(consumption)))

    assert result is not None
    assert len(result) == 2
    # Jan: 2976 intervals × 1 × 5 (31 days)
    assert result[(CostGroup.CONSUMPTION, "energy")].iloc[0] == pytest.approx(2976 * 5.0)
    # Feb: 2688 intervals × 1 × 5 (28 days in 2025)
    assert result[(CostGroup.CONSUMPTION, "energy")].iloc[1] == pytest.approx(2688 * 5.0)


def test_apply_explicit_start_end_restricts_billing_period() -> None:
    """Data outside [start, end) does not contribute to cost."""
    tariff = _tariff(energy_rate=10.0)
    timestamps = pd.date_range("2025-01-01", periods=4, freq="15min", tz=dt.UTC)
    consumption = _consumption(timestamps, value=1.0)

    # Only include the first two intervals
    result = tariff.apply(
        Meter(measurements=TimeseriesFrame(consumption)),
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 0, 30),
        output_resolution=dt.timedelta(minutes=30),
    )

    assert result is not None
    assert len(result) == 1
    assert result[(CostGroup.CONSUMPTION, "energy")].iloc[0] == pytest.approx(20.0)  # 2 × 1 × 10


def test_apply_uses_full_consumption_for_capacity_but_slices_output(tmp_path) -> None:
    """apply_capacity_cost receives the full DataFrame; billed capacity is restricted to
    [start, end)."""
    capacity_tariff_yaml = tmp_path / "cap.yml"
    capacity_tariff_yaml.write_text(
        "- start: 2025-01-01T00:00:00\n"
        "  consumption:\n"
        "    energy:\n"
        "      constant_cost: 0.0\n"
        "  capacity:\n"
        "    formula:\n"
        "      constant_cost: 1.0\n"
        "      capacity_based: true\n",
        encoding="utf-8",
    )
    from energy_cost.capacity import CapacityRule

    cap_rule = CapacityRule(
        measurement_period=isodate.parse_duration("PT15M"),
        billing_period=isodate.parse_duration("P1M"),
    )
    tariff = Tariff.from_yaml(capacity_tariff_yaml)

    # Provide data for two months; only bill the second month.
    jan_ts = pd.date_range("2025-01-01", periods=4, freq="15min", tz=dt.UTC)
    feb_ts = pd.date_range("2025-02-01", periods=4, freq="15min", tz=dt.UTC)
    all_ts = pd.concat([pd.Series(jan_ts), pd.Series(feb_ts)], ignore_index=True)
    raw_df = pd.DataFrame({"timestamp": all_ts, "value": 5.0})
    raw_meter = Meter(measurements=TimeseriesFrame(raw_df))
    consumption = cap_rule.apply(raw_meter)
    result = tariff.apply(
        consumption,
        start=dt.datetime(2025, 2, 1, 0, 0),
        end=dt.datetime(2025, 3, 1, 0, 0),
    )

    # February should have the capacity cost; January should have none.
    assert result is not None
    feb_row = result[result["timestamp"] == pd.Timestamp("2025-02-01", tz=dt.UTC)]
    assert len(feb_row) == 1
    # Capacity column should be present with a non-NaN value.
    assert (CostGroup.CAPACITY, "total") in result.columns
    assert pd.notna(feb_row[(CostGroup.CAPACITY, "total")].iloc[0])
    assert feb_row[(CostGroup.CAPACITY, "total")].iloc[0] > 0


def test_apply_custom_output_resolution() -> None:
    """Costs can be aggregated to a daily resolution."""
    tariff = _tariff(energy_rate=10.0)
    # Full day at 15min resolution (96 intervals)
    timestamps = pd.date_range("2025-01-01", "2025-01-02", freq="15min", inclusive="left", tz=dt.UTC)
    consumption = _consumption(timestamps, value=1.0)

    result = tariff.apply(Meter(measurements=TimeseriesFrame(consumption)), output_resolution=dt.timedelta(days=1))

    assert result is not None
    assert len(result) == 1
    # 96 intervals × 1 MWh × 10 €/MWh = 960 €
    assert result[(CostGroup.CONSUMPTION, "energy")].iloc[0] == pytest.approx(960.0)


def test_apply_omits_consumption_frame_when_tariff_has_no_consumption_formulas(tmp_path) -> None:
    """A capacity-only tariff produces no consumption column in the output."""
    cap_yaml = tmp_path / "cap_only.yml"
    cap_yaml.write_text(
        "- start: 2025-01-01T00:00:00\n"
        "  capacity:\n"
        "    formula:\n"
        "      constant_cost: 1.0\n"
        "      capacity_based: true\n",
        encoding="utf-8",
    )
    from energy_cost.capacity import CapacityRule

    cap_rule = CapacityRule(
        measurement_period=isodate.parse_duration("PT15M"),
        billing_period=isodate.parse_duration("P1M"),
    )
    tariff = Tariff.from_yaml(cap_yaml)
    # Use 2 months so capacity has 2 rows (required for resolution detection)
    timestamps = pd.date_range("2025-01-01", "2025-03-01", freq="15min", tz=dt.UTC, inclusive="left")
    raw_meter = Meter(measurements=TimeseriesFrame(_consumption(timestamps, value=5.0)))
    result = tariff.apply(
        cap_rule.apply(raw_meter),
        start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
        end=dt.datetime(2025, 2, 1, tzinfo=dt.UTC),
    )

    assert result is not None
    data_cols = [c for c in result.columns if c != "timestamp"]
    assert not any(c[0] == CostGroup.CONSUMPTION for c in data_cols)
    assert (CostGroup.CAPACITY, "total") in result.columns
    assert ("total", "total") in result.columns


def test_apply_column_structure_is_two_level_multiindex() -> None:
    """The output columns form a two-level MultiIndex by default (excluding the timestamp column)."""
    tariff = _tariff(energy_rate=10.0)
    timestamps = pd.date_range("2025-01-01", periods=2, freq="15min", tz=dt.UTC)
    result = tariff.apply(Meter(measurements=TimeseriesFrame(_consumption(timestamps))))

    assert result is not None
    data_cols = [c for c in result.columns if c != "timestamp"]
    assert all(isinstance(c, tuple) and len(c) == 2 for c in data_cols)
    assert (CostGroup.CONSUMPTION, "energy") in result.columns
    assert ("total", "total") in result.columns


# ---------------------------------------------------------------------------
# Tariff.apply – injection
# ---------------------------------------------------------------------------


def test_apply_with_injection_adds_injection_columns() -> None:
    """When injection data is supplied the result includes injection cost columns."""
    tariff = _tariff(energy_rate=10.0, injection_rate=5.0)
    # Full month of January 2025
    timestamps = pd.date_range("2025-01-01", "2025-02-01", freq="15min", inclusive="left", tz=dt.UTC)
    consumption = _consumption(timestamps, value=2.0)
    injection = _consumption(timestamps, value=1.0)

    result = tariff.apply(
        Meter(measurements=TimeseriesFrame(consumption)),
        injection=Meter(measurements=TimeseriesFrame(injection)),
    )

    assert result is not None
    assert (CostGroup.INJECTION, "energy") in result.columns
    n = 2976  # 31 days × 96 intervals/day
    # Consumption: n × 2 × 10; Injection: n × 1 × 5; Total = both
    assert result[(CostGroup.CONSUMPTION, "energy")].iloc[0] == pytest.approx(n * 2.0 * 10.0)
    assert result[(CostGroup.INJECTION, "energy")].iloc[0] == pytest.approx(n * 1.0 * 5.0)
    assert result[(CostGroup.CONSUMPTION, "total")].iloc[0] == pytest.approx(n * 2.0 * 10.0)
    assert result[(CostGroup.INJECTION, "total")].iloc[0] == pytest.approx(n * 1.0 * 5.0)
    assert result[("total", "total")].iloc[0] == pytest.approx(n * 2.0 * 10.0 + n * 1.0 * 5.0)


# ---------------------------------------------------------------------------
# Tariff.apply – fixed / periodic costs
# ---------------------------------------------------------------------------


def test_apply_includes_fixed_costs_prorated_per_output_period() -> None:
    """Periodic costs appear under the ``fixed`` group for the full snapped billing period."""
    tariff = _tariff(energy_rate=0.0, daily_fixed=24.0)
    # 2 hours of data inside January 2025; billing is snapped to the full month
    timestamps = pd.date_range("2025-01-01", periods=8, freq="15min", tz=dt.UTC)
    consumption = _consumption(timestamps)

    result = tariff.apply(Meter(measurements=TimeseriesFrame(consumption)))

    assert result is not None
    # Billing period is snapped to full January (31 days) → 24 * 31 = 744 €
    assert (CostGroup.FIXED, "total") in result.columns
    assert result[(CostGroup.FIXED, "total")].iloc[0] == pytest.approx(744.0)


# ---------------------------------------------------------------------------
# Contract.calculate
# ---------------------------------------------------------------------------


def test_contract_combines_supplier_and_distributor() -> None:
    """calculate() returns a single DataFrame with supplier + distributor data."""
    contract = Contract(
        supplier=_tariff(energy_rate=100.0),
        distributor=_tariff(energy_rate=50.0),
    )
    timestamps = pd.date_range("2025-01-01", periods=2, freq="15min", tz=dt.UTC)
    consumption = _consumption(timestamps, value=1.0)

    result = contract.apply(Meter(measurements=TimeseriesFrame(consumption)))

    assert (TariffCategory.SUPPLIER, CostGroup.CONSUMPTION, "energy") in result.columns
    assert (TariffCategory.DISTRIBUTOR, CostGroup.CONSUMPTION, "energy") in result.columns


def test_contract_taxes_applied_to_all_tariffs() -> None:
    """Taxes are computed on the sum of ALL tariff totals, including fees."""
    contract = Contract(
        supplier=_tariff(energy_rate=100.0),
        distributor=_tariff(energy_rate=100.0),
        fees=_tariff(energy_rate=50.0),
        taxes=Tax([TaxVersion(start=dt.datetime(2025, 1, 1), default=0.10)]),
    )
    # Full month of January
    timestamps = pd.date_range("2025-01-01", "2025-02-01", freq="15min", inclusive="left", tz=dt.UTC)
    consumption = _consumption(timestamps, value=1.0)

    result = contract.apply(Meter(measurements=TimeseriesFrame(consumption)))

    n = 2976  # 31 days × 96
    supplier_total = result[(TariffCategory.SUPPLIER, "total", "total")].iloc[0]
    distributor_total = result[(TariffCategory.DISTRIBUTOR, "total", "total")].iloc[0]
    fees_total = result[(TariffCategory.FEES, "total", "total")].iloc[0]
    taxes = result[(TariffCategory.TAXES, "total", "total")].iloc[0]
    total_cost = result[("total", "total", "total")].iloc[0]

    assert supplier_total == pytest.approx(n * 100.0)
    assert distributor_total == pytest.approx(n * 100.0)
    assert fees_total == pytest.approx(n * 50.0)
    assert taxes == pytest.approx((n * 100.0 + n * 100.0 + n * 50.0) * 0.10)
    assert total_cost == pytest.approx(supplier_total + distributor_total + fees_total + taxes)


def test_contract_list_of_taxes() -> None:
    """Multiple Tax specs are summed independently."""
    vat = Tax([TaxVersion(start=dt.datetime(2025, 1, 1), default=0.10)])
    levy = Tax([TaxVersion(start=dt.datetime(2025, 1, 1), default=0.05)])
    contract = Contract(
        supplier=_tariff(energy_rate=100.0),
        taxes=[vat, levy],
    )
    # Full month of January
    timestamps = pd.date_range("2025-01-01", "2025-02-01", freq="15min", inclusive="left", tz=dt.UTC)
    consumption = _consumption(timestamps, value=1.0)

    result = contract.apply(Meter(measurements=TimeseriesFrame(consumption)))

    n = 2976
    # supplier: n × 100; taxes = n*100*(0.10 + 0.05)
    taxes = result[(TariffCategory.TAXES, "total", "total")].iloc[0]
    assert taxes == pytest.approx(n * 100.0 * 0.15)


def test_contract_no_fees_omits_fees_columns() -> None:
    """When no fees tariff is present the result has no fees columns."""
    contract = Contract(
        supplier=_tariff(energy_rate=100.0),
        distributor=_tariff(energy_rate=50.0),
    )
    timestamps = pd.date_range("2025-01-01", periods=2, freq="15min", tz=dt.UTC)
    result = contract.apply(Meter(measurements=TimeseriesFrame(_consumption(timestamps))))

    assert not any(isinstance(c, tuple) and c[0] == TariffCategory.FEES for c in result.columns)


def test_contract_column_structure_is_three_level_multiindex() -> None:
    """Data columns form a three-level MultiIndex by default (collapsed); timestamp is a plain column."""
    contract = Contract(
        supplier=_tariff(energy_rate=10.0),
        distributor=_tariff(energy_rate=5.0),
        taxes=Tax([TaxVersion(start=dt.datetime(2025, 1, 1), default=0.21)]),
    )
    timestamps = pd.date_range("2025-01-01", periods=2, freq="15min", tz=dt.UTC)
    result = contract.apply(Meter(measurements=TimeseriesFrame(_consumption(timestamps))))

    data_cols = [c for c in result.columns if c != "timestamp"]
    assert all(isinstance(c, tuple) and len(c) == 3 for c in data_cols)


def test_contract_total_cost_equals_manual_sum() -> None:
    """total_cost == supplier_total + distributor_total + taxes (no fees case)."""
    contract = Contract(
        supplier=_tariff(energy_rate=100.0),
        distributor=_tariff(energy_rate=50.0),
        taxes=Tax([TaxVersion(start=dt.datetime(2025, 1, 1), default=0.21)]),
    )
    # Full month of January
    timestamps = pd.date_range("2025-01-01", "2025-02-01", freq="15min", inclusive="left", tz=dt.UTC)
    consumption = _consumption(timestamps, value=2.0)

    result = contract.apply(Meter(measurements=TimeseriesFrame(consumption)))

    p = result[(TariffCategory.SUPPLIER, "total", "total")].iloc[0]
    d = result[(TariffCategory.DISTRIBUTOR, "total", "total")].iloc[0]
    taxes = result[(TariffCategory.TAXES, "total", "total")].iloc[0]
    total = result[("total", "total", "total")].iloc[0]

    n = 2976
    assert p == pytest.approx(n * 2.0 * 100.0)
    assert d == pytest.approx(n * 2.0 * 50.0)
    assert taxes == pytest.approx((p + d) * 0.21)
    assert total == pytest.approx(p + d + taxes)


# ---------------------------------------------------------------------------
# Contract.apply — timezone-aware start / end
# ---------------------------------------------------------------------------

_CET = dt.timezone(dt.timedelta(hours=1))


def _tz_contract(
    *,
    energy_rate: float = 100.0,
    daily_fixed: float | None = None,
) -> Contract:
    """Contract with tz-aware version starts, matching real YAML-based tariffs."""
    start = dt.datetime(2025, 1, 1, 0, 0, tzinfo=_CET)
    consumption: dict = {"energy": IndexFormula(constant_cost=energy_rate)}
    fixed: dict = (
        {"fixed_fee": PeriodicFormula(period=isodate.parse_duration("P1D"), constant_cost=daily_fixed)}
        if daily_fixed is not None
        else {}
    )
    version = TariffVersion(start=start, consumption=consumption, fixed=fixed)
    tariff = Tariff([version])
    return Contract(supplier=tariff, distributor=tariff, timezone=_CET)


def test_calculate_cost_output_timestamps_match_input_timezone() -> None:
    """Timestamps in the result must retain the input data's timezone, not shift to UTC."""
    contract = _tz_contract(energy_rate=10.0)
    timestamps = pd.date_range("2025-01-01T00:00:00+01:00", periods=4, freq="15min")
    consumption = pd.DataFrame({"timestamp": timestamps, "value": 1.0})

    result = contract.apply(Meter(measurements=TimeseriesFrame(consumption)))

    assert result["timestamp"].dt.tz is not None
    # Midnight CET must appear as 2025-01-01 00:00 +01:00, not 2024-12-31 23:00 UTC
    assert result["timestamp"].iloc[0] == pd.Timestamp("2025-01-01T00:00:00+01:00")


def test_calculate_cost_zoneinfo_start_end_work_correctly() -> None:
    """zoneinfo-aware start/end must be accepted without any loss of precision."""
    contract = _tz_contract(energy_rate=5.0)
    # Full month of January in CET
    timestamps = pd.date_range("2025-01-01T00:00:00+01:00", "2025-02-01T00:00:00+01:00", freq="15min", inclusive="left")
    consumption = pd.DataFrame({"timestamp": timestamps, "value": 2.0})

    z = ZoneInfo("Europe/Brussels")
    start = dt.datetime(2025, 1, 1, 0, 0, tzinfo=z)
    end = dt.datetime(2025, 2, 1, 0, 0, tzinfo=z)

    result = contract.apply(Meter(measurements=TimeseriesFrame(consumption)), start=start, end=end)

    assert result is not None
    n = len(timestamps)  # 2976
    assert result[(TariffCategory.SUPPLIER, "total", "total")].iloc[0] == pytest.approx(n * 2.0 * 5.0)
    assert result["timestamp"].iloc[0] == pd.Timestamp("2025-01-01T00:00:00+01:00")


# ---------------------------------------------------------------------------
# Tariff.apply / Contract.apply — TOU meters & multiple meters
# ---------------------------------------------------------------------------


def test_apply_capacity_costs_returns_none_when_filtered_slice_is_empty(tmp_path) -> None:
    """_apply_capacity_costs returns None when the capacity row timestamps lie entirely
    outside [billing_start, billing_end).

    With a monthly billing period and two months of data, the capacity component
    produces 2 rows. When the billing window is restricted to a window after all
    data → filtered is empty → result is None or has no capacity rows.
    """
    cap_yaml = tmp_path / "cap_annual.yml"
    cap_yaml.write_text(
        "- start: 2025-01-01T00:00:00\n"
        "  capacity:\n"
        "    formula:\n"
        "      constant_cost: 1.0\n"
        "      capacity_based: true\n",
        encoding="utf-8",
    )
    from energy_cost.capacity import CapacityRule

    cap_rule = CapacityRule(
        measurement_period=isodate.parse_duration("PT15M"),
        billing_period=isodate.parse_duration("P1M"),
    )
    tariff = Tariff.from_yaml(cap_yaml)

    # Data spans Jan + Feb so we have 2 capacity rows (required for resolution detection)
    jan_ts = pd.date_range("2025-01-01", periods=4, freq="15min", tz=dt.UTC)
    feb_ts = pd.date_range("2025-02-01", periods=4, freq="15min", tz=dt.UTC)
    all_ts = pd.concat([pd.Series(jan_ts), pd.Series(feb_ts)], ignore_index=True)
    raw_df = pd.DataFrame({"timestamp": all_ts, "value": 5.0})
    consumption = cap_rule.apply(Meter(measurements=TimeseriesFrame(raw_df)))

    # Billing window is April (after all data) → capacity rows at Jan/Feb are outside
    # [2025-04-01, 2025-05-01) → capacity should not contribute.
    result = tariff.apply(
        consumption,
        start=dt.datetime(2025, 4, 1, 0, 0, tzinfo=dt.UTC),
        end=dt.datetime(2025, 5, 1, 0, 0, tzinfo=dt.UTC),
    )

    # No in-window capacity → either None or no capacity costs.
    assert result is None or result.get((CostGroup.CAPACITY, "total"), pd.Series([0.0])).sum() == 0.0


def test_avoid_regression_on_real_world_data() -> None:
    import yaml

    from energy_cost.index import DataFrameIndex, Index

    Index.register(
        "spot",
        DataFrameIndex(
            pd.DataFrame(
                {
                    "timestamp": pd.date_range("2025-01-01", periods=4 * 24 * 365 * 2, freq="15min", tz="UTC"),
                    "value": 100,
                }
            )
        ),
    )

    tariff = """
- start: 2025-01-01T00:00:00+01:00
  consumption:
    constant_cost: 10.0
    variable_costs:
    - index: spot
      scalar: 1.05
- start: 2026-01-01T00:00:00+01:00
  consumption:
    constant_cost: 12.0
    variable_costs:
    - index: spot
      scalar: 1.10
"""
    contract = Contract(
        supplier=Tariff.model_validate(yaml.safe_load(tariff)),
        region="be_flanders",
        connection_type=ConnectionType.ELECTRICITY,
        customer_type=CustomerType.RESIDENTIAL,
        distributor_key="fluvius_antwerpen",
    )
    consumption = Meter(
        direction=PowerDirection.CONSUMPTION,
        type=MeterType.SINGLE_RATE,
        measurements=TimeseriesFrame(
            {
                "timestamp": pd.date_range("2026-05-01", "2026-06-01", freq="15min", tz="CET", inclusive="left"),
                "value": 0.0025,
            }
        ),
    )
    result = contract.apply(consumption)
    expected: dict[TariffCategory | Literal["total"], float] = {
        TariffCategory.SUPPLIER: 907.68,  # 4*24*31 * 0.0025 * (12.0 + 100*1.10) = 907.68 €
        TariffCategory.DISTRIBUTOR: 418.397301583,  # (0.01 MW capacity peak * 4116.971358333333) + (consumption * 50.5027) + (data_management/12) => 0.01 * 4116.971358333333 + 7.44 * 50.5027 + 17.85/12 = 418.397301583 €
        TariffCategory.FEES: 353.812176,  # consumption * (excise + energy_contribution) => 353.812176 €
    }
    expected[TariffCategory.TAXES] = (sum(expected.values())) * 0.06
    expected["total"] = sum(expected.values())

    assert result is not None
    # dataframe has one row starting on 2026-05-01 (monthly resolution)
    assert len(result) == 1
    assert result["timestamp"].iloc[0] == pd.Timestamp("2026-05-01T00:00:00+02:00")
    assert result[(TariffCategory.SUPPLIER, "total", "total")].iloc[0] == pytest.approx(
        expected[TariffCategory.SUPPLIER]
    )
    assert result[(TariffCategory.DISTRIBUTOR, "total", "total")].iloc[0] == pytest.approx(
        expected[TariffCategory.DISTRIBUTOR]
    )
    assert result[(TariffCategory.FEES, "total", "total")].iloc[0] == pytest.approx(expected[TariffCategory.FEES])
    assert result[(TariffCategory.TAXES, "total", "total")].iloc[0] == pytest.approx(expected[TariffCategory.TAXES])
    assert result[("total", "total", "total")].iloc[0] == pytest.approx(expected["total"])


# ---------------------------------------------------------------------------
# Contract — list[Tariff] merging
# ---------------------------------------------------------------------------


def test_contract_merges_list_of_tariffs_under_same_category() -> None:
    """When multiple tariffs are provided as a list under one category, their outputs are summed."""
    tariff_a = _tariff(energy_rate=100.0)
    tariff_b = _tariff(energy_rate=50.0)
    contract = Contract(
        supplier=_tariff(energy_rate=10.0),
        fees=[tariff_a, tariff_b],
    )
    # Full month of January
    timestamps = pd.date_range("2025-01-01", "2025-02-01", freq="15min", inclusive="left", tz=dt.UTC)
    consumption = _consumption(timestamps, value=1.0)

    result = contract.apply(Meter(measurements=TimeseriesFrame(consumption)))

    n = 2976
    # fees: n × (100 + 50)
    assert result[(TariffCategory.FEES, "total", "total")].iloc[0] == pytest.approx(n * 150.0)


# ---------------------------------------------------------------------------
# Default MeterType collapsing
# ---------------------------------------------------------------------------


def test_calculate_cost_with_mixed_offset_meter_data() -> None:
    """A billing period crossing the DST boundary where meter readings carry different
    UTC offsets (+01:00 before, +02:00 after spring-forward) must not raise."""

    contract = Contract(
        region="be_flanders",
        connection_type=ConnectionType.ELECTRICITY,
        customer_type=CustomerType.RESIDENTIAL,
        distributor_key="fluvius_antwerpen",
    )
    meter = Meter(
        measurements=TimeseriesFrame(
            pd.DataFrame(
                {
                    "timestamp": [
                        dt.datetime.fromisoformat("2024-03-31T01:45:00+01:00"),  # CET
                        dt.datetime.fromisoformat("2024-03-31T03:00:00+02:00"),  # CEST (after spring-forward)
                    ],
                    "value": [150.5, 75.3],
                }
            )
        ),
    )

    result = contract.apply(
        meter,
        start=dt.datetime.fromisoformat("2024-03-31T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2024-03-31T04:00:00+02:00"),
    )

    assert result is not None
    assert len(result) == 1  # one monthly billing row


# ---------------------------------------------------------------------------
# Contract.apply — monthly output for partial-month 15-min data
# ---------------------------------------------------------------------------


def test_contract_taxes_correct_for_range_spanning_month_boundary_with_monthly_output() -> None:
    """15-minute data spanning a month boundary (e.g. March 25 – April 9) with monthly
    output should produce two rows (one per month) each with a non-null tax."""
    from isodate import Duration

    contract = Contract(
        supplier=_tariff(energy_rate=100.0, start=dt.datetime(2024, 1, 1, tzinfo=dt.UTC)),
        taxes=Tax([TaxVersion(start=dt.datetime(2024, 1, 1, tzinfo=dt.UTC), default=0.10)]),
        timezone=dt.UTC,
    )

    start = dt.datetime(2024, 3, 1, tzinfo=dt.UTC)
    end = dt.datetime(2024, 5, 1, tzinfo=dt.UTC)
    timestamps = pd.date_range(start, end, freq="15min", inclusive="left", tz=dt.UTC)
    consumption = _consumption(timestamps, value=1.0)

    result = contract.apply(
        Meter(measurements=TimeseriesFrame(consumption)),
        start=start,
        end=end,
        output_resolution=Duration(months=1),
    )

    assert result is not None
    assert len(result) == 2
    assert result["timestamp"].iloc[0] == pd.Timestamp("2024-03-01", tz=dt.UTC)
    assert result["timestamp"].iloc[1] == pd.Timestamp("2024-04-01", tz=dt.UTC)

    # All tax values must be non-null and positive
    taxes = result[(TariffCategory.TAXES, "total", "total")]
    assert taxes.notna().all()
    assert (taxes > 0).all()


def test_contract_taxes_correct_for_complete_months_with_monthly_output() -> None:
    """When input data covers complete calendar months exactly, each month gets the
    correct tax (no off-by-one at period boundaries)."""
    from isodate import Duration

    contract = Contract(
        supplier=_tariff(energy_rate=100.0, start=dt.datetime(2024, 1, 1, tzinfo=dt.UTC)),
        taxes=Tax([TaxVersion(start=dt.datetime(2024, 1, 1, tzinfo=dt.UTC), default=0.10)]),
        timezone=dt.UTC,
    )

    start = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
    end = dt.datetime(2024, 4, 1, tzinfo=dt.UTC)  # three complete months
    timestamps = pd.date_range(start, end, freq="15min", inclusive="left", tz=dt.UTC)
    consumption = _consumption(timestamps, value=1.0)

    result = contract.apply(
        Meter(measurements=TimeseriesFrame(consumption)),
        start=start,
        end=end,
        output_resolution=Duration(months=1),
    )

    assert result is not None
    assert len(result) == 3
    assert result["timestamp"].iloc[0] == pd.Timestamp("2024-01-01", tz=dt.UTC)
    assert result["timestamp"].iloc[1] == pd.Timestamp("2024-02-01", tz=dt.UTC)
    assert result["timestamp"].iloc[2] == pd.Timestamp("2024-03-01", tz=dt.UTC)

    taxes = result[(TariffCategory.TAXES, "total", "total")]
    assert taxes.notna().all()

    # Jan=31d, Feb=29d (2024 is a leap year), Mar=31d
    days = [31, 29, 31]
    for i, d in enumerate(days):
        intervals = d * 24 * 4
        expected_supplier = intervals * 100.0
        expected_tax = expected_supplier * 0.10
        assert taxes.iloc[i] == pytest.approx(expected_tax)


def test_contract_taxes_correct_for_single_complete_month() -> None:
    """A billing window aligned exactly to a complete month produces a single row
    with the correct tax and no nulls."""
    from isodate import Duration

    contract = Contract(
        supplier=_tariff(energy_rate=100.0, start=dt.datetime(2024, 1, 1, tzinfo=dt.UTC)),
        taxes=Tax([TaxVersion(start=dt.datetime(2024, 1, 1, tzinfo=dt.UTC), default=0.21)]),
        timezone=dt.UTC,
    )

    start = dt.datetime(2024, 2, 1, tzinfo=dt.UTC)
    end = dt.datetime(2024, 3, 1, tzinfo=dt.UTC)
    timestamps = pd.date_range(start, end, freq="15min", inclusive="left", tz=dt.UTC)
    consumption = _consumption(timestamps, value=2.0)

    result = contract.apply(
        Meter(measurements=TimeseriesFrame(consumption)),
        start=start,
        end=end,
        output_resolution=Duration(months=1),
    )

    assert result is not None
    assert len(result) == 1
    assert result["timestamp"].iloc[0] == pd.Timestamp("2024-02-01", tz=dt.UTC)

    # Feb 2024 = 29 days (leap year)
    intervals = 29 * 24 * 4
    expected_supplier = intervals * 2.0 * 100.0
    tax = result[(TariffCategory.TAXES, "total", "total")].iloc[0]
    assert pd.notna(tax)
    assert tax == pytest.approx(expected_supplier * 0.21)


# ---------------------------------------------------------------------------
# Contract — reference key resolution
# ---------------------------------------------------------------------------


def test_contract_resolves_region_keys() -> None:
    """Contract with region/connection_type/customer_type/distributor_key resolves
    fees, taxes, distributor, and timezone from the registry."""
    contract = Contract(
        region="be_flanders",
        connection_type=ConnectionType.ELECTRICITY,
        customer_type=CustomerType.RESIDENTIAL,
        distributor_key="fluvius_antwerpen",
        supplier=_tariff(energy_rate=10.0),
    )

    assert contract.distributor is not None
    assert contract.fees is not None
    assert contract.taxes is not None
    from zoneinfo import ZoneInfo

    assert str(contract.timezone) == str(ZoneInfo("Europe/Brussels"))


def test_contract_region_keys_produce_valid_result() -> None:
    """A contract built from reference keys can be applied and produces a non-empty result."""
    contract = Contract(
        region="be_flanders",
        connection_type=ConnectionType.ELECTRICITY,
        customer_type=CustomerType.RESIDENTIAL,
        distributor_key="fluvius_antwerpen",
        supplier=_tariff(energy_rate=10.0),
    )
    # Full month of January 2025 in Brussels timezone
    timestamps = pd.date_range("2025-01-01", "2025-02-01", freq="15min", inclusive="left", tz="Europe/Brussels")
    consumption = _consumption(timestamps, value=1.0)

    result = contract.apply(Meter(measurements=TimeseriesFrame(consumption)))

    assert result is not None
    assert not result.empty
    assert (TariffCategory.SUPPLIER, "total", "total") in result.columns
    assert (TariffCategory.DISTRIBUTOR, "total", "total") in result.columns
    assert (TariffCategory.FEES, "total", "total") in result.columns
    assert (TariffCategory.TAXES, "total", "total") in result.columns


def test_contract_inline_overrides_reference_keys() -> None:
    """Inline fees/taxes/distributor take precedence over reference keys."""
    inline_distributor = _tariff(energy_rate=999.0)
    contract = Contract(
        region="be_flanders",
        connection_type=ConnectionType.ELECTRICITY,
        customer_type=CustomerType.RESIDENTIAL,
        distributor_key="fluvius_antwerpen",
        distributor=inline_distributor,  # inline override
        supplier=_tariff(energy_rate=10.0),
    )

    # The inline distributor should win over the registry one
    assert contract.distributor is inline_distributor


def test_contract_inline_fees_override_customer_type() -> None:
    """Inline fees take precedence over customer_type resolution."""
    inline_fees = _tariff(energy_rate=42.0)
    contract = Contract(
        region="be_flanders",
        connection_type=ConnectionType.ELECTRICITY,
        customer_type=CustomerType.RESIDENTIAL,
        fees=inline_fees,
    )

    assert contract.fees is inline_fees


def test_contract_inline_timezone_overrides_region() -> None:
    """An explicit timezone overrides the region default."""
    contract = Contract(
        region="be_flanders",
        connection_type=ConnectionType.ELECTRICITY,
        timezone=dt.UTC,
    )

    assert contract.timezone is dt.UTC


def test_contract_model_validate_rejects_non_dict_with_validation_error() -> None:
    """Passing a non-dict (e.g. a list) raises a clean ValidationError."""
    with pytest.raises(ValidationError):
        Contract.model_validate([{"start": "2025-01-01"}])


def test_contract_without_region_works_as_before() -> None:
    """Contracts without reference keys still work (pure inline)."""
    contract = Contract(
        supplier=_tariff(energy_rate=100.0),
        distributor=_tariff(energy_rate=50.0),
    )

    assert contract.region is None
    assert contract.distributor is not None
    assert contract.fees is None


# ---------------------------------------------------------------------------
# Contract — supplier registry resolution
# ---------------------------------------------------------------------------


def test_contract_resolves_supplier_key() -> None:
    """supplier_key + product_key resolves the supplier from the registry."""
    product = _tariff(energy_rate=42.0)
    Supplier.register("test_supplier", Supplier(products={"basic": product}))
    contract = Contract(supplier_key="test_supplier", product_key="basic")

    assert contract.supplier is product


def test_contract_inline_supplier_overrides_supplier_key() -> None:
    """An inline supplier takes precedence over supplier_key + product_key."""
    product = _tariff(energy_rate=42.0)
    Supplier.register("test_supplier", Supplier(products={"basic": product}))
    inline = _tariff(energy_rate=999.0)
    contract = Contract(
        supplier_key="test_supplier",
        product_key="basic",
        supplier=inline,
    )

    assert contract.supplier is inline


# ---------------------------------------------------------------------------
# ContractHistory
# ---------------------------------------------------------------------------


def test_contract_history_single_contract() -> None:
    """A history with one contract produces the same result as calling contract.apply directly."""
    history = ContractHistory(
        [
            Contract(
                start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
                supplier=_tariff(energy_rate=10.0),
            )
        ]
    )
    # Full month of January
    timestamps = pd.date_range("2025-01-01", "2025-02-01", freq="15min", inclusive="left", tz=dt.UTC)
    consumption = _consumption(timestamps, value=1.0)

    result = history.apply(Meter(measurements=TimeseriesFrame(consumption)))

    assert result is not None
    assert len(result) == 1
    assert (TariffCategory.SUPPLIER, "total", "total") in result.columns
    n = 2976
    assert result[("total", "total", "total")].iloc[0] == pytest.approx(n * 10.0)


def test_contract_history_two_contracts_sequential() -> None:
    """Two sequential contracts each produce rows for their respective period."""
    history = ContractHistory(
        [
            Contract(
                start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
                end=dt.datetime(2025, 2, 1, tzinfo=dt.UTC),
                supplier=_tariff(energy_rate=10.0),
            ),
            Contract(
                start=dt.datetime(2025, 2, 1, tzinfo=dt.UTC),
                supplier=_tariff(energy_rate=20.0),
            ),
        ]
    )
    # Full Jan+Feb
    timestamps = pd.date_range("2025-01-01", "2025-03-01", freq="15min", inclusive="left", tz=dt.UTC)
    consumption = _consumption(timestamps, value=1.0)

    result = history.apply(
        Meter(measurements=TimeseriesFrame(consumption)),
        start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
        end=dt.datetime(2025, 3, 1, tzinfo=dt.UTC),
    )

    assert result is not None
    assert len(result) == 2
    total = ("total", "total", "total")
    # Jan: 2976 × 1 × 10
    assert result[total].iloc[0] == pytest.approx(2976 * 10.0)
    # Feb: 2688 × 1 × 20
    assert result[total].iloc[1] == pytest.approx(2688 * 20.0)


def test_contract_history_gap_produces_no_rows() -> None:
    """A gap between contracts produces no rows for the gap period."""
    history = ContractHistory(
        [
            Contract(
                start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
                end=dt.datetime(2025, 2, 1, tzinfo=dt.UTC),
                supplier=_tariff(energy_rate=10.0),
            ),
            Contract(
                start=dt.datetime(2025, 4, 1, tzinfo=dt.UTC),
                supplier=_tariff(energy_rate=20.0),
            ),
        ]
    )
    ts = pd.date_range("2025-01-01", "2025-05-01", freq="15min", tz=dt.UTC, inclusive="left")
    consumption = pd.DataFrame({"timestamp": ts, "value": 1.0})

    result = history.apply(
        Meter(measurements=TimeseriesFrame(consumption)),
        start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
        end=dt.datetime(2025, 5, 1, tzinfo=dt.UTC),
    )

    assert result is not None
    # Jan row + Apr row = 2 rows (Feb, Mar gap → no rows)
    assert len(result) == 2
    assert result["timestamp"].iloc[0] == pd.Timestamp("2025-01-01", tz=dt.UTC)
    assert result["timestamp"].iloc[1] == pd.Timestamp("2025-04-01", tz=dt.UTC)


def test_contract_history_different_columns_zero_filled() -> None:
    """When contracts produce different columns, missing columns are zero-filled."""
    history = ContractHistory(
        [
            Contract(
                start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
                end=dt.datetime(2025, 2, 1, tzinfo=dt.UTC),
                supplier=_tariff(energy_rate=10.0),
            ),
            Contract(
                start=dt.datetime(2025, 2, 1, tzinfo=dt.UTC),
                supplier=_tariff(energy_rate=20.0),
                distributor=_tariff(energy_rate=5.0),
            ),
        ]
    )
    # Full Jan+Feb
    timestamps = pd.date_range("2025-01-01", "2025-03-01", freq="15min", inclusive="left", tz=dt.UTC)
    consumption = _consumption(timestamps, value=1.0)

    result = history.apply(
        Meter(measurements=TimeseriesFrame(consumption)),
        start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
        end=dt.datetime(2025, 3, 1, tzinfo=dt.UTC),
    )

    assert result is not None
    assert len(result) == 2
    # Jan row should have distributor columns filled with 0
    dist_col = (TariffCategory.DISTRIBUTOR, "total", "total")
    assert dist_col in result.columns
    assert result[dist_col].iloc[0] == pytest.approx(0.0)  # zero-filled
    # Feb: 2688 × 1 × 5
    assert result[dist_col].iloc[1] == pytest.approx(2688 * 5.0)


def test_contract_history_returns_none_when_no_contracts_overlap() -> None:
    """Querying a period with no active contracts returns None."""
    history = ContractHistory(
        [
            Contract(
                start=dt.datetime(2025, 6, 1, tzinfo=dt.UTC),
                supplier=_tariff(energy_rate=10.0),
            ),
        ]
    )
    ts = pd.date_range("2025-01-01", periods=4, freq="15min", tz=dt.UTC)
    consumption = _consumption(ts, value=1.0)

    result = history.apply(
        Meter(measurements=TimeseriesFrame(consumption)),
        start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
        end=dt.datetime(2025, 2, 1, tzinfo=dt.UTC),
    )

    assert result is None


def test_contract_history_from_yaml(tmp_path) -> None:
    """ContractHistory can be loaded from YAML via the inherited from_yaml."""
    yaml_content = """
- start: 2025-01-01T00:00:00+00:00
  end: 2025-06-01T00:00:00+00:00
  supplier:
  - start: 2025-01-01T00:00:00+00:00
    consumption:
      constant_cost: 10.0
- start: 2025-06-01T00:00:00+00:00
  supplier:
  - start: 2025-06-01T00:00:00+00:00
    consumption:
      constant_cost: 20.0
"""
    path = tmp_path / "history.yml"
    path.write_text(yaml_content, encoding="utf-8")

    history = ContractHistory.from_yaml(path)

    assert len(history.root) == 2
    assert history.root[0].end == dt.datetime(2025, 6, 1, tzinfo=dt.UTC)
    assert history.root[1].end is None


# ---------------------------------------------------------------------------
# Regression — weekly binning anchor misalignment (fees vs. supplier)
# ---------------------------------------------------------------------------


def test_weekly_output_no_nan_when_fees_and_supplier_span_different_version_segments() -> None:
    """Regression: weekly (P7D) output must not produce NaN rows when a supplier tariff
    has a version boundary mid-billing-window while the fees tariff is a single version
    spanning the whole window.

    Without the ``binning_anchor`` fix, each tariff component used its own segment
    start as the resample anchor.  The supplier's January-2026 segment started on
    2026-01-01 (Thursday), anchoring its weekly bins to Thursdays (Jan 1, Jan 8, …).
    The fees component had a single version whose segment start equalled billing_start
    (June 1, 2025 — a Sunday), anchoring its bins to Sundays (Jan 4, Jan 11, …).
    The two DataFrames were then merged via ``pd.concat(axis=1)``, producing
    alternating rows of NaN where timestamps didn't match.
    """
    # Supplier: version boundary at 2026-01-01 (Thursday in UTC), which is offset from
    # billing_start so the two produce weekly bins on different days of week.
    supplier = Tariff(
        [
            TariffVersion(
                start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
                consumption={"energy": IndexFormula(constant_cost=10.0)},
            ),
            TariffVersion(
                start=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
                consumption={"energy": IndexFormula(constant_cost=20.0)},
            ),
        ]
    )
    # Fees: single version (started long before billing window) with only a periodic
    # monthly fixed cost, which must go through redistribute_to_resolution (P1M → P7D).
    fees = Tariff(
        [
            TariffVersion(
                start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
                fixed={"fund": PeriodicFormula(period=isodate.parse_duration("P1M"), constant_cost=5.0)},
            )
        ]
    )
    contract = Contract(supplier=supplier, fees=fees, timezone=dt.UTC)

    billing_start = dt.datetime(2025, 6, 1, tzinfo=dt.UTC)
    # Daily data spanning the version boundary; cover enough to fill every weekly bin.
    # The billing end is inferred from the meter; we extend past the last full week to
    # ensure all weekly bins within the billing window have complete data.
    timestamps = pd.date_range(billing_start, dt.datetime(2026, 1, 18, tzinfo=dt.UTC), freq="D", inclusive="left")
    consumption = pd.DataFrame({"timestamp": timestamps, "value": 1.0})

    result = contract.apply(
        Meter(measurements=TimeseriesFrame(consumption)),
        start=billing_start,
        output_resolution=isodate.parse_duration("P7D"),
    )

    assert result is not None
    data_cols = [c for c in result.columns if c != "timestamp"]
    nan_rows = result[result[data_cols].isnull().any(axis=1)]
    assert len(nan_rows) == 0, (
        f"Found {len(nan_rows)} NaN row(s) — weekly bins misaligned between tariff components:\n"
        f"{nan_rows[['timestamp']].to_string()}"
    )


def test_contract_apply_with_injection_aligns_to_contract_timezone() -> None:
    """Contract.apply must align injection data to the contract timezone (contract.py line 112)."""
    contract = Contract(
        supplier=_tariff(energy_rate=10.0, injection_rate=5.0),
        timezone=dt.UTC,
    )
    # Full month of January
    timestamps = pd.date_range("2025-01-01", "2025-02-01", freq="15min", inclusive="left", tz=dt.UTC)
    consumption = _consumption(timestamps, value=2.0)
    injection = _consumption(timestamps, value=1.0)

    result = contract.apply(
        Meter(measurements=TimeseriesFrame(consumption)),
        injection=Meter(measurements=TimeseriesFrame(injection)),
    )

    assert result is not None
    assert (TariffCategory.SUPPLIER, CostGroup.INJECTION, "energy") in result.columns
    n = 2976
    # n intervals × 1 MWh × 5 €/MWh
    assert result[(TariffCategory.SUPPLIER, CostGroup.INJECTION, "energy")].iloc[0] == pytest.approx(n * 5.0)
