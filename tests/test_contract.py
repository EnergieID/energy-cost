from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import isodate
import pandas as pd
import pytest

from energy_cost.contract import Contract, ContractHistory
from energy_cost.data import ConnectionType, CustomerType, RegionalData
from energy_cost.formula import IndexFormula, PeriodicFormula
from energy_cost.meter import CostGroup, Meter, MeterType, PowerDirection, TariffCategory
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
    consumption: dict = {"all": {"energy": IndexFormula(constant_cost=energy_rate)}}
    injection: dict = (
        {"all": {"energy": IndexFormula(constant_cost=injection_rate)}} if injection_rate is not None else {}
    )
    periodic: dict = (
        {"fixed": PeriodicFormula(period=isodate.parse_duration("P1D"), constant_cost=daily_fixed)}
        if daily_fixed is not None
        else {}
    )
    return Tariff(
        versions=[
            TariffVersion(
                start=version_start,
                consumption=consumption,
                injection=injection,
                periodic=periodic,
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
    timestamps = pd.date_range("2025-01-01", periods=4, freq="15min")
    consumption = _consumption(timestamps, value=2.0)

    result = tariff.apply([Meter(data=consumption)])

    assert result is not None
    assert len(result) == 1
    # 4 intervals × 2 MWh × 10 €/MWh = 80 €
    assert result[(CostGroup.CONSUMPTION, "energy")].iloc[0] == pytest.approx(80.0)
    assert result[(CostGroup.TOTAL, "total")].iloc[0] == pytest.approx(80.0)


def test_apply_two_months_produces_two_rows() -> None:
    """One output row per calendar month when data spans two months."""
    tariff = _tariff(energy_rate=5.0)
    jan_ts = pd.date_range("2025-01-01", periods=2, freq="15min")
    feb_ts = pd.date_range("2025-02-01", periods=3, freq="15min")
    consumption = pd.DataFrame(
        {
            "timestamp": pd.concat([pd.Series(jan_ts), pd.Series(feb_ts)], ignore_index=True),
            "value": 1.0,
        }
    )

    result = tariff.apply([Meter(data=consumption)])

    assert result is not None
    assert len(result) == 2
    # Jan: 2 × 1 × 5 = 10 €
    assert result[(CostGroup.CONSUMPTION, "energy")].iloc[0] == pytest.approx(10.0)
    # Feb: 3 × 1 × 5 = 15 €
    assert result[(CostGroup.CONSUMPTION, "energy")].iloc[1] == pytest.approx(15.0)


def test_apply_explicit_start_end_restricts_billing_period() -> None:
    """Data outside [start, end) does not contribute to cost."""
    tariff = _tariff(energy_rate=10.0)
    timestamps = pd.date_range("2025-01-01", periods=4, freq="15min")
    consumption = _consumption(timestamps, value=1.0)

    # Only include the first two intervals
    result = tariff.apply(
        [Meter(data=consumption)],
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 0, 30),
        resolution=dt.timedelta(minutes=30),
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
        "    all:\n"
        "      energy:\n"
        "        constant_cost: 0.0\n"
        "  capacity:\n"
        "    measurement_period: PT15M\n"
        "    billing_period: P1M\n"
        "    formula:\n"
        "      constant_cost: 1.0\n",
        encoding="utf-8",
    )
    tariff = Tariff.from_yaml(capacity_tariff_yaml)

    # Provide data for two months; only bill the second month.
    jan_ts = pd.date_range("2025-01-01", periods=4, freq="15min")
    feb_ts = pd.date_range("2025-02-01", periods=4, freq="15min")
    all_ts = pd.concat([pd.Series(jan_ts), pd.Series(feb_ts)], ignore_index=True)
    consumption = pd.DataFrame({"timestamp": all_ts, "value": 5.0})

    result = tariff.apply(
        [Meter(data=consumption)],
        start=dt.datetime(2025, 2, 1, 0, 0),
        end=dt.datetime(2025, 3, 1, 0, 0),
    )

    # Only February should appear in the output.
    assert result is not None
    assert len(result) == 1
    assert result["timestamp"].iloc[0] == pd.Timestamp("2025-02-01", tz=dt.UTC)
    # Capacity column should be present with a non-NaN value.
    assert (CostGroup.CAPACITY, "total") in result.columns
    assert pd.notna(result[(CostGroup.CAPACITY, "total")].iloc[0])


def test_apply_custom_output_resolution() -> None:
    """Costs can be aggregated to a daily resolution."""
    tariff = _tariff(energy_rate=10.0)
    timestamps = pd.date_range("2025-01-01", periods=8, freq="15min")
    consumption = _consumption(timestamps, value=1.0)

    result = tariff.apply([Meter(data=consumption)], resolution=dt.timedelta(days=1))

    assert result is not None
    assert len(result) == 1
    # 8 intervals × 1 MWh × 10 €/MWh = 80 €
    assert result[(CostGroup.CONSUMPTION, "energy")].iloc[0] == pytest.approx(80.0)


def test_apply_omits_consumption_frame_when_tariff_has_no_consumption_formulas(tmp_path) -> None:
    """A capacity-only tariff produces no consumption column in the output."""
    cap_yaml = tmp_path / "cap_only.yml"
    cap_yaml.write_text(
        "- start: 2025-01-01T00:00:00\n"
        "  capacity:\n"
        "    measurement_period: PT15M\n"
        "    billing_period: P1M\n"
        "    formula:\n"
        "      constant_cost: 1.0\n",
        encoding="utf-8",
    )
    tariff = Tariff.from_yaml(cap_yaml)
    timestamps = pd.date_range("2025-01-01", periods=4, freq="15min")
    result = tariff.apply([Meter(data=_consumption(timestamps, value=5.0))])

    assert result is not None
    data_cols = [c for c in result.columns if c != "timestamp"]
    assert not any(c[0] == CostGroup.CONSUMPTION for c in data_cols)
    assert (CostGroup.CAPACITY, "total") in result.columns
    assert (CostGroup.TOTAL, "total") in result.columns


def test_apply_column_structure_is_two_level_multiindex() -> None:
    """The output columns form a two-level MultiIndex by default (excluding the timestamp column)."""
    tariff = _tariff(energy_rate=10.0)
    timestamps = pd.date_range("2025-01-01", periods=2, freq="15min")
    result = tariff.apply([Meter(data=_consumption(timestamps))])

    assert result is not None
    data_cols = [c for c in result.columns if c != "timestamp"]
    assert all(isinstance(c, tuple) and len(c) == 2 for c in data_cols)
    assert (CostGroup.CONSUMPTION, "energy") in result.columns
    assert (CostGroup.TOTAL, "total") in result.columns


# ---------------------------------------------------------------------------
# Tariff.apply – injection
# ---------------------------------------------------------------------------


def test_apply_with_injection_adds_injection_columns() -> None:
    """When injection data is supplied the result includes injection cost columns."""
    tariff = _tariff(energy_rate=10.0, injection_rate=5.0)
    timestamps = pd.date_range("2025-01-01", periods=2, freq="15min")
    consumption = _consumption(timestamps, value=2.0)
    injection = _consumption(timestamps, value=1.0)

    result = tariff.apply(
        [
            Meter(data=consumption),
            Meter(data=injection, direction=PowerDirection.INJECTION),
        ]
    )

    assert result is not None
    assert (CostGroup.INJECTION, "energy") in result.columns
    # Consumption: 2 × 2 × 10 = 40; Injection: 2 × 1 × 5 = 10; Total = 50
    assert result[(CostGroup.CONSUMPTION, "energy")].iloc[0] == pytest.approx(40.0)
    assert result[(CostGroup.INJECTION, "energy")].iloc[0] == pytest.approx(10.0)
    assert result[(CostGroup.CONSUMPTION, "total")].iloc[0] == pytest.approx(40.0)
    assert result[(CostGroup.INJECTION, "total")].iloc[0] == pytest.approx(10.0)
    assert result[(CostGroup.TOTAL, "total")].iloc[0] == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Tariff.apply – fixed / periodic costs
# ---------------------------------------------------------------------------


def test_apply_includes_fixed_costs_prorated_per_output_period() -> None:
    """Periodic costs appear under the ``fixed`` group for the full snapped billing period."""
    tariff = _tariff(energy_rate=0.0, daily_fixed=24.0)
    # 2 hours of data inside January 2025; billing is snapped to the full month
    timestamps = pd.date_range("2025-01-01", periods=8, freq="15min")
    consumption = _consumption(timestamps)

    result = tariff.apply([Meter(data=consumption)])

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
    timestamps = pd.date_range("2025-01-01", periods=2, freq="15min")
    consumption = _consumption(timestamps, value=1.0)

    result = contract.apply([Meter(data=consumption)])

    assert (TariffCategory.SUPPLIER, CostGroup.CONSUMPTION, "energy") in result.columns
    assert (TariffCategory.DISTRIBUTOR, CostGroup.CONSUMPTION, "energy") in result.columns


def test_contract_taxes_applied_to_all_tariffs() -> None:
    """Taxes are computed on the sum of ALL tariff totals, including fees."""
    contract = Contract(
        supplier=_tariff(energy_rate=100.0),
        distributor=_tariff(energy_rate=100.0),
        fees=_tariff(energy_rate=50.0),
        taxes=Tax(versions=[TaxVersion(start=dt.datetime(2025, 1, 1), default=0.10)]),
    )
    timestamps = pd.date_range("2025-01-01", periods=2, freq="15min")
    # 2 intervals × 1 MWh each
    consumption = _consumption(timestamps, value=1.0)

    result = contract.apply([Meter(data=consumption)])

    supplier_total = result[(TariffCategory.SUPPLIER, CostGroup.TOTAL, "total")].iloc[0]
    distributor_total = result[(TariffCategory.DISTRIBUTOR, CostGroup.TOTAL, "total")].iloc[0]
    fees_total = result[(TariffCategory.FEES, CostGroup.TOTAL, "total")].iloc[0]
    taxes = result[(TariffCategory.TAXES, CostGroup.TOTAL, "total")].iloc[0]
    total_cost = result[(TariffCategory.TOTAL, CostGroup.TOTAL, "total")].iloc[0]

    # supplier: 2 × 100 = 200; distributor: 2 × 100 = 200; fees: 2 × 50 = 100
    assert supplier_total == pytest.approx(200.0)
    assert distributor_total == pytest.approx(200.0)
    assert fees_total == pytest.approx(100.0)
    # taxes = (200 + 200 + 100) × 0.10 = 50 (fees now included in tax base)
    assert taxes == pytest.approx(50.0)
    # total = 200 + 200 + 100 + 50 = 550
    assert total_cost == pytest.approx(550.0)


def test_contract_list_of_taxes() -> None:
    """Multiple Tax specs are summed independently."""
    vat = Tax(versions=[TaxVersion(start=dt.datetime(2025, 1, 1), default=0.10)])
    levy = Tax(versions=[TaxVersion(start=dt.datetime(2025, 1, 1), default=0.05)])
    contract = Contract(
        supplier=_tariff(energy_rate=100.0),
        taxes=[vat, levy],
    )
    timestamps = pd.date_range("2025-01-01", periods=2, freq="15min")
    consumption = _consumption(timestamps, value=1.0)

    result = contract.apply([Meter(data=consumption)])

    # supplier: 2 × 100 = 200; taxes = 200*(0.10 + 0.05) = 30
    taxes = result[(TariffCategory.TAXES, CostGroup.TOTAL, "total")].iloc[0]
    assert taxes == pytest.approx(30.0)


def test_contract_no_fees_omits_fees_columns() -> None:
    """When no fees tariff is present the result has no fees columns."""
    contract = Contract(
        supplier=_tariff(energy_rate=100.0),
        distributor=_tariff(energy_rate=50.0),
    )
    timestamps = pd.date_range("2025-01-01", periods=2, freq="15min")
    result = contract.apply([Meter(data=_consumption(timestamps))])

    assert not any(isinstance(c, tuple) and c[0] == TariffCategory.FEES for c in result.columns)


def test_contract_column_structure_is_three_level_multiindex() -> None:
    """Data columns form a three-level MultiIndex by default (collapsed); timestamp is a plain column."""
    contract = Contract(
        supplier=_tariff(energy_rate=10.0),
        distributor=_tariff(energy_rate=5.0),
        taxes=Tax(versions=[TaxVersion(start=dt.datetime(2025, 1, 1), default=0.21)]),
    )
    timestamps = pd.date_range("2025-01-01", periods=2, freq="15min")
    result = contract.apply([Meter(data=_consumption(timestamps))])

    data_cols = [c for c in result.columns if c != "timestamp"]
    assert all(isinstance(c, tuple) and len(c) == 3 for c in data_cols)


def test_contract_total_cost_equals_manual_sum() -> None:
    """total_cost == supplier_total + distributor_total + taxes (no fees case)."""
    contract = Contract(
        supplier=_tariff(energy_rate=100.0),
        distributor=_tariff(energy_rate=50.0),
        taxes=Tax(versions=[TaxVersion(start=dt.datetime(2025, 1, 1), default=0.21)]),
    )
    timestamps = pd.date_range("2025-01-01", periods=4, freq="15min")
    consumption = _consumption(timestamps, value=2.0)

    result = contract.apply([Meter(data=consumption)])

    p = result[(TariffCategory.SUPPLIER, CostGroup.TOTAL, "total")].iloc[0]
    d = result[(TariffCategory.DISTRIBUTOR, CostGroup.TOTAL, "total")].iloc[0]
    taxes = result[(TariffCategory.TAXES, CostGroup.TOTAL, "total")].iloc[0]
    total = result[(TariffCategory.TOTAL, CostGroup.TOTAL, "total")].iloc[0]

    # supplier: 4 × 2 × 100 = 800; distributor: 4 × 2 × 50 = 400
    assert p == pytest.approx(800.0)
    assert d == pytest.approx(400.0)
    assert taxes == pytest.approx((800.0 + 400.0) * 0.21)
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
    consumption: dict = {"all": {"energy": IndexFormula(constant_cost=energy_rate)}}
    periodic: dict = (
        {"fixed": PeriodicFormula(period=isodate.parse_duration("P1D"), constant_cost=daily_fixed)}
        if daily_fixed is not None
        else {}
    )
    version = TariffVersion(start=start, consumption=consumption, periodic=periodic)
    tariff = Tariff(versions=[version])
    return Contract(supplier=tariff, distributor=tariff, timezone=_CET)


def test_calculate_cost_output_timestamps_match_input_timezone() -> None:
    """Timestamps in the result must retain the input data's timezone, not shift to UTC."""
    contract = _tz_contract(energy_rate=10.0)
    timestamps = pd.date_range("2025-01-01T00:00:00+01:00", periods=4, freq="15min")
    consumption = pd.DataFrame({"timestamp": timestamps, "value": 1.0})

    result = contract.apply([Meter(data=consumption)])

    assert result["timestamp"].dt.tz is not None
    # Midnight CET must appear as 2025-01-01 00:00 +01:00, not 2024-12-31 23:00 UTC
    assert result["timestamp"].iloc[0] == pd.Timestamp("2025-01-01T00:00:00+01:00")


def test_calculate_cost_zoneinfo_start_end_work_correctly() -> None:
    """zoneinfo-aware start/end must be accepted without any loss of precision."""
    contract = _tz_contract(energy_rate=5.0)
    timestamps = pd.date_range("2025-01-01T00:00:00+01:00", periods=8, freq="15min")
    consumption = pd.DataFrame({"timestamp": timestamps, "value": 2.0})

    z = ZoneInfo("Europe/Brussels")
    start = dt.datetime(2025, 1, 1, 0, 0, tzinfo=z)
    end = dt.datetime(2025, 2, 1, 0, 0, tzinfo=z)

    result = contract.apply([Meter(data=consumption)], start=start, end=end)

    assert result is not None
    # 8 intervals × 2 MWh × 5 €/MWh = 80 €
    assert result[(TariffCategory.SUPPLIER, CostGroup.TOTAL, "total")].iloc[0] == pytest.approx(80.0)
    assert result["timestamp"].iloc[0] == pd.Timestamp("2025-01-01T00:00:00+01:00")


# ---------------------------------------------------------------------------
# Tariff.apply / Contract.apply — TOU meters & multiple meters
# ---------------------------------------------------------------------------


def test_apply_tou_peak_meter_billed_under_tou_column() -> None:
    """A TOU_PEAK meter uses the tou_peak formula; its column is prefixed with the meter type."""
    # a tariff that has both single_rate and tou_peak formulas.
    tou_tariff = Tariff(
        versions=[
            TariffVersion(
                start=dt.datetime(2025, 1, 1, 0, 0),
                consumption={
                    "single_rate": {"energy": IndexFormula(constant_cost=10.0)},
                    "tou_peak": {"energy": IndexFormula(constant_cost=30.0)},
                },
            )
        ]
    )
    timestamps = pd.date_range("2025-01-01", periods=4, freq="15min")
    data = _consumption(timestamps, value=1.0)

    result = tou_tariff.apply([Meter(data=data, type=MeterType.TOU_PEAK)], include_meter_type=True)

    assert result is not None
    # The output column must use MeterType.TOU_PEAK, not MeterType.SINGLE_RATE.
    assert (CostGroup.CONSUMPTION, MeterType.TOU_PEAK, "energy") in result.columns
    assert (CostGroup.CONSUMPTION, MeterType.SINGLE_RATE, "energy") not in result.columns
    # 4 intervals × 1 MWh × 30 €/MWh = 120 €
    assert result[(CostGroup.CONSUMPTION, MeterType.TOU_PEAK, "energy")].iloc[0] == pytest.approx(120.0)
    assert result[(CostGroup.TOTAL, MeterType.ALL, "total")].iloc[0] == pytest.approx(120.0)


def test_apply_multiple_consumption_meters_produce_separate_columns() -> None:
    """A single_rate and a tou_peak consumption meter each get their own column group."""
    tou_tariff = Tariff(
        versions=[
            TariffVersion(
                start=dt.datetime(2025, 1, 1, 0, 0),
                consumption={
                    "single_rate": {"energy": IndexFormula(constant_cost=10.0)},
                    "tou_peak": {"energy": IndexFormula(constant_cost=20.0)},
                },
            )
        ]
    )
    timestamps = pd.date_range("2025-01-01", periods=4, freq="15min")
    single_data = _consumption(timestamps, value=2.0)
    tou_data = _consumption(timestamps, value=3.0)

    result = tou_tariff.apply(
        [
            Meter(data=single_data, type=MeterType.SINGLE_RATE),
            Meter(data=tou_data, type=MeterType.TOU_PEAK),
        ],
        include_meter_type=True,
    )

    assert result is not None
    assert (CostGroup.CONSUMPTION, MeterType.SINGLE_RATE, "energy") in result.columns
    assert (CostGroup.CONSUMPTION, MeterType.TOU_PEAK, "energy") in result.columns
    # single_rate: 4 × 2 × 10 = 80 €; tou_peak: 4 × 3 × 20 = 240 €
    assert result[(CostGroup.CONSUMPTION, MeterType.SINGLE_RATE, "energy")].iloc[0] == pytest.approx(80.0)
    assert result[(CostGroup.CONSUMPTION, MeterType.TOU_PEAK, "energy")].iloc[0] == pytest.approx(240.0)
    assert result[(CostGroup.TOTAL, MeterType.ALL, "total")].iloc[0] == pytest.approx(320.0)


def test_contract_with_tou_meter_routes_cost_correctly() -> None:
    """contract.apply routes a TOU_PEAK meter through the correct formula."""
    tou_tariff = Tariff(
        versions=[
            TariffVersion(
                start=dt.datetime(2025, 1, 1, 0, 0),
                consumption={
                    "tou_peak": {"energy": IndexFormula(constant_cost=50.0)},
                },
            )
        ]
    )
    contract = Contract(supplier=tou_tariff, distributor=tou_tariff)
    timestamps = pd.date_range("2025-01-01", periods=4, freq="15min")
    data = _consumption(timestamps, value=1.0)

    result = contract.apply([Meter(data=data, type=MeterType.TOU_PEAK)])

    assert result is not None
    # supplier + distributor each: 4 × 1 × 50 = 200 €
    assert result[(TariffCategory.SUPPLIER, CostGroup.CONSUMPTION, "energy")].iloc[0] == pytest.approx(200.0)
    assert result[(TariffCategory.DISTRIBUTOR, CostGroup.CONSUMPTION, "energy")].iloc[0] == pytest.approx(200.0)
    assert result[(TariffCategory.TOTAL, CostGroup.TOTAL, "total")].iloc[0] == pytest.approx(400.0)


def test_contract_with_injection_and_tou_meters() -> None:
    """Contract correctly handles a mix of injection and TOU consumption meters."""
    supplier_tariff = Tariff(
        versions=[
            TariffVersion(
                start=dt.datetime(2025, 1, 1, 0, 0),
                consumption={
                    "tou_offpeak": {"energy": IndexFormula(constant_cost=8.0)},
                },
                injection={
                    "all": {"energy": IndexFormula(constant_cost=4.0)},
                },
            )
        ]
    )
    distributor_tariff = Tariff(
        versions=[
            TariffVersion(
                start=dt.datetime(2025, 1, 1, 0, 0),
                consumption={
                    "tou_offpeak": {"energy": IndexFormula(constant_cost=2.0)},
                },
            )
        ]
    )
    contract = Contract(supplier=supplier_tariff, distributor=distributor_tariff)
    timestamps = pd.date_range("2025-01-01", periods=4, freq="15min")
    cons_data = _consumption(timestamps, value=2.0)
    inj_data = _consumption(timestamps, value=1.0)

    result = contract.apply(
        [
            Meter(data=cons_data, type=MeterType.TOU_OFFPEAK),
            Meter(data=inj_data, direction=PowerDirection.INJECTION),
        ],
    )

    assert result is not None
    # supplier consumption_tou_offpeak: 4 × 2 × 8 = 64 €; injection: 4 × 1 × 4 = 16 €
    assert result[(TariffCategory.SUPPLIER, CostGroup.CONSUMPTION, "energy")].iloc[0] == pytest.approx(64.0)
    assert result[(TariffCategory.SUPPLIER, CostGroup.INJECTION, "energy")].iloc[0] == pytest.approx(16.0)
    # distributor consumption_tou_offpeak: 4 × 2 × 2 = 16 €
    assert result[(TariffCategory.DISTRIBUTOR, CostGroup.CONSUMPTION, "energy")].iloc[0] == pytest.approx(16.0)
    # total: supplier(64+16) + distributor(16) = 96 €
    assert result[(TariffCategory.TOTAL, CostGroup.TOTAL, "total")].iloc[0] == pytest.approx(96.0)


# ---------------------------------------------------------------------------
# meter.as_single_meter – error path
# ---------------------------------------------------------------------------


def test_as_single_meter_raises_when_no_matching_direction() -> None:
    """as_single_meter raises ValueError when no meter matches the requested direction."""
    from energy_cost.meter import as_single_meter

    meters = [
        Meter(
            data=pd.DataFrame({"timestamp": pd.Series([], dtype="datetime64[ns]"), "value": pd.Series([], dtype=float)})
        )
    ]
    with pytest.raises(ValueError, match="No meters found for direction"):
        as_single_meter(meters, PowerDirection.INJECTION)


# ---------------------------------------------------------------------------
# Tariff._apply_capacity_costs – empty-after-filter path
# ---------------------------------------------------------------------------


def test_apply_capacity_costs_returns_none_when_filtered_slice_is_empty(tmp_path) -> None:
    """_apply_capacity_costs returns None when the capacity row timestamps lie entirely
    outside [billing_start, billing_end).

    With an annual billing period the capacity component produces a single row
    timestamped at the start of the year (2025-01-01).  When the billing window
    is restricted to February the Jan row is excluded → filtered is empty → None.
    """
    cap_yaml = tmp_path / "cap_annual.yml"
    cap_yaml.write_text(
        "- start: 2025-01-01T00:00:00\n"
        "  capacity:\n"
        "    measurement_period: PT15M\n"
        "    billing_period: P1Y\n"
        "    formula:\n"
        "      constant_cost: 1.0\n",
        encoding="utf-8",
    )
    tariff = Tariff.from_yaml(cap_yaml)

    # Data spans Jan + Feb so the Feb slice is non-empty (no energy formula, so the
    # direction-cost frame is None, but detect_resolution_and_range won't error).
    jan_ts = pd.date_range("2025-01-01", periods=4, freq="15min")
    feb_ts = pd.date_range("2025-02-01", periods=4, freq="15min")
    all_ts = pd.concat([pd.Series(jan_ts), pd.Series(feb_ts)], ignore_index=True)
    consumption = pd.DataFrame({"timestamp": all_ts, "value": 5.0})

    # Billing window is February only → annual capacity row at 2025-01-01 is outside
    # [2025-02-01, 2025-03-01) → _apply_capacity_costs returns None.
    result = tariff.apply(
        [Meter(data=consumption)],
        start=dt.datetime(2025, 2, 1, 0, 0),
        end=dt.datetime(2025, 3, 1, 0, 0),
    )

    # No energy formula and no in-window capacity → nothing to report.
    assert result is None


def test_avoid_regression_on_real_world_data() -> None:
    import yaml

    data = RegionalData.get("be_flanders", ConnectionType.ELECTRICITY)
    from energy_cost.index import DataFrameIndex, Index

    Index.register(
        "Belpex15min",
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
    - index: Belpex15min
      scalar: 1.05
- start: 2026-01-01T00:00:00+01:00
  consumption:
    constant_cost: 12.0
    variable_costs:
    - index: Belpex15min
      scalar: 1.10
"""
    contract = Contract(
        supplier=Tariff.model_validate({"versions": yaml.safe_load(tariff)}),
        distributor=data.distributors["fluvius_antwerpen"],
        fees=data.fees[CustomerType.RESIDENTIAL],
        taxes=data.taxes,
    )
    meters = [
        Meter(
            direction=PowerDirection.CONSUMPTION,
            type=MeterType.SINGLE_RATE,
            data=pd.DataFrame(
                {
                    "timestamp": pd.date_range("2026-04-09", periods=4, freq="15min", tz="UTC"),
                    "value": [0.0025, 0.0025, 0.0025, 0.0025],
                }
            ),
        )
    ]
    result = contract.apply(meters)
    expected = {
        TariffCategory.SUPPLIER: 1.22,  # 4 × 0.0025 × (12.0 + 100*1.10) = 1.22 €
        TariffCategory.DISTRIBUTOR: 43.162240583,  # (capacity 0.01 * 4116.9713583) + (consumption * 50.5027) + (data_management/12) => 41.1697 + 0.01 * 50.5027 + 17.85/12 = 43.162227 €
        TariffCategory.FEES: 0.475554,  # consumption * (excise + energy_contribution) => 0.475554 €
    }
    expected[TariffCategory.TAXES] = (sum(expected.values())) * 0.06
    expected[TariffCategory.TOTAL] = sum(expected.values())

    assert result is not None
    # dataframe has one row starting on 2026-04-01 (monthly resolution)
    assert len(result) == 1
    assert result["timestamp"].iloc[0] == pd.Timestamp("2026-04-01T00:00:00+00:00")
    assert result[(TariffCategory.SUPPLIER, CostGroup.TOTAL, "total")].iloc[0] == pytest.approx(
        expected[TariffCategory.SUPPLIER]
    )
    assert result[(TariffCategory.DISTRIBUTOR, CostGroup.TOTAL, "total")].iloc[0] == pytest.approx(
        expected[TariffCategory.DISTRIBUTOR]
    )
    assert result[(TariffCategory.FEES, CostGroup.TOTAL, "total")].iloc[0] == pytest.approx(
        expected[TariffCategory.FEES]
    )
    assert result[(TariffCategory.TAXES, CostGroup.TOTAL, "total")].iloc[0] == pytest.approx(
        expected[TariffCategory.TAXES]
    )
    assert result[(TariffCategory.TOTAL, CostGroup.TOTAL, "total")].iloc[0] == pytest.approx(
        expected[TariffCategory.TOTAL]
    )


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
    timestamps = pd.date_range("2025-01-01", periods=2, freq="15min")
    consumption = _consumption(timestamps, value=1.0)

    result = contract.apply([Meter(data=consumption)])

    # fees: (2 × 100) + (2 × 50) = 300
    assert result[(TariffCategory.FEES, CostGroup.TOTAL, "total")].iloc[0] == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# Default MeterType collapsing
# ---------------------------------------------------------------------------


def test_contract_collapses_meter_types_by_default() -> None:
    """contract.apply collapses MeterType by default (3-level output)."""
    tou_tariff = Tariff(
        versions=[
            TariffVersion(
                start=dt.datetime(2025, 1, 1, 0, 0),
                consumption={
                    "single_rate": {"energy": IndexFormula(constant_cost=10.0)},
                    "tou_peak": {"energy": IndexFormula(constant_cost=20.0)},
                },
            )
        ]
    )
    contract = Contract(supplier=tou_tariff)
    timestamps = pd.date_range("2025-01-01", periods=4, freq="15min")
    result = contract.apply(
        [
            Meter(data=_consumption(timestamps, value=1.0), type=MeterType.SINGLE_RATE),
            Meter(data=_consumption(timestamps, value=1.0), type=MeterType.TOU_PEAK),
        ]
    )

    data_cols = [c for c in result.columns if c != "timestamp"]
    assert all(isinstance(c, tuple) and len(c) == 3 for c in data_cols)
    # single_rate energy (4*10=40) + tou_peak energy (4*20=80) = 120
    assert result[(TariffCategory.SUPPLIER, CostGroup.CONSUMPTION, "energy")].iloc[0] == pytest.approx(120.0)
    assert result[(TariffCategory.TOTAL, CostGroup.TOTAL, "total")].iloc[0] == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# Contract.apply — mixed-offset timestamps (DST boundary)
# ---------------------------------------------------------------------------


def test_calculate_cost_with_mixed_offset_meter_data() -> None:
    """A billing period crossing the DST boundary where meter readings carry different
    UTC offsets (+01:00 before, +02:00 after spring-forward) must not raise."""

    from isodate import Duration

    data = RegionalData.get("be_flanders", ConnectionType.ELECTRICITY)

    contract = Contract(
        distributor=data.distributors["fluvius_antwerpen"],
        fees=data.fees[CustomerType.RESIDENTIAL],
        taxes=data.taxes,
        timezone=data.timezone,
    )
    meter = Meter(
        direction=PowerDirection.CONSUMPTION,
        type=MeterType.SINGLE_RATE,
        data=pd.DataFrame(
            {
                "timestamp": [
                    dt.datetime.fromisoformat("2024-03-31T01:45:00+01:00"),  # CET
                    dt.datetime.fromisoformat("2024-03-31T03:00:00+02:00"),  # CEST (after spring-forward)
                ],
                "value": [150.5, 75.3],
            }
        ),
    )

    result = contract.apply(
        meters=[meter],
        start=dt.datetime.fromisoformat("2024-03-31T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2024-03-31T04:00:00+02:00"),
        resolution=Duration(months=1),
    )

    assert result is not None
    assert len(result) == 1  # one monthly billing row


# ---------------------------------------------------------------------------
# Contract.apply — monthly output for partial-month 15-min data
# ---------------------------------------------------------------------------


def test_contract_taxes_not_null_for_partial_month_input_with_monthly_output() -> None:
    """Regression: tax must not be null when 15-minute input data covers only part of a
    month (e.g. March 15-17) but the output resolution is monthly.

    The tariff aggregates the data into a single monthly row with timestamp
    2024-03-01.  That row is then passed to Tax.apply with start=2024-03-15 and
    end=2024-03-17.  The old exact-match filter (timestamp >= start) incorrectly
    excluded the row because 2024-03-01 < 2024-03-15, yielding a null tax.
    The fix extends the filter to an overlap check: the row's period [2024-03-01,
    2024-04-01) clearly overlaps with [2024-03-15, 2024-03-17).
    """
    from isodate import Duration

    contract = Contract(
        supplier=_tariff(energy_rate=100.0, start=dt.datetime(2024, 1, 1, tzinfo=dt.UTC)),
        taxes=Tax(versions=[TaxVersion(start=dt.datetime(2024, 1, 1, tzinfo=dt.UTC), default=0.10)]),
        timezone=dt.UTC,
    )

    start = dt.datetime(2024, 3, 15, tzinfo=dt.UTC)
    end = dt.datetime(2024, 3, 17, tzinfo=dt.UTC)
    # 2 full days × 4 intervals/h × 24 h = 192 fifteen-minute intervals
    timestamps = pd.date_range(start, end, freq="15min", inclusive="left")
    consumption = _consumption(timestamps, value=1.0)

    result = contract.apply(
        [Meter(data=consumption)],
        start=start,
        end=end,
        resolution=Duration(months=1),
    )

    assert result is not None
    assert len(result) == 1
    # Monthly output row is timestamped at the start of March
    assert result["timestamp"].iloc[0] == pd.Timestamp("2024-03-01", tz=dt.UTC)
    # Tax must not be null
    tax = result[(TariffCategory.TAXES, CostGroup.TOTAL, "total")].iloc[0]
    assert pd.notna(tax)
    # supplier: 192 intervals × 1.0 MWh × 100 €/MWh = 19 200 €; tax = 10 % = 1 920 €
    assert tax == pytest.approx(1920.0)


def test_contract_taxes_correct_for_range_spanning_month_boundary_with_monthly_output() -> None:
    """15-minute data spanning a month boundary (e.g. March 25 – April 9) with monthly
    output should produce two rows (one per month) each with a non-null tax."""
    from isodate import Duration

    contract = Contract(
        supplier=_tariff(energy_rate=100.0, start=dt.datetime(2024, 1, 1, tzinfo=dt.UTC)),
        taxes=Tax(versions=[TaxVersion(start=dt.datetime(2024, 1, 1, tzinfo=dt.UTC), default=0.10)]),
        timezone=dt.UTC,
    )

    start = dt.datetime(2024, 3, 25, tzinfo=dt.UTC)
    end = dt.datetime(2024, 4, 9, tzinfo=dt.UTC)
    timestamps = pd.date_range(start, end, freq="15min", inclusive="left")
    consumption = _consumption(timestamps, value=1.0)

    result = contract.apply(
        [Meter(data=consumption)],
        start=start,
        end=end,
        resolution=Duration(months=1),
    )

    assert result is not None
    assert len(result) == 2
    assert result["timestamp"].iloc[0] == pd.Timestamp("2024-03-01", tz=dt.UTC)
    assert result["timestamp"].iloc[1] == pd.Timestamp("2024-04-01", tz=dt.UTC)

    # All tax values must be non-null and positive
    taxes = result[(TariffCategory.TAXES, CostGroup.TOTAL, "total")]
    assert taxes.notna().all()
    assert (taxes > 0).all()

    # March slice: March 25 00:00 – April 1 00:00 = 7 days = 672 intervals
    march_supplier = result[(TariffCategory.SUPPLIER, CostGroup.TOTAL, "total")].iloc[0]
    assert march_supplier == pytest.approx(672 * 1.0 * 100.0)
    assert taxes.iloc[0] == pytest.approx(march_supplier * 0.10)

    # April slice: April 1 00:00 – April 9 00:00 = 8 days = 768 intervals
    april_supplier = result[(TariffCategory.SUPPLIER, CostGroup.TOTAL, "total")].iloc[1]
    assert april_supplier == pytest.approx(768 * 1.0 * 100.0)
    assert taxes.iloc[1] == pytest.approx(april_supplier * 0.10)

    # Grand total tax = 10 % of total supplier cost
    total = result[(TariffCategory.TOTAL, CostGroup.TOTAL, "total")]
    expected_tax = (march_supplier + april_supplier) * 0.10
    assert taxes.sum() == pytest.approx(expected_tax)
    assert total.sum() == pytest.approx((march_supplier + april_supplier) * 1.10)


def test_contract_taxes_correct_for_complete_months_with_monthly_output() -> None:
    """When input data covers complete calendar months exactly, each month gets the
    correct tax (no off-by-one at period boundaries)."""
    from isodate import Duration

    contract = Contract(
        supplier=_tariff(energy_rate=100.0, start=dt.datetime(2024, 1, 1, tzinfo=dt.UTC)),
        taxes=Tax(versions=[TaxVersion(start=dt.datetime(2024, 1, 1, tzinfo=dt.UTC), default=0.10)]),
        timezone=dt.UTC,
    )

    start = dt.datetime(2024, 1, 1, tzinfo=dt.UTC)
    end = dt.datetime(2024, 4, 1, tzinfo=dt.UTC)  # three complete months
    timestamps = pd.date_range(start, end, freq="15min", inclusive="left")
    consumption = _consumption(timestamps, value=1.0)

    result = contract.apply(
        [Meter(data=consumption)],
        start=start,
        end=end,
        resolution=Duration(months=1),
    )

    assert result is not None
    assert len(result) == 3
    assert result["timestamp"].iloc[0] == pd.Timestamp("2024-01-01", tz=dt.UTC)
    assert result["timestamp"].iloc[1] == pd.Timestamp("2024-02-01", tz=dt.UTC)
    assert result["timestamp"].iloc[2] == pd.Timestamp("2024-03-01", tz=dt.UTC)

    taxes = result[(TariffCategory.TAXES, CostGroup.TOTAL, "total")]
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
        taxes=Tax(versions=[TaxVersion(start=dt.datetime(2024, 1, 1, tzinfo=dt.UTC), default=0.21)]),
        timezone=dt.UTC,
    )

    start = dt.datetime(2024, 2, 1, tzinfo=dt.UTC)
    end = dt.datetime(2024, 3, 1, tzinfo=dt.UTC)
    timestamps = pd.date_range(start, end, freq="15min", inclusive="left")
    consumption = _consumption(timestamps, value=2.0)

    result = contract.apply(
        [Meter(data=consumption)],
        start=start,
        end=end,
        resolution=Duration(months=1),
    )

    assert result is not None
    assert len(result) == 1
    assert result["timestamp"].iloc[0] == pd.Timestamp("2024-02-01", tz=dt.UTC)

    # Feb 2024 = 29 days (leap year)
    intervals = 29 * 24 * 4
    expected_supplier = intervals * 2.0 * 100.0
    tax = result[(TariffCategory.TAXES, CostGroup.TOTAL, "total")].iloc[0]
    assert pd.notna(tax)
    assert tax == pytest.approx(expected_supplier * 0.21)


def test_contract_taxes_zero_when_no_overlap_with_billing_window() -> None:
    """A billing window that does not overlap the output period should produce no tax rows.
    This ensures the overlap filter does not over-include periods."""
    from isodate import Duration

    contract = Contract(
        supplier=_tariff(energy_rate=100.0, start=dt.datetime(2024, 1, 1, tzinfo=dt.UTC)),
        taxes=Tax(versions=[TaxVersion(start=dt.datetime(2024, 1, 1, tzinfo=dt.UTC), default=0.10)]),
        timezone=dt.UTC,
    )

    # Data in January only
    start = dt.datetime(2024, 1, 10, tzinfo=dt.UTC)
    end = dt.datetime(2024, 1, 20, tzinfo=dt.UTC)
    timestamps = pd.date_range(start, end, freq="15min", inclusive="left")
    consumption = _consumption(timestamps, value=1.0)

    result = contract.apply(
        [Meter(data=consumption)],
        start=start,
        end=end,
        resolution=Duration(months=1),
    )

    assert result is not None
    assert len(result) == 1
    assert result["timestamp"].iloc[0] == pd.Timestamp("2024-01-01", tz=dt.UTC)

    taxes = result[(TariffCategory.TAXES, CostGroup.TOTAL, "total")]
    assert taxes.notna().all()
    assert (taxes > 0).all()

    # Only the January bucket covers [Jan 10, Jan 20) — no February bucket exists
    assert not any(result["timestamp"] == pd.Timestamp("2024-02-01", tz=dt.UTC))


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
    timestamps = pd.date_range("2025-01-01", periods=4, freq="15min", tz="Europe/Brussels")
    consumption = _consumption(timestamps, value=1.0)

    result = contract.apply([Meter(data=consumption)])

    assert result is not None
    assert not result.empty
    assert (TariffCategory.SUPPLIER, CostGroup.TOTAL, "total") in result.columns
    assert (TariffCategory.DISTRIBUTOR, CostGroup.TOTAL, "total") in result.columns
    assert (TariffCategory.FEES, CostGroup.TOTAL, "total") in result.columns
    assert (TariffCategory.TAXES, CostGroup.TOTAL, "total") in result.columns


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
# ContractHistory
# ---------------------------------------------------------------------------


def test_contract_history_single_contract() -> None:
    """A history with one contract produces the same result as calling contract.apply directly."""
    history = ContractHistory(
        versions=[
            Contract(
                start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
                supplier=_tariff(energy_rate=10.0),
            )
        ]
    )
    timestamps = pd.date_range("2025-01-01", periods=4, freq="15min", tz=dt.UTC)
    consumption = _consumption(timestamps, value=1.0)

    result = history.apply([Meter(data=consumption)])

    assert result is not None
    assert len(result) == 1
    assert (TariffCategory.SUPPLIER, CostGroup.TOTAL, "total") in result.columns
    assert result[(TariffCategory.TOTAL, CostGroup.TOTAL, "total")].iloc[0] == pytest.approx(40.0)


def test_contract_history_two_contracts_sequential() -> None:
    """Two sequential contracts each produce rows for their respective period."""
    history = ContractHistory(
        versions=[
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
    jan_ts = pd.date_range("2025-01-01", periods=4, freq="15min", tz=dt.UTC)
    feb_ts = pd.date_range("2025-02-01", periods=4, freq="15min", tz=dt.UTC)
    all_ts = pd.concat([pd.Series(jan_ts), pd.Series(feb_ts)], ignore_index=True)
    consumption = pd.DataFrame({"timestamp": all_ts, "value": 1.0})

    result = history.apply(
        [Meter(data=consumption)],
        start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
        end=dt.datetime(2025, 3, 1, tzinfo=dt.UTC),
    )

    assert result is not None
    assert len(result) == 2
    total = (TariffCategory.TOTAL, CostGroup.TOTAL, "total")
    # Jan: 4 × 1 × 10 = 40
    assert result[total].iloc[0] == pytest.approx(40.0)
    # Feb: 4 × 1 × 20 = 80
    assert result[total].iloc[1] == pytest.approx(80.0)


def test_contract_history_gap_produces_no_rows() -> None:
    """A gap between contracts produces no rows for the gap period."""
    history = ContractHistory(
        versions=[
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
        [Meter(data=consumption)],
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
        versions=[
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
    jan_ts = pd.date_range("2025-01-01", periods=4, freq="15min", tz=dt.UTC)
    feb_ts = pd.date_range("2025-02-01", periods=4, freq="15min", tz=dt.UTC)
    all_ts = pd.concat([pd.Series(jan_ts), pd.Series(feb_ts)], ignore_index=True)
    consumption = pd.DataFrame({"timestamp": all_ts, "value": 1.0})

    result = history.apply(
        [Meter(data=consumption)],
        start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
        end=dt.datetime(2025, 3, 1, tzinfo=dt.UTC),
    )

    assert result is not None
    assert len(result) == 2
    # Jan row should have distributor columns filled with 0
    dist_col = (TariffCategory.DISTRIBUTOR, CostGroup.TOTAL, "total")
    assert dist_col in result.columns
    assert result[dist_col].iloc[0] == pytest.approx(0.0)  # zero-filled
    assert result[dist_col].iloc[1] == pytest.approx(20.0)  # 4 × 1 × 5


def test_contract_history_returns_none_when_no_contracts_overlap() -> None:
    """Querying a period with no active contracts returns None."""
    history = ContractHistory(
        versions=[
            Contract(
                start=dt.datetime(2025, 6, 1, tzinfo=dt.UTC),
                supplier=_tariff(energy_rate=10.0),
            ),
        ]
    )
    ts = pd.date_range("2025-01-01", periods=4, freq="15min", tz=dt.UTC)
    consumption = _consumption(ts, value=1.0)

    result = history.apply(
        [Meter(data=consumption)],
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
    versions:
    - start: 2025-01-01T00:00:00+00:00
      consumption:
        constant_cost: 10.0
- start: 2025-06-01T00:00:00+00:00
  supplier:
    versions:
    - start: 2025-06-01T00:00:00+00:00
      consumption:
        constant_cost: 20.0
"""
    path = tmp_path / "history.yml"
    path.write_text(yaml_content, encoding="utf-8")

    history = ContractHistory.from_yaml(path)

    assert len(history.versions) == 2
    assert history.versions[0].end == dt.datetime(2025, 6, 1, tzinfo=dt.UTC)
    assert history.versions[1].end is None
