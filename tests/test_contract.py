from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from energy_cost.contract import Contract, TariffCategory
from energy_cost.formula import IndexFormula, PeriodicFormula
from energy_cost.fractional_periods import Period
from energy_cost.meter import CostGroup, Meter, MeterType, PowerDirection
from energy_cost.tariff import Tariff
from energy_cost.tariff_version import TariffVersion

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
        {"fixed": PeriodicFormula(period=Period.DAILY, constant_cost=daily_fixed)} if daily_fixed is not None else {}
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
    assert result[(CostGroup.CONSUMPTION, MeterType.SINGLE_RATE, "energy")].iloc[0] == pytest.approx(80.0)
    assert result[(CostGroup.TOTAL, MeterType.ALL, "total")].iloc[0] == pytest.approx(80.0)


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
    assert result[(CostGroup.CONSUMPTION, MeterType.SINGLE_RATE, "energy")].iloc[0] == pytest.approx(10.0)
    # Feb: 3 × 1 × 5 = 15 €
    assert result[(CostGroup.CONSUMPTION, MeterType.SINGLE_RATE, "energy")].iloc[1] == pytest.approx(15.0)


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
    assert result[(CostGroup.CONSUMPTION, MeterType.SINGLE_RATE, "energy")].iloc[0] == pytest.approx(20.0)  # 2 × 1 × 10


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
    assert (CostGroup.CAPACITY, MeterType.ALL, "total") in result.columns
    assert pd.notna(result[(CostGroup.CAPACITY, MeterType.ALL, "total")].iloc[0])


def test_apply_custom_output_resolution() -> None:
    """Costs can be aggregated to a daily resolution."""
    tariff = _tariff(energy_rate=10.0)
    timestamps = pd.date_range("2025-01-01", periods=8, freq="15min")
    consumption = _consumption(timestamps, value=1.0)

    result = tariff.apply([Meter(data=consumption)], resolution=dt.timedelta(days=1))

    assert result is not None
    assert len(result) == 1
    # 8 intervals × 1 MWh × 10 €/MWh = 80 €
    assert result[(CostGroup.CONSUMPTION, MeterType.SINGLE_RATE, "energy")].iloc[0] == pytest.approx(80.0)


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
    assert (CostGroup.CAPACITY, MeterType.ALL, "total") in result.columns
    assert (CostGroup.TOTAL, MeterType.ALL, "total") in result.columns


def test_apply_column_structure_is_three_level_multiindex() -> None:
    """The output columns form a three-level MultiIndex (excluding the timestamp column)."""
    tariff = _tariff(energy_rate=10.0)
    timestamps = pd.date_range("2025-01-01", periods=2, freq="15min")
    result = tariff.apply([Meter(data=_consumption(timestamps))])

    assert result is not None
    data_cols = [c for c in result.columns if c != "timestamp"]
    assert all(isinstance(c, tuple) and len(c) == 3 for c in data_cols)
    assert (CostGroup.CONSUMPTION, MeterType.SINGLE_RATE, "energy") in result.columns
    assert (CostGroup.TOTAL, MeterType.ALL, "total") in result.columns


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
    assert (CostGroup.INJECTION, MeterType.SINGLE_RATE, "energy") in result.columns
    # Consumption: 2 × 2 × 10 = 40; Injection: 2 × 1 × 5 = 10; Total = 50
    assert result[(CostGroup.CONSUMPTION, MeterType.SINGLE_RATE, "energy")].iloc[0] == pytest.approx(40.0)
    assert result[(CostGroup.INJECTION, MeterType.SINGLE_RATE, "energy")].iloc[0] == pytest.approx(10.0)
    assert result[(CostGroup.CONSUMPTION, MeterType.SINGLE_RATE, "total")].iloc[0] == pytest.approx(40.0)
    assert result[(CostGroup.INJECTION, MeterType.SINGLE_RATE, "total")].iloc[0] == pytest.approx(10.0)
    assert result[(CostGroup.TOTAL, MeterType.ALL, "total")].iloc[0] == pytest.approx(50.0)


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
    assert (CostGroup.FIXED, MeterType.ALL, "total") in result.columns
    assert result[(CostGroup.FIXED, MeterType.ALL, "total")].iloc[0] == pytest.approx(744.0)


# ---------------------------------------------------------------------------
# Contract.calculate
# ---------------------------------------------------------------------------


def test_contract_combines_provider_and_distributor() -> None:
    """calculate() returns a single DataFrame with provider + distributor data."""
    contract = Contract(
        provider=_tariff(energy_rate=100.0),
        distributor=_tariff(energy_rate=50.0),
    )
    timestamps = pd.date_range("2025-01-01", periods=2, freq="15min")
    consumption = _consumption(timestamps, value=1.0)

    result = contract.calculate_cost([Meter(data=consumption)])

    assert (TariffCategory.PROVIDER, CostGroup.CONSUMPTION, "energy") in result.columns
    assert (TariffCategory.DISTRIBUTOR, CostGroup.CONSUMPTION, "energy") in result.columns


def test_contract_taxes_applied_to_all_tariffs() -> None:
    """Taxes are computed on the sum of ALL tariff totals, including fees."""
    contract = Contract(
        provider=_tariff(energy_rate=100.0),
        distributor=_tariff(energy_rate=100.0),
        fees=_tariff(energy_rate=50.0),
        tax_rate=0.10,
    )
    timestamps = pd.date_range("2025-01-01", periods=2, freq="15min")
    # 2 intervals × 1 MWh each
    consumption = _consumption(timestamps, value=1.0)

    result = contract.calculate_cost([Meter(data=consumption)])

    provider_total = result[(TariffCategory.PROVIDER, CostGroup.TOTAL, "total")].iloc[0]
    distributor_total = result[(TariffCategory.DISTRIBUTOR, CostGroup.TOTAL, "total")].iloc[0]
    fees_total = result[(TariffCategory.FEES, CostGroup.TOTAL, "total")].iloc[0]
    taxes = result[(TariffCategory.TAXES, CostGroup.TOTAL, "total")].iloc[0]
    total_cost = result[(TariffCategory.TOTAL, CostGroup.TOTAL, "total")].iloc[0]

    # provider: 2 × 100 = 200; distributor: 2 × 100 = 200; fees: 2 × 50 = 100
    assert provider_total == pytest.approx(200.0)
    assert distributor_total == pytest.approx(200.0)
    assert fees_total == pytest.approx(100.0)
    # taxes = (200 + 200 + 100) × 0.10 = 50 (fees now included in tax base)
    assert taxes == pytest.approx(50.0)
    # total = 200 + 200 + 100 + 50 = 550
    assert total_cost == pytest.approx(550.0)


def test_contract_no_fees_omits_fees_columns() -> None:
    """When no fees tariff is present the result has no fees columns."""
    contract = Contract(
        provider=_tariff(energy_rate=100.0),
        distributor=_tariff(energy_rate=50.0),
    )
    timestamps = pd.date_range("2025-01-01", periods=2, freq="15min")
    result = contract.calculate_cost([Meter(data=_consumption(timestamps))])

    assert not any(isinstance(c, tuple) and c[0] == TariffCategory.FEES for c in result.columns)


def test_contract_column_structure_is_three_level_multiindex() -> None:
    """Data columns form a three-level MultiIndex by default (collapsed); timestamp is a plain column."""
    contract = Contract(
        provider=_tariff(energy_rate=10.0),
        distributor=_tariff(energy_rate=5.0),
        tax_rate=0.21,
    )
    timestamps = pd.date_range("2025-01-01", periods=2, freq="15min")
    result = contract.calculate_cost([Meter(data=_consumption(timestamps))])

    data_cols = [c for c in result.columns if c != "timestamp"]
    assert all(isinstance(c, tuple) and len(c) == 3 for c in data_cols)


def test_contract_expand_meter_types_gives_four_level_multiindex() -> None:
    """With expand_meter_types=True, data columns form a four-level MultiIndex."""
    contract = Contract(
        provider=_tariff(energy_rate=10.0),
        distributor=_tariff(energy_rate=5.0),
        tax_rate=0.21,
    )
    timestamps = pd.date_range("2025-01-01", periods=2, freq="15min")
    result = contract.calculate_cost([Meter(data=_consumption(timestamps))], expand_meter_types=True)

    data_cols = [c for c in result.columns if c != "timestamp"]
    assert all(isinstance(c, tuple) and len(c) == 4 for c in data_cols)


def test_contract_total_cost_equals_manual_sum() -> None:
    """total_cost == provider_total + distributor_total + taxes (no fees case)."""
    contract = Contract(
        provider=_tariff(energy_rate=100.0),
        distributor=_tariff(energy_rate=50.0),
        tax_rate=0.21,
    )
    timestamps = pd.date_range("2025-01-01", periods=4, freq="15min")
    consumption = _consumption(timestamps, value=2.0)

    result = contract.calculate_cost([Meter(data=consumption)])

    p = result[(TariffCategory.PROVIDER, CostGroup.TOTAL, "total")].iloc[0]
    d = result[(TariffCategory.DISTRIBUTOR, CostGroup.TOTAL, "total")].iloc[0]
    taxes = result[(TariffCategory.TAXES, CostGroup.TOTAL, "total")].iloc[0]
    total = result[(TariffCategory.TOTAL, CostGroup.TOTAL, "total")].iloc[0]

    # provider: 4 × 2 × 100 = 800; distributor: 4 × 2 × 50 = 400
    assert p == pytest.approx(800.0)
    assert d == pytest.approx(400.0)
    assert taxes == pytest.approx((800.0 + 400.0) * 0.21)
    assert total == pytest.approx(p + d + taxes)


# ---------------------------------------------------------------------------
# Contract.calculate_cost — timezone-aware start / end
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
        {"fixed": PeriodicFormula(period=Period.DAILY, constant_cost=daily_fixed)} if daily_fixed is not None else {}
    )
    version = TariffVersion(start=start, consumption=consumption, periodic=periodic)
    tariff = Tariff(versions=[version])
    return Contract(provider=tariff, distributor=tariff, timezone=_CET)


def test_calculate_cost_output_timestamps_match_input_timezone() -> None:
    """Timestamps in the result must retain the input data's timezone, not shift to UTC."""
    contract = _tz_contract(energy_rate=10.0)
    timestamps = pd.date_range("2025-01-01T00:00:00+01:00", periods=4, freq="15min")
    consumption = pd.DataFrame({"timestamp": timestamps, "value": 1.0})

    result = contract.calculate_cost([Meter(data=consumption)])

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

    result = contract.calculate_cost([Meter(data=consumption)], start=start, end=end)

    assert result is not None
    # 8 intervals × 2 MWh × 5 €/MWh = 80 €
    assert result[(TariffCategory.PROVIDER, CostGroup.TOTAL, "total")].iloc[0] == pytest.approx(80.0)
    assert result["timestamp"].iloc[0] == pd.Timestamp("2025-01-01T00:00:00+01:00")


# ---------------------------------------------------------------------------
# Tariff.apply / Contract.calculate_cost — TOU meters & multiple meters
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

    result = tou_tariff.apply([Meter(data=data, type=MeterType.TOU_PEAK)])

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
        ]
    )

    assert result is not None
    assert (CostGroup.CONSUMPTION, MeterType.SINGLE_RATE, "energy") in result.columns
    assert (CostGroup.CONSUMPTION, MeterType.TOU_PEAK, "energy") in result.columns
    # single_rate: 4 × 2 × 10 = 80 €; tou_peak: 4 × 3 × 20 = 240 €
    assert result[(CostGroup.CONSUMPTION, MeterType.SINGLE_RATE, "energy")].iloc[0] == pytest.approx(80.0)
    assert result[(CostGroup.CONSUMPTION, MeterType.TOU_PEAK, "energy")].iloc[0] == pytest.approx(240.0)
    assert result[(CostGroup.TOTAL, MeterType.ALL, "total")].iloc[0] == pytest.approx(320.0)


def test_contract_with_tou_meter_routes_cost_correctly() -> None:
    """contract.calculate_cost routes a TOU_PEAK meter through the correct formula."""
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
    contract = Contract(provider=tou_tariff, distributor=tou_tariff)
    timestamps = pd.date_range("2025-01-01", periods=4, freq="15min")
    data = _consumption(timestamps, value=1.0)

    result = contract.calculate_cost([Meter(data=data, type=MeterType.TOU_PEAK)], expand_meter_types=True)

    assert result is not None
    # provider + distributor each: 4 × 1 × 50 = 200 €
    assert result[(TariffCategory.PROVIDER, CostGroup.CONSUMPTION, MeterType.TOU_PEAK, "energy")].iloc[
        0
    ] == pytest.approx(200.0)
    assert result[(TariffCategory.DISTRIBUTOR, CostGroup.CONSUMPTION, MeterType.TOU_PEAK, "energy")].iloc[
        0
    ] == pytest.approx(200.0)
    assert result[(TariffCategory.TOTAL, CostGroup.TOTAL, MeterType.ALL, "total")].iloc[0] == pytest.approx(400.0)


def test_contract_with_injection_and_tou_meters() -> None:
    """Contract correctly handles a mix of injection and TOU consumption meters."""
    provider_tariff = Tariff(
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
    contract = Contract(provider=provider_tariff, distributor=distributor_tariff)
    timestamps = pd.date_range("2025-01-01", periods=4, freq="15min")
    cons_data = _consumption(timestamps, value=2.0)
    inj_data = _consumption(timestamps, value=1.0)

    result = contract.calculate_cost(
        [
            Meter(data=cons_data, type=MeterType.TOU_OFFPEAK),
            Meter(data=inj_data, direction=PowerDirection.INJECTION),
        ],
        expand_meter_types=True,
    )

    assert result is not None
    # provider consumption_tou_offpeak: 4 × 2 × 8 = 64 €; injection: 4 × 1 × 4 = 16 €
    assert result[(TariffCategory.PROVIDER, CostGroup.CONSUMPTION, MeterType.TOU_OFFPEAK, "energy")].iloc[
        0
    ] == pytest.approx(64.0)
    assert result[(TariffCategory.PROVIDER, CostGroup.INJECTION, MeterType.SINGLE_RATE, "energy")].iloc[
        0
    ] == pytest.approx(16.0)
    # distributor consumption_tou_offpeak: 4 × 2 × 2 = 16 €
    assert result[(TariffCategory.DISTRIBUTOR, CostGroup.CONSUMPTION, MeterType.TOU_OFFPEAK, "energy")].iloc[
        0
    ] == pytest.approx(16.0)
    # total: provider(64+16) + distributor(16) = 96 €
    assert result[(TariffCategory.TOTAL, CostGroup.TOTAL, MeterType.ALL, "total")].iloc[0] == pytest.approx(96.0)


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

    from energy_cost.data.be import distributors, fees, tax_rate
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
        provider=Tariff.model_validate({"versions": yaml.safe_load(tariff)}),
        distributor=distributors["fluvius_antwerpen"],
        fees=[fees["be_residential"], fees["flanders_residential"]],
        tax_rate=tax_rate,
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
    result = contract.calculate_cost(meters)
    expected = {
        TariffCategory.PROVIDER: 1.22,  # 4 × 0.0025 × (12.0 + 100*1.10) = 1.22 €
        TariffCategory.DISTRIBUTOR: 43.14186387067,  # (capacity 0.01 * 4116.9713583) + (consumption * 50.5027) + (data_management/12) => 41.1697 + 0.01 * 50.5027 + 17.85/12 = 43.162227 €
        TariffCategory.FEES: 0.475554,  # consumption * (excise + energy_contribution) => 0.475554 €
    }
    expected[TariffCategory.TAXES] = (sum(expected.values())) * tax_rate
    expected[TariffCategory.TOTAL] = sum(expected.values())

    assert result is not None
    # dataframe has one row starting on 2026-04-01 (monthly resolution)
    assert len(result) == 1
    assert result["timestamp"].iloc[0] == pd.Timestamp("2026-04-01T00:00:00+00:00")
    assert result[(TariffCategory.PROVIDER, CostGroup.TOTAL, "total")].iloc[0] == pytest.approx(
        expected[TariffCategory.PROVIDER]
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
        provider=_tariff(energy_rate=10.0),
        fees=[tariff_a, tariff_b],
    )
    timestamps = pd.date_range("2025-01-01", periods=2, freq="15min")
    consumption = _consumption(timestamps, value=1.0)

    result = contract.calculate_cost([Meter(data=consumption)])

    # fees: (2 × 100) + (2 × 50) = 300
    assert result[(TariffCategory.FEES, CostGroup.TOTAL, "total")].iloc[0] == pytest.approx(300.0)


# ---------------------------------------------------------------------------
# Default MeterType collapsing
# ---------------------------------------------------------------------------


def test_contract_collapses_meter_types_by_default() -> None:
    """contract.calculate_cost collapses MeterType by default (3-level output)."""
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
    contract = Contract(provider=tou_tariff)
    timestamps = pd.date_range("2025-01-01", periods=4, freq="15min")
    result = contract.calculate_cost(
        [
            Meter(data=_consumption(timestamps, value=1.0), type=MeterType.SINGLE_RATE),
            Meter(data=_consumption(timestamps, value=1.0), type=MeterType.TOU_PEAK),
        ]
    )

    data_cols = [c for c in result.columns if c != "timestamp"]
    assert all(isinstance(c, tuple) and len(c) == 3 for c in data_cols)
    # single_rate energy (4*10=40) + tou_peak energy (4*20=80) = 120
    assert result[(TariffCategory.PROVIDER, CostGroup.CONSUMPTION, "energy")].iloc[0] == pytest.approx(120.0)
    assert result[(TariffCategory.TOTAL, CostGroup.TOTAL, "total")].iloc[0] == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# Contract.calculate_cost — mixed-offset timestamps (DST boundary)
# ---------------------------------------------------------------------------


def test_calculate_cost_with_mixed_offset_meter_data() -> None:
    """A billing period crossing the DST boundary where meter readings carry different
    UTC offsets (+01:00 before, +02:00 after spring-forward) must not raise."""
    from zoneinfo import ZoneInfo

    from isodate import Duration

    from energy_cost.data.be import distributors, fees, tax_rate

    contract = Contract(
        distributor=distributors["fluvius_antwerpen"],
        fees=[fees["be_residential"], fees["flanders_residential"]],
        tax_rate=tax_rate,
        timezone=ZoneInfo("Europe/Brussels"),
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

    result = contract.calculate_cost(
        meters=[meter],
        start=dt.datetime.fromisoformat("2024-03-31T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2024-03-31T04:00:00+02:00"),
        resolution=Duration(months=1),
    )

    assert result is not None
    assert len(result) == 1  # one monthly billing row
