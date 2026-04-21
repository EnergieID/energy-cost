from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import isodate
import pandas as pd
import pytest

from energy_cost.formula import IndexFormula, PeriodicFormula
from energy_cost.meter import CostGroup, Meter, MeterType, PowerDirection, TariffCategory
from energy_cost.tariff import Tariff
from energy_cost.tariff_version import TariffVersion


def test_tariff_from_yaml_versioned_segments(tmp_path: Path) -> None:
    path = tmp_path / "tariff.yml"
    path.write_text(
        "- start: 2025-01-01T00:00:00\n"
        "  consumption:\n"
        "    all:\n"
        "      energy:\n"
        "        constant_cost: 1.0\n"
        "- start: 2026-01-01T00:00:00\n"
        "  consumption:\n"
        "    all:\n"
        "      energy:\n"
        "        constant_cost: 2.0\n",
        encoding="utf-8",
    )

    tariff = Tariff.from_yaml(path)

    assert len(tariff.versions) == 2
    assert tariff.versions[0].start == dt.datetime(2025, 1, 1, 0, 0)
    assert tariff.versions[1].start == dt.datetime(2026, 1, 1, 0, 0)


def test_tariff_from_yaml_supports_scheduled_formula_dict() -> None:
    tariff = Tariff.from_yaml("examples/tariffs/scheduled.yml")

    out = tariff.get_energy_cost(
        start=dt.datetime.fromisoformat("2025-01-06T05:00:00+01:00"),
        end=dt.datetime.fromisoformat("2025-01-06T11:00:00+01:00"),
        resolution=dt.timedelta(hours=1),
        timezone=dt.timezone(dt.timedelta(hours=1)),
    )

    assert out is not None
    assert out["total"].tolist() == [100.0, 300.0, 300.0, 300.0, 300.0, 150.0]


def test_get_energy_cost_uses_correct_segment_for_time_range() -> None:
    tariff = Tariff(
        versions=[
            TariffVersion(
                start=dt.datetime(2025, 1, 1, 0, 0),
                consumption={"all": {"energy": IndexFormula(constant_cost=1.0)}},
            ),
            TariffVersion(
                start=dt.datetime(2025, 1, 1, 0, 30),
                consumption={"all": {"energy": IndexFormula(constant_cost=2.0)}},
            ),
        ]
    )

    out = tariff.get_energy_cost(
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 1, 0),
        resolution=dt.timedelta(minutes=15),
    )

    assert out is not None
    assert out["timestamp"].tolist() == list(pd.date_range("2025-01-01", periods=4, freq="15min", tz=dt.UTC))
    assert out["energy"].tolist() == [1.0, 1.0, 2.0, 2.0]
    assert out["total"].tolist() == [1.0, 1.0, 2.0, 2.0]


def test_get_energy_cost_returns_none_when_no_formulas_found() -> None:
    tariff = Tariff(versions=[TariffVersion(start=dt.datetime(2025, 1, 1, 0, 0))])

    result = tariff.get_energy_cost(
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 1, 0),
    )
    assert result is None


def test_get_energy_cost_returns_none_when_no_versions_overlap_interval() -> None:
    tariff = Tariff(
        versions=[
            TariffVersion(
                start=dt.datetime(2025, 1, 2, 0, 0),
                consumption={"all": {"energy": IndexFormula(constant_cost=1.0)}},
            )
        ]
    )

    result = tariff.get_energy_cost(
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 1, 0),
    )
    assert result is None


def test_apply_periodic_costs_spans_multiple_segments() -> None:
    tariff = Tariff(
        versions=[
            TariffVersion(
                start=dt.datetime(2025, 1, 1, 0, 0),
                periodic={"admin": PeriodicFormula(period=isodate.parse_duration("P1D"), constant_cost=24.0)},
            ),
            TariffVersion(
                start=dt.datetime(2025, 1, 1, 12, 0),
                periodic={"admin": PeriodicFormula(period=isodate.parse_duration("P1D"), constant_cost=48.0)},
            ),
        ]
    )

    result = tariff.apply_periodic_costs(
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 2, 0, 0),
        output_resolution=dt.timedelta(hours=1),
    )

    assert result is not None
    assert result["admin"].sum() == pytest.approx(36.0)


def test_apply_capacity_returns_empty_dataframe_when_no_active_versions() -> None:
    tariff = Tariff(
        versions=[
            TariffVersion(
                start=dt.datetime(2026, 1, 1, 0, 0),
                consumption={"all": {"energy": IndexFormula(constant_cost=1.0)}},
            )
        ]
    )

    assert (
        tariff.apply_capacity_cost(
            pd.DataFrame(
                {"timestamp": pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 01:00:00"]), "value": [10.0, 20.0]}
            ),
        )
        is None
    )


def test_apply_capacity_returns_empty_dataframe_when_no_active_versions_with_capacity_cost() -> None:
    tariff = Tariff(
        versions=[
            TariffVersion(
                start=dt.datetime(2025, 1, 1, 0, 0),
                consumption={"all": {"energy": IndexFormula(constant_cost=1.0)}},
            )
        ]
    )

    assert (
        tariff.apply_capacity_cost(
            pd.DataFrame(
                {"timestamp": pd.to_datetime(["2025-01-01 00:00:00", "2025-01-01 01:00:00"]), "value": [10.0, 20.0]}
            ),
        )
        is None
    )


# ---------------------------------------------------------------------------
# Timezone-aware start / end — Tariff.apply
# ---------------------------------------------------------------------------

_CET = dt.timezone(dt.timedelta(hours=1))


def _tz_tariff(*, energy_rate: float = 100.0, daily_fixed: float | None = None) -> Tariff:
    """Tariff whose version start is tz-aware (matches real YAML files)."""
    periodic: dict = (
        {"fixed": PeriodicFormula(period=isodate.parse_duration("P1D"), constant_cost=daily_fixed)}
        if daily_fixed is not None
        else {}
    )
    return Tariff(
        versions=[
            TariffVersion(
                start=dt.datetime(2025, 1, 1, 0, 0, tzinfo=_CET),
                consumption={"all": {"energy": IndexFormula(constant_cost=energy_rate)}},
                periodic=periodic,
            )
        ]
    )


def test_apply_output_timestamps_match_input_timezone() -> None:
    """The output index timezone must equal the input data timezone."""
    tariff = _tz_tariff(energy_rate=10.0)
    timestamps = pd.date_range("2025-01-01T00:00:00+01:00", periods=4, freq="15min")
    consumption = pd.DataFrame({"timestamp": timestamps, "value": 1.0})

    result = tariff.apply([Meter(data=consumption)], timezone=_CET)

    assert result is not None
    assert result["timestamp"].dt.tz is not None
    # Must show local midnight, not UTC midnight
    assert result["timestamp"].iloc[0] == pd.Timestamp("2025-01-01T00:00:00+01:00")


def test_apply_zoneinfo_start_preserves_timezone_in_output() -> None:
    """zoneinfo-based start/end are accepted and output tz matches the data."""
    tariff = _tz_tariff(energy_rate=10.0)
    timestamps = pd.date_range("2025-01-01T00:00:00+01:00", periods=4, freq="15min")
    consumption = pd.DataFrame({"timestamp": timestamps, "value": 1.0})

    z = ZoneInfo("Europe/Brussels")
    start = dt.datetime(2025, 1, 1, 0, 0, tzinfo=z)
    end = dt.datetime(2025, 2, 1, 0, 0, tzinfo=z)

    result = tariff.apply([Meter(data=consumption)], start=start, end=end, timezone=z)

    assert result is not None
    assert result["timestamp"].iloc[0] == pd.Timestamp("2025-01-01T00:00:00+01:00")


def test_apply_fixed_costs_timestamps_are_at_billing_start_not_utc() -> None:
    """Fixed-cost rows must be labelled at local midnight."""
    tariff = _tz_tariff(energy_rate=0.0, daily_fixed=24.0)
    timestamps = pd.date_range("2025-01-01T00:00:00+01:00", periods=48, freq="h")
    consumption = pd.DataFrame({"timestamp": timestamps, "value": 0.0})

    z = ZoneInfo("Europe/Brussels")
    start = dt.datetime(2025, 1, 1, 0, 0, tzinfo=z)
    result = tariff.apply([Meter(data=consumption)], start=start, timezone=_CET)

    assert result is not None
    assert (CostGroup.FIXED, "total") in result.columns
    first_ts = result["timestamp"].iloc[0]
    assert first_ts == pd.Timestamp("2025-01-01T00:00:00+01:00")


def test_apply_capacity_includes_first_billing_month_when_start_is_tz_aware(tmp_path: Path) -> None:
    """Capacity for the first billing month must not be lost when start is tz-aware."""
    cap_yaml = tmp_path / "cap.yml"
    cap_yaml.write_text(
        "- start: 2025-01-01T00:00:00+01:00\n"
        "  capacity:\n"
        "    measurement_period: PT15M\n"
        "    billing_period: P1M\n"
        "    formula:\n"
        "      constant_cost: 1.0\n",
        encoding="utf-8",
    )
    tariff = Tariff.from_yaml(cap_yaml)

    ts = pd.date_range("2025-01-01T00:00:00+01:00", "2026-01-01T00:00:00+01:00", freq="15min")
    consumption = pd.DataFrame({"timestamp": ts[:-1], "value": 5.0})

    z = ZoneInfo("Europe/Brussels")
    start = dt.datetime(2025, 1, 1, 0, 0, tzinfo=z)
    end = dt.datetime(2026, 1, 1, 0, 0, tzinfo=z)

    result = tariff.apply([Meter(data=consumption)], start=start, end=end, timezone=z)

    assert result is not None
    assert len(result) == 12
    assert (CostGroup.CAPACITY, "total") in result.columns
    first_ts = result["timestamp"].iloc[0]
    assert first_ts.month == 1


# ---------------------------------------------------------------------------
# Tariff.apply – TOU meters
# ---------------------------------------------------------------------------


def test_apply_tou_peak_meter_uses_tou_formula() -> None:
    """A TOU_PEAK meter selects the tou_peak formula and the output column reflects the meter type."""
    tariff = Tariff(
        versions=[
            TariffVersion(
                start=dt.datetime(2025, 1, 1, 0, 0, tzinfo=_CET),
                consumption={
                    "single_rate": {"energy": IndexFormula(constant_cost=10.0)},
                    "tou_peak": {"energy": IndexFormula(constant_cost=20.0)},
                },
            )
        ]
    )
    timestamps = pd.date_range("2025-01-01T00:00:00+01:00", periods=4, freq="15min")
    data = pd.DataFrame({"timestamp": timestamps, "value": 1.0})

    result = tariff.apply([Meter(data=data, type=MeterType.TOU_PEAK)], include_meter_type=True)

    assert result is not None
    assert (CostGroup.CONSUMPTION, MeterType.TOU_PEAK, "energy") in result.columns
    assert (CostGroup.CONSUMPTION, MeterType.SINGLE_RATE, "energy") not in result.columns
    # 4 intervals × 1 MWh × 20 €/MWh = 80 €
    assert result[(CostGroup.CONSUMPTION, MeterType.TOU_PEAK, "energy")].iloc[0] == pytest.approx(80.0)
    assert result[(CostGroup.TOTAL, MeterType.ALL, "total")].iloc[0] == pytest.approx(80.0)


def test_apply_mixed_meter_types_produce_separate_columns() -> None:
    """A single_rate meter and a tou_peak meter each get their own output column group."""
    tariff = Tariff(
        versions=[
            TariffVersion(
                start=dt.datetime(2025, 1, 1, 0, 0, tzinfo=_CET),
                consumption={
                    "single_rate": {"energy": IndexFormula(constant_cost=10.0)},
                    "tou_peak": {"energy": IndexFormula(constant_cost=20.0)},
                },
            )
        ]
    )
    timestamps = pd.date_range("2025-01-01T00:00:00+01:00", periods=4, freq="15min")
    single_data = pd.DataFrame({"timestamp": timestamps, "value": 1.0})
    tou_data = pd.DataFrame({"timestamp": timestamps, "value": 2.0})

    result = tariff.apply(
        [
            Meter(data=single_data, type=MeterType.SINGLE_RATE),
            Meter(data=tou_data, type=MeterType.TOU_PEAK),
        ],
        include_meter_type=True,
    )

    assert result is not None
    assert (CostGroup.CONSUMPTION, MeterType.SINGLE_RATE, "energy") in result.columns
    assert (CostGroup.CONSUMPTION, MeterType.TOU_PEAK, "energy") in result.columns
    # single_rate: 4 × 1 × 10 = 40 €; tou_peak: 4 × 2 × 20 = 160 €
    assert result[(CostGroup.CONSUMPTION, MeterType.SINGLE_RATE, "energy")].iloc[0] == pytest.approx(40.0)
    assert result[(CostGroup.CONSUMPTION, MeterType.TOU_PEAK, "energy")].iloc[0] == pytest.approx(160.0)
    assert result[(CostGroup.TOTAL, MeterType.ALL, "total")].iloc[0] == pytest.approx(200.0)


def test_apply_tou_offpeak_and_injection_meters() -> None:
    """TOU_OFFPEAK consumption and injection meters are handled independently."""
    tariff = Tariff(
        versions=[
            TariffVersion(
                start=dt.datetime(2025, 1, 1, 0, 0, tzinfo=_CET),
                consumption={
                    "tou_offpeak": {"energy": IndexFormula(constant_cost=5.0)},
                },
                injection={
                    "all": {"energy": IndexFormula(constant_cost=3.0)},
                },
            )
        ]
    )
    timestamps = pd.date_range("2025-01-01T00:00:00+01:00", periods=4, freq="15min")
    cons_data = pd.DataFrame({"timestamp": timestamps, "value": 1.0})
    inj_data = pd.DataFrame({"timestamp": timestamps, "value": 0.5})

    result = tariff.apply(
        [
            Meter(data=cons_data, type=MeterType.TOU_OFFPEAK),
            Meter(data=inj_data, direction=PowerDirection.INJECTION),
        ],
        include_meter_type=True,
    )

    assert result is not None
    assert (CostGroup.CONSUMPTION, MeterType.TOU_OFFPEAK, "energy") in result.columns
    assert (CostGroup.INJECTION, MeterType.SINGLE_RATE, "energy") in result.columns
    # consumption: 4 × 1 × 5 = 20 €; injection: 4 × 0.5 × 3 = 6 €
    assert result[(CostGroup.CONSUMPTION, MeterType.TOU_OFFPEAK, "energy")].iloc[0] == pytest.approx(20.0)
    assert result[(CostGroup.INJECTION, MeterType.SINGLE_RATE, "energy")].iloc[0] == pytest.approx(6.0)
    assert result[(CostGroup.TOTAL, MeterType.ALL, "total")].iloc[0] == pytest.approx(26.0)


# ---------------------------------------------------------------------------
# TariffVersion – apply with Tariff category
# ---------------------------------------------------------------------------


def test_apply_returns_extra_index_level_and_total_if_tariff_category_provided() -> None:
    """If a tariff category is provided, the output columns get an extra index level and a total column."""
    tariff = Tariff(
        versions=[
            TariffVersion(
                start=dt.datetime(2025, 1, 1, 0, 0),
                consumption={"all": {"energy": IndexFormula(constant_cost=10.0)}},
            )
        ]
    )
    timestamps = pd.date_range("2025-01-01T00:00:00", periods=4, freq="15min")
    data = pd.DataFrame({"timestamp": timestamps, "value": 1.0})

    result = tariff.apply([Meter(data=data)], include_meter_type=False, tariff_category=TariffCategory.FEES)

    assert result is not None
    assert (TariffCategory.FEES, CostGroup.CONSUMPTION, "energy") in result.columns
    assert (TariffCategory.FEES, CostGroup.TOTAL, "total") in result.columns
    # 4 intervals × 1 MWh × 10 €/MWh = 40 €
    assert result[(TariffCategory.FEES, CostGroup.CONSUMPTION, "energy")].iloc[0] == pytest.approx(40.0)
    assert result[(TariffCategory.FEES, CostGroup.TOTAL, "total")].iloc[0] == pytest.approx(40.0)


def test_apply_correctly_returns_four_index_levels_when_include_meter_type_and_tariff_category() -> None:
    """When both include_meter_type and tariff_category are set, the output columns have four index levels."""
    tariff = Tariff(
        versions=[
            TariffVersion(
                start=dt.datetime(2025, 1, 1, 0, 0),
                consumption={"all": {"energy": IndexFormula(constant_cost=10.0)}},
            )
        ]
    )
    timestamps = pd.date_range("2025-01-01T00:00:00", periods=4, freq="15min")
    data = pd.DataFrame({"timestamp": timestamps, "value": 1.0})

    result = tariff.apply(
        [Meter(data=data, type=MeterType.TOU_PEAK), Meter(data=data, type=MeterType.TOU_OFFPEAK)],
        include_meter_type=True,
        tariff_category=TariffCategory.FEES,
    )

    assert result is not None
    assert (TariffCategory.FEES, CostGroup.CONSUMPTION, MeterType.TOU_PEAK, "energy") in result.columns
    assert (TariffCategory.FEES, CostGroup.CONSUMPTION, MeterType.TOU_OFFPEAK, "energy") in result.columns
    assert (TariffCategory.FEES, CostGroup.TOTAL, MeterType.ALL, "total") in result.columns
    # Each meter: 4 intervals × 1 MWh × 10 €/MWh = 40 €; two meters = 80 € total
    assert result[(TariffCategory.FEES, CostGroup.CONSUMPTION, MeterType.TOU_PEAK, "energy")].iloc[0] == pytest.approx(
        40.0
    )
    assert result[(TariffCategory.FEES, CostGroup.CONSUMPTION, MeterType.TOU_OFFPEAK, "energy")].iloc[
        0
    ] == pytest.approx(40.0)
    assert result[(TariffCategory.FEES, CostGroup.TOTAL, MeterType.ALL, "total")].iloc[0] == pytest.approx(80.0)


# ---------------------------------------------------------------------------
# resample_or_distribute: direction costs
# ---------------------------------------------------------------------------


def test_direction_cost_redistributed_evenly_to_finer_resolution() -> None:
    """15-min cost split into 3 equal 5-min rows; their sum equals the original 15-min cost."""
    tariff = Tariff(
        versions=[
            TariffVersion(
                start=dt.datetime(2025, 1, 1, 0, 0),
                consumption={"all": {"energy": IndexFormula(constant_cost=10.0)}},
            )
        ]
    )
    # 4 × 15-min intervals; each has 1 MWh → cost = 10 € per 15-min slot
    timestamps = pd.date_range("2025-01-01", periods=4, freq="15min", tz=dt.UTC)
    data = pd.DataFrame({"timestamp": timestamps, "value": 1.0})

    result = tariff.apply([Meter(data=data)], resolution=dt.timedelta(minutes=5))

    assert result is not None
    # Each 15-min slot becomes 3 equal 5-min rows → sum over those 3 should equal original 10 €
    energy_col = (CostGroup.CONSUMPTION, "energy")
    assert energy_col in result.columns
    # 4 intervals × 3 sub-slots = 12 rows
    assert len(result) == 12
    # All rows should have the same non-zero value (10 / 3 ≈ 3.333… €)
    assert result[energy_col].iloc[0] == pytest.approx(10.0 / 3, rel=1e-6)
    assert result[energy_col].iloc[1] == pytest.approx(10.0 / 3, rel=1e-6)
    assert result[energy_col].iloc[2] == pytest.approx(10.0 / 3, rel=1e-6)
    # Grand total must equal the sum had we used the 15-min resolution
    assert result[energy_col].sum() == pytest.approx(4 * 10.0, rel=1e-6)


def test_direction_cost_aggregated_correctly_to_coarser_resolution() -> None:
    """15-min costs summed to hourly buckets."""
    tariff = Tariff(
        versions=[
            TariffVersion(
                start=dt.datetime(2025, 1, 1, 0, 0),
                consumption={"all": {"energy": IndexFormula(constant_cost=10.0)}},
            )
        ]
    )
    timestamps = pd.date_range("2025-01-01", periods=8, freq="15min", tz=dt.UTC)
    data = pd.DataFrame({"timestamp": timestamps, "value": 1.0})

    result = tariff.apply([Meter(data=data)], resolution=dt.timedelta(hours=1))

    assert result is not None
    # 8 intervals → 2 hourly buckets; each hour = 4 × 10 = 40 €
    assert len(result) == 2
    assert result[(CostGroup.CONSUMPTION, "energy")].iloc[0] == pytest.approx(40.0)
    assert result[(CostGroup.CONSUMPTION, "energy")].iloc[1] == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# resample_or_distribute: capacity costs
# ---------------------------------------------------------------------------


def test_capacity_cost_redistributed_evenly_to_finer_resolution(tmp_path) -> None:
    """Monthly capacity cost is split uniformly across all 5-min output slots of that month."""
    cap_yaml = tmp_path / "cap.yml"
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

    # 15-min readings for all of January; peak = 1 MW → capacity cost = 1.0 €
    jan_ts = pd.date_range("2025-01-01", "2025-02-01", freq="15min", tz=dt.UTC, inclusive="left")
    consumption = pd.DataFrame({"timestamp": jan_ts, "value": 1.0})

    result = tariff.apply([Meter(data=consumption)], resolution=dt.timedelta(minutes=5))

    assert result is not None
    cap_col = (CostGroup.CAPACITY, "total")
    assert cap_col in result.columns
    # Every 5-min slot should carry a small non-zero (uniformly distributed) capacity cost
    assert (result[cap_col] > 0).all(), "All 5-min capacity slots should be non-zero"
    # Peak = 1 MWh / 0.25 h = 4 MW; cost = 4 MW × 1.0 €/MW = 4.0 € for Jan
    # Sum over all 5-min slots must equal that monthly total
    assert result[cap_col].sum() == pytest.approx(4.0, rel=1e-6)


def test_capacity_cost_aggregated_to_yearly_output(tmp_path) -> None:
    """Two monthly capacity rows aggregate into a single yearly output row."""
    cap_yaml = tmp_path / "cap.yml"
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

    jan_feb_ts = pd.date_range("2025-01-01", "2025-03-01", freq="15min", tz=dt.UTC, inclusive="left")
    consumption = pd.DataFrame({"timestamp": jan_feb_ts, "value": 1.0})

    result = tariff.apply(
        [Meter(data=consumption)],
        start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
        end=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
        resolution=isodate.parse_duration("P1Y"),
    )

    assert result is not None
    cap_col = (CostGroup.CAPACITY, "total")
    assert cap_col in result.columns
    # Peak = 4 MW each month; cost = 4 × 1.0 = 4.0 € per month
    # Jan + Feb = 2 × 4.0 = 8.0 €
    assert len(result) == 1
    assert result[cap_col].iloc[0] == pytest.approx(8.0, rel=1e-6)
