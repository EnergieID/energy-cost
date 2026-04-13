from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from energy_cost.formula import IndexFormula, PeriodicFormula
from energy_cost.fractional_periods import Period
from energy_cost.meter import CostGroup, Meter, MeterType, PowerDirection
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


def test_get_periodic_cost_spans_multiple_segments() -> None:
    tariff = Tariff(
        versions=[
            TariffVersion(
                start=dt.datetime(2025, 1, 1, 0, 0),
                periodic={"admin": PeriodicFormula(period=Period.DAILY, constant_cost=24.0)},
            ),
            TariffVersion(
                start=dt.datetime(2025, 1, 1, 12, 0),
                periodic={"admin": PeriodicFormula(period=Period.DAILY, constant_cost=48.0)},
            ),
        ]
    )

    costs = tariff.get_periodic_cost(
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 2, 0, 0),
    )

    assert costs == pytest.approx({"admin": 36.0})


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
        {"fixed": PeriodicFormula(period=Period.DAILY, constant_cost=daily_fixed)} if daily_fixed is not None else {}
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
    assert (CostGroup.FIXED, MeterType.ALL, "total") in result.columns
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
    assert (CostGroup.CAPACITY, MeterType.ALL, "total") in result.columns
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

    result = tariff.apply([Meter(data=data, type=MeterType.TOU_PEAK)])

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
        ]
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
        ]
    )

    assert result is not None
    assert (CostGroup.CONSUMPTION, MeterType.TOU_OFFPEAK, "energy") in result.columns
    assert (CostGroup.INJECTION, MeterType.SINGLE_RATE, "energy") in result.columns
    # consumption: 4 × 1 × 5 = 20 €; injection: 4 × 0.5 × 3 = 6 €
    assert result[(CostGroup.CONSUMPTION, MeterType.TOU_OFFPEAK, "energy")].iloc[0] == pytest.approx(20.0)
    assert result[(CostGroup.INJECTION, MeterType.SINGLE_RATE, "energy")].iloc[0] == pytest.approx(6.0)
    assert result[(CostGroup.TOTAL, MeterType.ALL, "total")].iloc[0] == pytest.approx(26.0)
