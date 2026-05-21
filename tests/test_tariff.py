from __future__ import annotations

import datetime as dt
from pathlib import Path
from zoneinfo import ZoneInfo

import isodate
import pandas as pd
import pytest

from energy_cost.formula import IndexFormula, PeriodicFormula
from energy_cost.meter import CostGroup, Meter, TimeseriesFrame
from energy_cost.tariff import Tariff
from energy_cost.tariff_version import TariffVersion


def _meter(timestamps: pd.DatetimeIndex, value: float = 1.0) -> Meter:
    return Meter(power=TimeseriesFrame(pd.DataFrame({"timestamp": timestamps, "value": value})))


def test_tariff_from_yaml_versioned_segments(tmp_path: Path) -> None:
    path = tmp_path / "tariff.yml"
    path.write_text(
        "- start: 2025-01-01T00:00:00\n"
        "  consumption:\n"
        "    energy:\n"
        "      constant_cost: 1.0\n"
        "- start: 2026-01-01T00:00:00\n"
        "  consumption:\n"
        "    energy:\n"
        "      constant_cost: 2.0\n",
        encoding="utf-8",
    )

    tariff = Tariff.from_yaml(path)

    assert len(tariff.root) == 2
    assert tariff.root[0].start == dt.datetime(2025, 1, 1, 0, 0)
    assert tariff.root[1].start == dt.datetime(2026, 1, 1, 0, 0)


def test_tariff_from_yaml_supports_scheduled_formula_dict() -> None:
    tariff = Tariff.from_yaml("examples/tariffs/scheduled.yml")

    out = tariff.get_values(
        start=dt.datetime.fromisoformat("2025-01-06T05:00:00+01:00"),
        end=dt.datetime.fromisoformat("2025-01-06T11:00:00+01:00"),
        output_resolution=dt.timedelta(hours=1),
        timezone=dt.timezone(dt.timedelta(hours=1)),
    )

    assert out is not None
    assert out["total"].tolist() == [100.0, 300.0, 300.0, 300.0, 300.0, 150.0]


def test_get_values_uses_correct_segment_for_time_range() -> None:
    tariff = Tariff(
        [
            TariffVersion(
                start=dt.datetime(2025, 1, 1, 0, 0),
                consumption={"energy": IndexFormula(constant_cost=1.0)},
            ),
            TariffVersion(
                start=dt.datetime(2025, 1, 1, 0, 30),
                consumption={"energy": IndexFormula(constant_cost=2.0)},
            ),
        ]
    )

    out = tariff.get_values(
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 1, 0),
        output_resolution=dt.timedelta(minutes=15),
    )

    assert out is not None
    assert out["timestamp"].tolist() == list(pd.date_range("2025-01-01", periods=4, freq="15min", tz=dt.UTC))
    assert out["energy"].tolist() == [1.0, 1.0, 2.0, 2.0]
    assert out["total"].tolist() == [1.0, 1.0, 2.0, 2.0]


def test_get_values_returns_none_when_no_formulas_found() -> None:
    tariff = Tariff([TariffVersion(start=dt.datetime(2025, 1, 1, 0, 0))])

    result = tariff.get_values(
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 1, 0),
    )
    assert result is None


def test_get_values_returns_none_when_no_versions_overlap_interval() -> None:
    tariff = Tariff(
        [
            TariffVersion(
                start=dt.datetime(2025, 1, 2, 0, 0),
                consumption={"energy": IndexFormula(constant_cost=1.0)},
            )
        ]
    )

    result = tariff.get_values(
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 1, 0),
    )
    assert result is None


def test_fixed_costs_span_multiple_segments() -> None:
    tariff = Tariff(
        [
            TariffVersion(
                start=dt.datetime(2025, 1, 1, 0, 0),
                fixed={"admin": PeriodicFormula(period=isodate.parse_duration("P1D"), constant_cost=24.0)},
            ),
            TariffVersion(
                start=dt.datetime(2025, 1, 1, 12, 0),
                fixed={"admin": PeriodicFormula(period=isodate.parse_duration("P1D"), constant_cost=48.0)},
            ),
        ]
    )
    ts = pd.date_range("2025-01-01", "2025-01-02", freq="h", tz=dt.UTC, inclusive="left")
    meter = _meter(ts)

    result = tariff.apply(
        meter,
        start=dt.datetime(2025, 1, 1, 0, 0, tzinfo=dt.UTC),
        end=dt.datetime(2025, 1, 2, 0, 0, tzinfo=dt.UTC),
        output_resolution=dt.timedelta(hours=1),
    )

    assert result is not None
    assert result[(CostGroup.FIXED, "admin")].sum() == pytest.approx(36.0)


def test_apply_returns_none_when_no_active_versions() -> None:
    tariff = Tariff(
        [
            TariffVersion(
                start=dt.datetime(2026, 1, 1, 0, 0),
                consumption={"energy": IndexFormula(constant_cost=1.0)},
            )
        ]
    )

    ts = pd.date_range("2025-01-01 00:00:00", periods=2, freq="h", tz=dt.UTC)
    result = tariff.apply(_meter(ts))
    assert result is None


# ---------------------------------------------------------------------------
# Timezone-aware start / end — Tariff.apply
# ---------------------------------------------------------------------------

_CET = dt.timezone(dt.timedelta(hours=1))


def _tz_tariff(*, energy_rate: float = 100.0, daily_fixed: float | None = None) -> Tariff:
    """Tariff whose version start is tz-aware (matches real YAML files)."""
    fixed: dict = (
        {"fixed_fee": PeriodicFormula(period=isodate.parse_duration("P1D"), constant_cost=daily_fixed)}
        if daily_fixed is not None
        else {}
    )
    return Tariff(
        [
            TariffVersion(
                start=dt.datetime(2025, 1, 1, 0, 0, tzinfo=_CET),
                consumption={"energy": IndexFormula(constant_cost=energy_rate)},
                fixed=fixed,
            )
        ]
    )


def test_apply_output_timestamps_match_input_timezone() -> None:
    """The output index timezone must equal the input data timezone."""
    tariff = _tz_tariff(energy_rate=10.0)
    timestamps = pd.date_range("2025-01-01T00:00:00+01:00", periods=4, freq="15min")
    result = tariff.apply(_meter(timestamps), timezone=_CET)

    assert result is not None
    assert result["timestamp"].dt.tz is not None
    assert result["timestamp"].iloc[0] == pd.Timestamp("2025-01-01T00:00:00+01:00")


def test_apply_zoneinfo_start_preserves_timezone_in_output() -> None:
    """zoneinfo-based start/end are accepted and output tz matches the data."""
    tariff = _tz_tariff(energy_rate=10.0)
    z = ZoneInfo("Europe/Brussels")
    timestamps = pd.date_range("2025-01-01T00:00:00", periods=4, freq="15min", tz=z)

    start = dt.datetime(2025, 1, 1, 0, 0, tzinfo=z)
    end = dt.datetime(2025, 2, 1, 0, 0, tzinfo=z)

    result = tariff.apply(_meter(timestamps), start=start, end=end, timezone=z)

    assert result is not None
    assert result["timestamp"].iloc[0] == pd.Timestamp("2025-01-01T00:00:00+01:00")


def test_apply_fixed_costs_timestamps_are_at_billing_start_not_utc() -> None:
    """Fixed-cost rows must be labelled at local midnight."""
    tariff = _tz_tariff(energy_rate=0.0, daily_fixed=24.0)
    timestamps = pd.date_range("2025-01-01T00:00:00+01:00", periods=48, freq="h")

    z = ZoneInfo("Europe/Brussels")
    start = dt.datetime(2025, 1, 1, 0, 0, tzinfo=z)
    result = tariff.apply(_meter(timestamps, value=0.0), start=start, timezone=_CET)

    assert result is not None
    assert (CostGroup.FIXED, "total") in result.columns
    assert result["timestamp"].iloc[0] == pd.Timestamp("2025-01-01T00:00:00+01:00")


def test_apply_capacity_includes_first_billing_month_when_start_is_tz_aware(tmp_path: Path) -> None:
    """Capacity for the first billing month must not be lost when start is tz-aware."""
    cap_yaml = tmp_path / "cap.yml"
    cap_yaml.write_text(
        "- start: 2025-01-01T00:00:00+01:00\n"
        "  capacity:\n"
        "    formula:\n"
        "      constant_cost: 1.0\n"
        "      capacity_based: true\n",
        encoding="utf-8",
    )
    from energy_cost.capacity import CapacityRule

    tariff = Tariff.from_yaml(cap_yaml)
    cap_rule = CapacityRule(
        measurement_period=isodate.parse_duration("PT15M"),
        billing_period=isodate.parse_duration("P1M"),
    )

    ts = pd.date_range("2025-01-01T00:00:00+01:00", "2026-01-01T00:00:00+01:00", freq="15min")
    raw_meter = Meter(power=TimeseriesFrame(pd.DataFrame({"timestamp": ts[:-1], "value": 5.0})))
    consumption = cap_rule.apply(raw_meter)

    z = ZoneInfo("Europe/Brussels")
    start = dt.datetime(2025, 1, 1, 0, 0, tzinfo=z)
    end = dt.datetime(2026, 1, 1, 0, 0, tzinfo=z)

    result = tariff.apply(consumption, start=start, end=end, timezone=z)

    assert result is not None
    assert len(result) == 12
    assert (CostGroup.CAPACITY, "total") in result.columns
    assert result["timestamp"].iloc[0].month == 1


# ---------------------------------------------------------------------------
# redistribute_to_resolution: consumption costs
# ---------------------------------------------------------------------------


def test_consumption_cost_redistributed_evenly_to_finer_resolution() -> None:
    """15-min cost split into 3 equal 5-min rows; their sum equals the original 15-min cost."""
    tariff = Tariff(
        [
            TariffVersion(
                start=dt.datetime(2025, 1, 1, 0, 0),
                consumption={"energy": IndexFormula(constant_cost=10.0)},
            )
        ]
    )
    timestamps = pd.date_range("2025-01-01", periods=4, freq="15min", tz=dt.UTC)
    result = tariff.apply(_meter(timestamps), output_resolution=dt.timedelta(minutes=5))

    assert result is not None
    energy_col = (CostGroup.CONSUMPTION, "energy")
    assert energy_col in result.columns
    assert len(result) == 12
    assert result[energy_col].iloc[0] == pytest.approx(10.0 / 3, rel=1e-6)
    assert result[energy_col].sum() == pytest.approx(4 * 10.0, rel=1e-6)


def test_consumption_cost_aggregated_correctly_to_coarser_resolution() -> None:
    """15-min costs summed to hourly buckets."""
    tariff = Tariff(
        [
            TariffVersion(
                start=dt.datetime(2025, 1, 1, 0, 0),
                consumption={"energy": IndexFormula(constant_cost=10.0)},
            )
        ]
    )
    timestamps = pd.date_range("2025-01-01", periods=8, freq="15min", tz=dt.UTC)
    result = tariff.apply(_meter(timestamps), output_resolution=dt.timedelta(hours=1))

    assert result is not None
    assert len(result) == 2
    assert result[(CostGroup.CONSUMPTION, "energy")].iloc[0] == pytest.approx(40.0)
    assert result[(CostGroup.CONSUMPTION, "energy")].iloc[1] == pytest.approx(40.0)


# ---------------------------------------------------------------------------
# redistribute_to_resolution: capacity costs
# ---------------------------------------------------------------------------


def test_capacity_cost_redistributed_evenly_to_finer_resolution(tmp_path: Path) -> None:
    """Monthly capacity cost is split uniformly across all 5-min output slots of that month."""
    from energy_cost.capacity import CapacityRule

    cap_yaml = tmp_path / "cap.yml"
    cap_yaml.write_text(
        "- start: 2025-01-01T00:00:00\n"
        "  capacity:\n"
        "    formula:\n"
        "      constant_cost: 1.0\n"
        "      capacity_based: true\n",
        encoding="utf-8",
    )
    tariff = Tariff.from_yaml(cap_yaml)
    cap_rule = CapacityRule(
        measurement_period=isodate.parse_duration("PT15M"),
        billing_period=isodate.parse_duration("P1M"),
    )

    # Use Jan+Feb to have 2 capacity rows (allows resolution detection)
    jan_feb_ts = pd.date_range("2025-01-01", "2025-03-01", freq="15min", tz=dt.UTC, inclusive="left")
    raw_meter = Meter(power=TimeseriesFrame(pd.DataFrame({"timestamp": jan_feb_ts, "value": 1.0})))
    consumption = cap_rule.apply(raw_meter)

    result = tariff.apply(
        consumption,
        start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
        end=dt.datetime(2025, 2, 1, tzinfo=dt.UTC),
        output_resolution=dt.timedelta(minutes=5),
    )

    assert result is not None
    cap_col = (CostGroup.CAPACITY, "total")
    assert cap_col in result.columns
    assert (result[cap_col] > 0).all(), "All 5-min capacity slots should be non-zero"
    # Peak = 1 MWh / 0.25 h = 4 MW; cost = 4 MW * 1.0 €/MW = 4.0 € for Jan
    assert result[cap_col].sum() == pytest.approx(4.0, rel=1e-6)


def test_capacity_cost_aggregated_to_yearly_output(tmp_path: Path) -> None:
    """Two monthly capacity rows aggregate into a single yearly output row."""
    from energy_cost.capacity import CapacityRule

    cap_yaml = tmp_path / "cap.yml"
    cap_yaml.write_text(
        "- start: 2025-01-01T00:00:00\n"
        "  capacity:\n"
        "    formula:\n"
        "      constant_cost: 1.0\n"
        "      capacity_based: true\n",
        encoding="utf-8",
    )
    tariff = Tariff.from_yaml(cap_yaml)
    cap_rule = CapacityRule(
        measurement_period=isodate.parse_duration("PT15M"),
        billing_period=isodate.parse_duration("P1M"),
    )

    jan_feb_ts = pd.date_range("2025-01-01", "2025-03-01", freq="15min", tz=dt.UTC, inclusive="left")
    raw_meter = Meter(power=TimeseriesFrame(pd.DataFrame({"timestamp": jan_feb_ts, "value": 1.0})))
    consumption = cap_rule.apply(raw_meter)

    result = tariff.apply(
        consumption,
        start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
        end=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
        output_resolution=isodate.parse_duration("P1Y"),
    )

    assert result is not None
    cap_col = (CostGroup.CAPACITY, "total")
    assert cap_col in result.columns
    # Peak = 4 MW each month; cost = 4 * 1.0 = 4.0 € per month; Jan + Feb = 8.0 €
    assert len(result) == 1
    assert result[cap_col].iloc[0] == pytest.approx(8.0, rel=1e-6)
