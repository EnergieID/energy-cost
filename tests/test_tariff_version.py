from __future__ import annotations

import datetime as dt

import isodate
import pandas as pd
import pytest

from energy_cost.formula import Formula, IndexFormula, PeriodicFormula, ScheduledFormulas
from energy_cost.resolution import Resolution
from energy_cost.tariff_version import (
    MeterType,
    PowerDirection,
    TariffVersion,
    _coerce_meter_formulas,
    _coerce_named_formulas,
)


def _constant_cost(formula: Formula) -> float:
    assert isinstance(formula, IndexFormula)
    return formula.constant_cost


def test_all_meter_type_formula_is_used_for_single_rate_injection() -> None:
    segment = TariffVersion(
        start=dt.datetime(2025, 1, 1, 0, 0),
        injection={"all": {"energy": IndexFormula(constant_cost=-5.0)}},
    )

    single = segment._resolve_energy_formulas(MeterType.SINGLE_RATE, PowerDirection.INJECTION)

    assert _constant_cost(single["energy"]) == -5.0


def test_all_meter_type_formula_is_used_for_tou_peak_injection() -> None:
    segment = TariffVersion(
        start=dt.datetime(2025, 1, 1, 0, 0),
        injection={"all": {"energy": IndexFormula(constant_cost=-5.0)}},
    )

    tou = segment._resolve_energy_formulas(MeterType.TOU_PEAK, PowerDirection.INJECTION)

    assert _constant_cost(tou["energy"]) == -5.0


def test_meter_specific_formula_overrides_all_formula_for_single_rate_consumption() -> None:
    segment = TariffVersion(
        start=dt.datetime(2025, 1, 1, 0, 0),
        consumption={
            "all": {"energy": IndexFormula(constant_cost=1.0)},
            "single_rate": {"energy": IndexFormula(constant_cost=99.0)},
        },
    )

    single = segment._resolve_energy_formulas(MeterType.SINGLE_RATE, PowerDirection.CONSUMPTION)
    assert _constant_cost(single["energy"]) == 99.0


def test_all_formula_is_kept_when_no_meter_specific_override_exists() -> None:
    segment = TariffVersion(
        start=dt.datetime(2025, 1, 1, 0, 0),
        consumption={
            "all": {"energy": IndexFormula(constant_cost=1.0)},
            "single_rate": {"energy": IndexFormula(constant_cost=99.0)},
        },
    )

    tou = segment._resolve_energy_formulas(MeterType.TOU_PEAK, PowerDirection.CONSUMPTION)

    assert _constant_cost(tou["energy"]) == 1.0


def test_get_energy_cost_returns_one_column_per_resolved_cost_type() -> None:
    segment = TariffVersion(
        start=dt.datetime(2025, 1, 1, 0, 0),
        consumption={
            "all": {
                "energy": IndexFormula(constant_cost=10.0),
                "chp_certificates": IndexFormula(constant_cost=2.0),
                "renewable_certificates": IndexFormula(constant_cost=3.0),
            }
        },
    )

    out = segment.get_energy_cost(
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 1, 0),
        resolution=dt.timedelta(minutes=15),
        meter_type=MeterType.SINGLE_RATE,
        direction=PowerDirection.CONSUMPTION,
    )

    assert out is not None
    assert set(out.columns) == {"timestamp", "energy", "chp_certificates", "renewable_certificates", "total"}
    assert out["energy"].tolist() == [10.0, 10.0, 10.0, 10.0]
    assert out["chp_certificates"].tolist() == [2.0, 2.0, 2.0, 2.0]
    assert out["renewable_certificates"].tolist() == [3.0, 3.0, 3.0, 3.0]
    assert out["total"].tolist() == [15.0, 15.0, 15.0, 15.0]


def test_apply_periodic_costs_returns_prorated_cost_for_each_periodic_entry() -> None:
    segment = TariffVersion(
        start=dt.datetime(2025, 1, 1, 0, 0),
        periodic={
            "admin": PeriodicFormula(period=isodate.parse_duration("P1D"), constant_cost=24.0),
            "billing": PeriodicFormula(period=isodate.parse_duration("P1D"), constant_cost=12.0),
        },
    )

    result = segment.apply_periodic_costs(
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 2, 0, 0),
        output_resolution=dt.timedelta(hours=1),
    )

    assert result is not None
    assert result["admin"].sum() == pytest.approx(24.0)
    assert result["billing"].sum() == pytest.approx(12.0)


def test_model_validation_treats_bare_consumption_formula_as_all_meter_type_total() -> None:
    segment = TariffVersion.model_validate(
        {
            "start": "2025-01-01T00:00:00",
            "consumption": {"constant_cost": 1.0},
        }
    )

    resolved = segment._resolve_energy_formulas(MeterType.SINGLE_RATE, PowerDirection.CONSUMPTION)

    assert _constant_cost(resolved["total"]) == 1.0


def test_model_validation_treats_cost_type_map_as_shared_across_all_meter_types() -> None:
    segment = TariffVersion.model_validate(
        {
            "start": "2025-01-01T00:00:00",
            "consumption": {
                "energy": {"constant_cost": 1.0},
                "chp_certificates": {"constant_cost": 2.0},
            },
        }
    )

    resolved = segment._resolve_energy_formulas(MeterType.TOU_PEAK, PowerDirection.CONSUMPTION)

    assert _constant_cost(resolved["energy"]) == 1.0
    assert _constant_cost(resolved["chp_certificates"]) == 2.0


def test_model_validation_treats_scheduled_formula_dict_as_all_meter_type_total() -> None:
    segment = TariffVersion.model_validate(
        {
            "start": "2025-01-01T00:00:00+01:00",
            "consumption": {
                "kind": "scheduled",
                "schedule": [
                    {
                        "when": [{"days": ["monday"], "start": "06:00:00", "end": "10:00:00"}],
                        "formula": {"constant_cost": 300.0},
                    },
                    {"formula": {"constant_cost": 100.0}},
                ],
            },
        }
    )

    resolved = segment._resolve_energy_formulas(MeterType.SINGLE_RATE, PowerDirection.CONSUMPTION)
    formula = resolved["total"]

    assert isinstance(formula, ScheduledFormulas)

    out = formula.get_values(
        dt.datetime.fromisoformat("2025-01-06T05:00:00+01:00"),
        dt.datetime.fromisoformat("2025-01-06T07:00:00+01:00"),
        dt.timedelta(hours=1),
        timezone=dt.timezone(dt.timedelta(hours=1)),
    )
    assert out["value"].tolist() == [100.0, 300.0]


def test_cost_type_formula_coercion_wraps_bare_formula_as_total_cost() -> None:
    bare_formula = {"constant_cost": 1.0}

    result = _coerce_named_formulas(bare_formula)

    assert list(result.keys()) == ["total"]
    assert result["total"] == IndexFormula(constant_cost=1.0)


def test_meter_formula_coercion_leaves_non_dict_values_unchanged() -> None:
    sentinel = "not-a-dict"

    assert _coerce_meter_formulas(sentinel) == sentinel


def test_get_energy_cost_returns_none_when_all_resolved_formulas_return_empty_series() -> None:
    class EmptyIndexFormula(IndexFormula):
        def get_values(
            self, start: dt.datetime, end: dt.datetime, resolution: Resolution, timezone: dt.tzinfo = dt.UTC
        ) -> pd.DataFrame:
            return pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns]"), "value": pd.Series(dtype=float)})

    segment = TariffVersion(
        start=dt.datetime(2025, 1, 1, 0, 0),
        consumption={"all": {"energy": EmptyIndexFormula()}},
    )

    result = segment.get_energy_cost(
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 1, 0),
        resolution=dt.timedelta(minutes=15),
        meter_type=MeterType.SINGLE_RATE,
        direction=PowerDirection.CONSUMPTION,
    )
    assert result is None


def test_apply_energy_cost_returns_none_when_data_is_outside_start_end() -> None:
    """Data that falls entirely outside [start, end) becomes empty after slicing → returns None."""
    segment = TariffVersion(
        start=dt.datetime(2025, 1, 1, 0, 0),
        consumption={"all": {"energy": IndexFormula(constant_cost=10.0)}},
    )

    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-03-01", periods=4, freq="15min", tz=dt.UTC),
            "value": [1.0] * 4,
        }
    )

    result = segment.apply_energy_cost(
        data,
        meter_type=MeterType.SINGLE_RATE,
        direction=PowerDirection.CONSUMPTION,
        start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
        end=dt.datetime(2025, 2, 1, tzinfo=dt.UTC),
        input_resolution=dt.timedelta(minutes=15),
    )

    assert result is None


def test_apply_capacity_cost_returns_none_when_no_capacity_component_configured() -> None:
    segment = TariffVersion(
        start=dt.datetime(2025, 1, 1, 0, 0),
        injection={"all": {"energy": IndexFormula(constant_cost=-5.0)}},
    )

    assert (
        segment.apply_capacity_cost(
            pd.DataFrame({"timestamp": pd.to_datetime(["2025-01-01 00:00:00"]), "value": [10.0]}),
            start=dt.datetime(2025, 1, 1),
            end=dt.datetime(2025, 2, 1),
        )
        is None
    )


def test_apply_periodic_costs_returns_none_when_output_range_is_empty() -> None:
    """When end < start the output timestamp grid is empty and the result is None."""
    segment = TariffVersion(
        start=dt.datetime(2025, 1, 1, 0, 0),
        periodic={"admin": PeriodicFormula(period=isodate.parse_duration("P1D"), constant_cost=24.0)},
    )

    result = segment.apply_periodic_costs(
        start=dt.datetime(2025, 1, 2, 0, 0),
        end=dt.datetime(2025, 1, 1, 0, 0),  # end before start → empty grid
        output_resolution=dt.timedelta(hours=1),
    )

    assert result is None
