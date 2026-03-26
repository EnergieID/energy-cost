from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from energy_cost.fractional_periods import Period
from energy_cost.periodic_cost import PeriodicCost
from energy_cost.price_formula import PriceFormula
from energy_cost.scheduled_formula import ScheduledPriceFormulas
from energy_cost.tariff_version import (
    CostType,
    MeterType,
    PowerDirection,
    TariffVersion,
    _coerce_cost_type_formulas,
    _coerce_meter_formulas,
)


def _constant_cost(formula: PriceFormula | ScheduledPriceFormulas) -> float:
    assert isinstance(formula, PriceFormula)
    return formula.constant_cost


def test_all_meter_type_formula_is_used_for_single_rate_injection() -> None:
    segment = TariffVersion(
        start=dt.datetime(2025, 1, 1, 0, 0),
        injection={"all": {CostType.ENERGY: PriceFormula(constant_cost=-5.0)}},
    )

    single = segment.resolve_cost_formulas(MeterType.SINGLE_RATE, PowerDirection.INJECTION)

    assert _constant_cost(single[CostType.ENERGY]) == -5.0


def test_all_meter_type_formula_is_used_for_tou_peak_injection() -> None:
    segment = TariffVersion(
        start=dt.datetime(2025, 1, 1, 0, 0),
        injection={"all": {CostType.ENERGY: PriceFormula(constant_cost=-5.0)}},
    )

    tou = segment.resolve_cost_formulas(MeterType.TOU_PEAK, PowerDirection.INJECTION)

    assert _constant_cost(tou[CostType.ENERGY]) == -5.0


def test_meter_specific_formula_overrides_all_formula_for_single_rate_consumption() -> None:
    segment = TariffVersion(
        start=dt.datetime(2025, 1, 1, 0, 0),
        consumption={
            "all": {CostType.ENERGY: PriceFormula(constant_cost=1.0)},
            "single_rate": {CostType.ENERGY: PriceFormula(constant_cost=99.0)},
        },
    )

    single = segment.resolve_cost_formulas(MeterType.SINGLE_RATE, PowerDirection.CONSUMPTION)
    assert _constant_cost(single[CostType.ENERGY]) == 99.0


def test_all_formula_is_kept_when_no_meter_specific_override_exists() -> None:
    segment = TariffVersion(
        start=dt.datetime(2025, 1, 1, 0, 0),
        consumption={
            "all": {CostType.ENERGY: PriceFormula(constant_cost=1.0)},
            "single_rate": {CostType.ENERGY: PriceFormula(constant_cost=99.0)},
        },
    )

    tou = segment.resolve_cost_formulas(MeterType.TOU_PEAK, PowerDirection.CONSUMPTION)

    assert _constant_cost(tou[CostType.ENERGY]) == 1.0


def test_get_cost_returns_one_column_per_resolved_cost_type() -> None:
    segment = TariffVersion(
        start=dt.datetime(2025, 1, 1, 0, 0),
        consumption={
            "all": {
                CostType.ENERGY: PriceFormula(constant_cost=10.0),
                CostType.CHP_CERTIFICATES: PriceFormula(constant_cost=2.0),
                CostType.RENEWABLE_CERTIFICATES: PriceFormula(constant_cost=3.0),
            }
        },
    )

    out = segment.get_cost(
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 1, 0),
        resolution=dt.timedelta(minutes=15),
        meter_type=MeterType.SINGLE_RATE,
        direction=PowerDirection.CONSUMPTION,
    )

    assert set(out.columns) == {"timestamp", "energy", "chp_certificates", "renewable_certificates", "total"}
    assert out["energy"].tolist() == [10.0, 10.0, 10.0, 10.0]
    assert out["chp_certificates"].tolist() == [2.0, 2.0, 2.0, 2.0]
    assert out["renewable_certificates"].tolist() == [3.0, 3.0, 3.0, 3.0]
    assert out["total"].tolist() == [15.0, 15.0, 15.0, 15.0]


def test_get_periodic_cost_returns_prorated_cost_for_each_periodic_entry() -> None:
    segment = TariffVersion(
        start=dt.datetime(2025, 1, 1, 0, 0),
        periodic={
            "admin": PeriodicCost(period=Period.DAILY, constant_cost=24.0),
            "billing": PeriodicCost(period=Period.DAILY, constant_cost=12.0),
        },
    )

    costs = segment.get_periodic_cost(
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 1, 0),
    )

    assert costs == pytest.approx({"admin": 1.0, "billing": 0.5})


def test_model_validation_treats_bare_consumption_formula_as_all_meter_type_energy() -> None:
    segment = TariffVersion.model_validate(
        {
            "start": "2025-01-01T00:00:00",
            "consumption": {"constant_cost": 1.0},
        }
    )

    resolved = segment.resolve_cost_formulas(MeterType.SINGLE_RATE, PowerDirection.CONSUMPTION)

    assert _constant_cost(resolved[CostType.ENERGY]) == 1.0


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

    resolved = segment.resolve_cost_formulas(MeterType.TOU_PEAK, PowerDirection.CONSUMPTION)

    assert _constant_cost(resolved[CostType.ENERGY]) == 1.0
    assert _constant_cost(resolved[CostType.CHP_CERTIFICATES]) == 2.0


def test_model_validation_treats_scheduled_formula_list_as_all_meter_type_energy() -> None:
    segment = TariffVersion.model_validate(
        {
            "start": "2025-01-01T00:00:00+01:00",
            "consumption": [
                {
                    "when": [{"days": ["monday"], "start": "06:00:00", "end": "10:00:00"}],
                    "constant_cost": 300.0,
                },
                {"constant_cost": 100.0},
            ],
        }
    )

    resolved = segment.resolve_cost_formulas(MeterType.SINGLE_RATE, PowerDirection.CONSUMPTION)
    formula = resolved[CostType.ENERGY]

    assert isinstance(formula, ScheduledPriceFormulas)

    out = formula.get_values(
        dt.datetime.fromisoformat("2025-01-06T05:00:00+01:00"),
        dt.datetime.fromisoformat("2025-01-06T07:00:00+01:00"),
        dt.timedelta(hours=1),
    )
    assert out["value"].tolist() == [100.0, 300.0]


def test_cost_type_formula_coercion_wraps_bare_formula_as_energy_cost() -> None:
    bare_formula = {"constant_cost": 1.0}

    assert _coerce_cost_type_formulas(bare_formula) == {CostType.ENERGY: bare_formula}


def test_meter_formula_coercion_leaves_non_dict_values_unchanged() -> None:
    sentinel = "not-a-dict"

    assert _coerce_meter_formulas(sentinel) == sentinel


def test_get_cost_raises_when_all_resolved_formulas_return_empty_series() -> None:
    class EmptyPriceFormula(PriceFormula):
        def get_values(self, start: dt.datetime, end: dt.datetime, resolution: dt.timedelta) -> pd.DataFrame:
            return pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns]"), "value": pd.Series(dtype=float)})

    segment = TariffVersion(
        start=dt.datetime(2025, 1, 1, 0, 0),
        consumption={"all": {CostType.ENERGY: EmptyPriceFormula()}},
    )

    with pytest.raises(ValueError, match="No formulas for meter type 'single_rate' and direction 'consumption'"):
        segment.get_cost(
            start=dt.datetime(2025, 1, 1, 0, 0),
            end=dt.datetime(2025, 1, 1, 1, 0),
            resolution=dt.timedelta(minutes=15),
            meter_type=MeterType.SINGLE_RATE,
            direction=PowerDirection.CONSUMPTION,
        )
