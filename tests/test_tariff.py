from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from energy_cost.price_formula import PriceFormula
from energy_cost.tariff import CostType, MeterType, PowerDirection, Tariff, TimedPriceFormula


def _constant_cost(tpf: TimedPriceFormula) -> float:
    """Narrow formula to PriceFormula and return its constant_cost (test helper)."""
    assert isinstance(tpf.formula, PriceFormula)
    return tpf.formula.constant_cost


def test_timed_price_formula_returns_empty_when_outside_range() -> None:
    timed = TimedPriceFormula(
        start=dt.datetime(2025, 1, 2, 0, 0),
        formula=PriceFormula(constant_cost=1.0),
    )

    out = timed.get_values(
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 1, 0),
        resolution=dt.timedelta(minutes=15),
    )

    assert list(out.columns) == ["timestamp", "value"]
    assert out.empty


def test_tariff_from_yaml_loads_file(tmp_path: Path) -> None:
    path = tmp_path / "tariff.yml"
    path.write_text(
        "supplier: Demo\n"
        "product: Dynamic\n"
        "by_meter_type:\n"
        "  single_rate:\n"
        "    consumption:\n"
        "      energy:\n"
        "        - start: 2025-01-01T00:00:00\n"
        "          formula:\n"
        "            constant_cost: 1.25\n"
        "            variable_costs: []\n",
        encoding="utf-8",
    )

    tariff = Tariff.from_yaml(path)

    assert tariff.supplier == "Demo"
    assert tariff.product == "Dynamic"
    assert list(tariff.by_meter_type.keys()) == [MeterType.SINGLE_RATE]


def test_filter_formulas_returns_only_overlapping_items() -> None:
    formulas = [
        TimedPriceFormula(start=dt.datetime(2025, 1, 1, 0, 0), formula=PriceFormula(constant_cost=1.0)),
        TimedPriceFormula(start=dt.datetime(2025, 1, 1, 1, 0), formula=PriceFormula(constant_cost=2.0)),
        TimedPriceFormula(start=dt.datetime(2025, 1, 1, 2, 0), formula=PriceFormula(constant_cost=3.0)),
    ]
    tariff = Tariff(
        supplier="S",
        product="P",
        by_meter_type={MeterType.SINGLE_RATE: {PowerDirection.CONSUMPTION: {CostType.ENERGY: formulas}}},
    )

    resolved = tariff.resolve_cost_formulas(MeterType.SINGLE_RATE, PowerDirection.CONSUMPTION)
    out = tariff.filter_formulas(
        formulas=resolved[CostType.ENERGY],
        start=dt.datetime(2025, 1, 1, 1, 30),
        end=dt.datetime(2025, 1, 1, 2, 30),
    )

    assert [_constant_cost(x) for x in out] == [2.0, 3.0]


def test_get_cost_concatenates_formula_segments() -> None:
    formulas = [
        TimedPriceFormula(start=dt.datetime(2025, 1, 1, 0, 0), formula=PriceFormula(constant_cost=1.0)),
        TimedPriceFormula(start=dt.datetime(2025, 1, 1, 0, 30), formula=PriceFormula(constant_cost=2.0)),
    ]
    tariff = Tariff(
        supplier="S",
        product="P",
        by_meter_type={MeterType.SINGLE_RATE: {PowerDirection.CONSUMPTION: {CostType.ENERGY: formulas}}},
    )

    out = tariff.get_cost(
        meter_type=MeterType.SINGLE_RATE,
        direction=PowerDirection.CONSUMPTION,
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 1, 0),
        resolution=dt.timedelta(minutes=15),
    )

    assert out["timestamp"].tolist() == list(pd.date_range("2025-01-01", periods=4, freq="15min"))
    assert out["energy"].tolist() == [1.0, 1.0, 2.0, 2.0]
    assert out["total"].tolist() == [1.0, 1.0, 2.0, 2.0]


def test_get_cost_raises_when_meter_type_missing() -> None:
    tariff = Tariff(supplier="S", product="P")

    with pytest.raises(
        ValueError, match="No formulas for meter type 'single_rate' and direction 'consumption' found in tariff"
    ):
        tariff.get_cost(
            meter_type=MeterType.SINGLE_RATE,
            direction=PowerDirection.CONSUMPTION,
            start=dt.datetime(2025, 1, 1, 0, 0),
            end=dt.datetime(2025, 1, 1, 1, 0),
            resolution=dt.timedelta(minutes=15),
        )


def test_defaults_apply_to_all_meter_types() -> None:
    injection_formulas = [
        TimedPriceFormula(start=dt.datetime(2025, 1, 1, 0, 0), formula=PriceFormula(constant_cost=-5.0)),
    ]
    tariff = Tariff(
        supplier="S",
        product="P",
        defaults={PowerDirection.INJECTION: {CostType.ENERGY: injection_formulas}},
    )

    single = tariff.filter_formulas(
        formulas=tariff.resolve_cost_formulas(MeterType.SINGLE_RATE, PowerDirection.INJECTION)[CostType.ENERGY],
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 1, 0),
    )
    tou = tariff.filter_formulas(
        formulas=tariff.resolve_cost_formulas(MeterType.TOU_PEAK, PowerDirection.INJECTION)[CostType.ENERGY],
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 1, 0),
    )

    assert [_constant_cost(f) for f in single] == [-5.0]
    assert [_constant_cost(f) for f in tou] == [-5.0]


def test_by_meter_type_overrides_defaults() -> None:
    default_formulas = [
        TimedPriceFormula(start=dt.datetime(2025, 1, 1, 0, 0), formula=PriceFormula(constant_cost=1.0)),
    ]
    override_formulas = [
        TimedPriceFormula(start=dt.datetime(2025, 1, 1, 0, 0), formula=PriceFormula(constant_cost=99.0)),
    ]
    tariff = Tariff(
        supplier="S",
        product="P",
        defaults={PowerDirection.CONSUMPTION: {CostType.ENERGY: default_formulas}},
        by_meter_type={MeterType.SINGLE_RATE: {PowerDirection.CONSUMPTION: {CostType.ENERGY: override_formulas}}},
    )

    single = tariff.filter_formulas(
        formulas=tariff.resolve_cost_formulas(MeterType.SINGLE_RATE, PowerDirection.CONSUMPTION)[CostType.ENERGY],
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 1, 0),
    )
    tou = tariff.filter_formulas(
        formulas=tariff.resolve_cost_formulas(MeterType.TOU_PEAK, PowerDirection.CONSUMPTION)[CostType.ENERGY],
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 1, 0),
    )

    assert [_constant_cost(f) for f in single] == [99.0]
    assert [_constant_cost(f) for f in tou] == [1.0]


def test_get_cost_returns_column_per_cost_type() -> None:
    energy_formulas = [
        TimedPriceFormula(start=dt.datetime(2025, 1, 1, 0, 0), formula=PriceFormula(constant_cost=10.0)),
    ]
    chp_formulas = [
        TimedPriceFormula(start=dt.datetime(2025, 1, 1, 0, 0), formula=PriceFormula(constant_cost=2.0)),
    ]
    renewable_formulas = [
        TimedPriceFormula(start=dt.datetime(2025, 1, 1, 0, 0), formula=PriceFormula(constant_cost=3.0)),
    ]
    tariff = Tariff(
        supplier="S",
        product="P",
        by_meter_type={
            MeterType.SINGLE_RATE: {
                PowerDirection.CONSUMPTION: {
                    CostType.ENERGY: energy_formulas,
                    CostType.CHP_CERTIFICATES: chp_formulas,
                    CostType.RENEWABLE_CERTIFICATES: renewable_formulas,
                }
            }
        },
    )

    out = tariff.get_cost(
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 1, 0),
    )

    assert list(out.columns) == ["timestamp", "energy", "chp_certificates", "renewable_certificates", "total"]
    assert out["energy"].tolist() == [10.0, 10.0, 10.0, 10.0]
    assert out["chp_certificates"].tolist() == [2.0, 2.0, 2.0, 2.0]
    assert out["renewable_certificates"].tolist() == [3.0, 3.0, 3.0, 3.0]
    assert out["total"].tolist() == [15.0, 15.0, 15.0, 15.0]


def test_list_shorthand_is_interpreted_as_energy(tmp_path: Path) -> None:
    """A bare list of TimedPriceFormula should be treated as ``{energy: <list>}``."""
    path = tmp_path / "tariff.yml"
    path.write_text(
        "supplier: Demo\n"
        "product: Short\n"
        "defaults:\n"
        "  injection:\n"
        "    - start: 2025-01-01T00:00:00\n"
        "      formula:\n"
        "        constant_cost: -1.0\n",
        encoding="utf-8",
    )

    tariff = Tariff.from_yaml(path)
    resolved = tariff.resolve_cost_formulas(MeterType.SINGLE_RATE, PowerDirection.INJECTION)

    assert list(resolved.keys()) == [CostType.ENERGY]
    assert _constant_cost(resolved[CostType.ENERGY][0]) == -1.0


def test_direction_shorthand_bare_list_defaults_to_consumption(tmp_path: Path) -> None:
    """A bare list at the meter-type level implies consumption + energy."""
    path = tmp_path / "tariff.yml"
    path.write_text(
        "supplier: Demo\n"
        "product: Short\n"
        "by_meter_type:\n"
        "  single_rate:\n"
        "    - start: 2025-01-01T00:00:00\n"
        "      formula:\n"
        "        constant_cost: 7.0\n",
        encoding="utf-8",
    )

    tariff = Tariff.from_yaml(path)
    resolved = tariff.resolve_cost_formulas(MeterType.SINGLE_RATE, PowerDirection.CONSUMPTION)

    assert list(resolved.keys()) == [CostType.ENERGY]
    assert _constant_cost(resolved[CostType.ENERGY][0]) == 7.0


def test_direction_shorthand_cost_type_dict_defaults_to_consumption(tmp_path: Path) -> None:
    """A CostType-keyed dict at the meter-type level implies consumption."""
    path = tmp_path / "tariff.yml"
    path.write_text(
        "supplier: Demo\n"
        "product: Short\n"
        "by_meter_type:\n"
        "  single_rate:\n"
        "    energy:\n"
        "      - start: 2025-01-01T00:00:00\n"
        "        formula:\n"
        "          constant_cost: 5.0\n"
        "    chp_certificates:\n"
        "      - start: 2025-01-01T00:00:00\n"
        "        formula:\n"
        "          constant_cost: 1.5\n",
        encoding="utf-8",
    )

    tariff = Tariff.from_yaml(path)
    resolved = tariff.resolve_cost_formulas(MeterType.SINGLE_RATE, PowerDirection.CONSUMPTION)

    assert set(resolved.keys()) == {CostType.ENERGY, CostType.CHP_CERTIFICATES}
    assert _constant_cost(resolved[CostType.ENERGY][0]) == 5.0
    assert _constant_cost(resolved[CostType.CHP_CERTIFICATES][0]) == 1.5


def test_compute_cost_series_returns_empty_when_formulas_do_not_overlap() -> None:
    formulas = [
        TimedPriceFormula(start=dt.datetime(2025, 1, 2, 0, 0), formula=PriceFormula(constant_cost=1.0)),
    ]

    out = Tariff._compute_cost_series(
        formulas=formulas,
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 1, 0),
        resolution=dt.timedelta(minutes=15),
    )

    assert list(out.columns) == ["timestamp", "value"]
    assert out.empty


def test_get_cost_raises_when_all_resolved_formulas_are_outside_range() -> None:
    tariff = Tariff(
        supplier="S",
        product="P",
        by_meter_type={
            MeterType.SINGLE_RATE: {
                PowerDirection.CONSUMPTION: {
                    CostType.ENERGY: [
                        TimedPriceFormula(start=dt.datetime(2025, 1, 2, 0, 0), formula=PriceFormula(constant_cost=1.0))
                    ],
                    CostType.CHP_CERTIFICATES: [
                        TimedPriceFormula(start=dt.datetime(2025, 1, 2, 0, 0), formula=PriceFormula(constant_cost=2.0))
                    ],
                }
            }
        },
    )

    with pytest.raises(
        ValueError, match="No formulas for meter type 'single_rate' and direction 'consumption' found in tariff"
    ):
        tariff.get_cost(
            start=dt.datetime(2025, 1, 1, 0, 0),
            end=dt.datetime(2025, 1, 1, 1, 0),
        )
