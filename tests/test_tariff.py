from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from energy_cost.price_formula import PriceFormula
from energy_cost.tariff import MeterType, Tariff, TimedPriceFormula


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
        """
        supplier: Demo
        product: Dynamic
        by_meter_type:
          single_rate:
            - start: 2025-01-01T00:00:00
              formula:
                constant_cost: 1.25
                variable_costs: []
        """,
        encoding="utf-8",
    )

    tariff = Tariff.from_yaml(path)

    assert tariff.supplier == "Demo"
    assert tariff.product == "Dynamic"
    assert list(tariff.by_meter_type.keys()) == [MeterType.SINGLE_RATE]


def test_get_formulas_returns_only_overlapping_items() -> None:
    formulas = [
        TimedPriceFormula(start=dt.datetime(2025, 1, 1, 0, 0), formula=PriceFormula(constant_cost=1.0)),
        TimedPriceFormula(start=dt.datetime(2025, 1, 1, 1, 0), formula=PriceFormula(constant_cost=2.0)),
        TimedPriceFormula(start=dt.datetime(2025, 1, 1, 2, 0), formula=PriceFormula(constant_cost=3.0)),
    ]
    tariff = Tariff(
        supplier="S",
        product="P",
        by_meter_type={MeterType.SINGLE_RATE: formulas},
    )

    out = tariff.get_formulas(
        meter_type=MeterType.SINGLE_RATE,
        start=dt.datetime(2025, 1, 1, 1, 30),
        end=dt.datetime(2025, 1, 1, 2, 30),
    )

    assert [x.formula.constant_cost for x in out] == [2.0, 3.0]


def test_get_cost_concatenates_formula_segments() -> None:
    formulas = [
        TimedPriceFormula(start=dt.datetime(2025, 1, 1, 0, 0), formula=PriceFormula(constant_cost=1.0)),
        TimedPriceFormula(start=dt.datetime(2025, 1, 1, 0, 30), formula=PriceFormula(constant_cost=2.0)),
    ]
    tariff = Tariff(
        supplier="S",
        product="P",
        by_meter_type={MeterType.SINGLE_RATE: formulas},
    )

    out = tariff.get_cost(
        meter_type=MeterType.SINGLE_RATE,
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 1, 0),
        resolution=dt.timedelta(minutes=15),
    )

    assert out["timestamp"].tolist() == list(pd.date_range("2025-01-01", periods=4, freq="15min"))
    assert out["value"].tolist() == [1.0, 1.0, 2.0, 2.0]


def test_get_cost_raises_when_meter_type_missing() -> None:
    tariff = Tariff(supplier="S", product="P")

    with pytest.raises(ValueError, match="No meters of type single_rate found in tariff"):
        tariff.get_cost(
            meter_type=MeterType.SINGLE_RATE,
            start=dt.datetime(2025, 1, 1, 0, 0),
            end=dt.datetime(2025, 1, 1, 1, 0),
            resolution=dt.timedelta(minutes=15),
        )
