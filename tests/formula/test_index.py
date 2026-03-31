from __future__ import annotations

import datetime as dt

import pandas as pd

from energy_cost.formula import IndexAdder, IndexFormula
from energy_cost.index import DataFrameIndex, Index


def test_index_adder_multiplies_index_values() -> None:
    base = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=2, freq="15min"),
            "value": [10.0, 20.0],
        }
    )
    Index.register("idx", DataFrameIndex(base))
    adder = IndexAdder(index="idx", scalar=0.1)

    out = adder.get_values(
        start=dt.datetime(2025, 1, 1),
        end=dt.datetime(2025, 1, 1, 0, 30),
        resolution=dt.timedelta(minutes=15),
    )

    assert out["value"].tolist() == [1.0, 2.0]
    assert base["value"].tolist() == [10.0, 20.0]


def test_index_formula_constant_only() -> None:
    formula = IndexFormula(constant_cost=1.5)

    out = formula.get_values(
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 0, 45),
        resolution=dt.timedelta(minutes=15),
    )

    assert out["timestamp"].tolist() == list(pd.date_range("2025-01-01", periods=3, freq="15min"))
    assert out["value"].tolist() == [1.5, 1.5, 1.5]


def test_index_formula_adds_multiple_variable_costs() -> None:
    timestamps = pd.date_range("2025-01-01", periods=3, freq="15min")
    index_a_df = pd.DataFrame({"timestamp": timestamps, "value": [1.0, 2.0, 3.0]})
    index_b_df = pd.DataFrame({"timestamp": [timestamps[0], timestamps[2]], "value": [10.0, 30.0]})

    Index.register("a", DataFrameIndex(index_a_df))
    Index.register("b", DataFrameIndex(index_b_df))

    formula = IndexFormula(
        constant_cost=0.5,
        variable_costs=[IndexAdder(index="a", scalar=1.0), IndexAdder(index="b", scalar=0.1)],
    )

    out = formula.get_values(
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 0, 45),
        resolution=dt.timedelta(minutes=15),
    )

    assert out["value"].tolist() == [2.5, 3.5, 6.5]


def test_index_formula_apply_multiplies_input_dataframe() -> None:
    formula = IndexFormula(constant_cost=2.0)
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="15min"),
            "value": [1.0, 2.0, 3.0],
        }
    )

    out = formula.apply(data)

    assert out["value"].tolist() == [2.0, 4.0, 6.0]


def test_index_formula_returns_nan_outside_index_range() -> None:
    timestamps = pd.date_range("2025-01-01", periods=3, freq="15min")
    index_a_df = pd.DataFrame({"timestamp": timestamps, "value": [1.0, 2.0, 3.0]})
    index_b_df = pd.DataFrame({"timestamp": timestamps[:-1], "value": [10.0, 30.0]})

    Index.register("a-range", DataFrameIndex(index_a_df))
    Index.register("b-range", DataFrameIndex(index_b_df))

    formula = IndexFormula(
        constant_cost=0.5,
        variable_costs=[IndexAdder(index="a-range", scalar=1.0), IndexAdder(index="b-range", scalar=0.1)],
    )

    out = formula.get_values(
        start=dt.datetime(2024, 12, 31, 23, 30),
        end=dt.datetime(2025, 1, 1, 1, 0),
        resolution=dt.timedelta(minutes=15),
    )

    assert out["value"].isna().tolist() == [True, True, False, False, True, True]
