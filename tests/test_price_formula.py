from __future__ import annotations

import datetime as dt

import pandas as pd

from energy_cost.index import Index
from energy_cost.price_formula import IndexAdder, PriceFormula


class DataFrameIndex(Index):
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_values(self, start: dt.datetime, end: dt.datetime, resolution: dt.timedelta) -> pd.DataFrame:
        return self.df


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
    # Ensure original frame is untouched.
    assert base["value"].tolist() == [10.0, 20.0]


def test_price_formula_constant_only() -> None:
    formula = PriceFormula(constant_cost=1.5)

    out = formula.get_values(
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 0, 45),
        resolution=dt.timedelta(minutes=15),
    )

    assert out["timestamp"].tolist() == list(pd.date_range("2025-01-01", periods=3, freq="15min"))
    assert out["value"].tolist() == [1.5, 1.5, 1.5]


def test_price_formula_adds_multiple_variable_costs_and_fills_missing() -> None:
    timestamps = pd.date_range("2025-01-01", periods=3, freq="15min")
    index_a_df = pd.DataFrame({"timestamp": timestamps, "value": [1.0, 2.0, 3.0]})
    # Missing middle timestamp on purpose to validate fillna(0)
    index_b_df = pd.DataFrame({"timestamp": [timestamps[0], timestamps[2]], "value": [10.0, 30.0]})

    Index.register("a", DataFrameIndex(index_a_df))
    Index.register("b", DataFrameIndex(index_b_df))

    formula = PriceFormula(
        constant_cost=0.5,
        variable_costs=[IndexAdder(index="a", scalar=1.0), IndexAdder(index="b", scalar=0.1)],
    )

    out = formula.get_values(
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 0, 45),
        resolution=dt.timedelta(minutes=15),
    )

    # 0.5 + a + 0.1*b where middle b is missing -> +0
    assert out["value"].tolist() == [2.5, 2.5, 6.5]
