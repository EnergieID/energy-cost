from __future__ import annotations

import datetime as dt

import isodate
import pandas as pd

from energy_cost.formula import PeriodicFormula
from energy_cost.fractional_periods import Period


def test_periodic_formula_get_cost_for_interval_is_prorated() -> None:
    formula = PeriodicFormula(period=Period.DAILY, constant_cost=24.0)

    out = formula.get_cost_for_interval(
        dt.datetime(2025, 1, 1, 0, 0),
        dt.datetime(2025, 1, 1, 1, 0),
    )

    assert out == 1.0


def test_periodic_formula_get_values_returns_cost_per_interval() -> None:
    formula = PeriodicFormula(period=Period.DAILY, constant_cost=24.0)

    out = formula.get_values(
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 1, 0),
        resolution=dt.timedelta(minutes=15),
    )

    assert out["value"].tolist() == [0.25, 0.25, 0.25, 0.25]


def test_periodic_formula_apply_uses_input_resolution() -> None:
    formula = PeriodicFormula(period=Period.MONTHLY, constant_cost=100.0)
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00:00", "2025-02-01 00:00:00"]),
            "value": [1.0, 1.0],
        }
    )

    out = formula.apply(data, resolution=isodate.parse_duration("P1M"))

    assert out["value"].tolist() == [100.0, 100.0]
