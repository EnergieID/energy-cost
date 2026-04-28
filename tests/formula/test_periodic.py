from __future__ import annotations

import datetime as dt

import isodate
import pandas as pd
import pytest

from energy_cost.formula import PeriodicFormula


def test_periodic_formula_get_values_raises_not_implemented() -> None:
    formula = PeriodicFormula(period=dt.timedelta(days=1), constant_cost=24.0)

    with pytest.raises(NotImplementedError):
        formula.get_values(
            start=dt.datetime(2025, 1, 1, 0, 0),
            end=dt.datetime(2025, 1, 1, 1, 0),
            resolution=dt.timedelta(minutes=15),
        )


def test_periodic_formula_apply_uses_input_resolution() -> None:
    formula = PeriodicFormula(period=isodate.parse_duration("P1M"), constant_cost=100.0)
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00:00", "2025-02-01 00:00:00"]),
            "value": [1.0, 1.0],
        }
    )

    out = formula.apply(data, resolution=isodate.parse_duration("P1M"))

    assert out["value"].tolist() == [100.0, 100.0]


def test_periodic_formula_apply_distributes_cost_over_fine_slots() -> None:
    """A daily cost of 24 distributed across 24 hourly slots gives 1.0 per slot."""
    formula = PeriodicFormula(period=dt.timedelta(days=1), constant_cost=24.0)
    timestamps = pd.date_range("2025-01-01", periods=24, freq="h")
    data = pd.DataFrame({"timestamp": timestamps})

    out = formula.apply(data, resolution=dt.timedelta(hours=1))

    assert out["value"].sum() == pytest.approx(24.0)
    assert out["value"].iloc[0] == pytest.approx(1.0)
