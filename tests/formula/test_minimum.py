from __future__ import annotations

import datetime as dt

import isodate
import pandas as pd
import pytest

from energy_cost.formula import IndexFormula, MaximumFormula, MinimumFormula, PeriodicFormula


def test_minimum_formula_get_values_raises_not_implemented() -> None:
    formula = MinimumFormula(
        period=isodate.parse_duration("P1M"),
        minimum=[IndexFormula(constant_cost=1.0)],
    )

    with pytest.raises(NotImplementedError):
        formula.get_values(
            start=dt.datetime(2025, 1, 1, 0, 0),
            end=dt.datetime(2025, 2, 1, 0, 0),
            output_resolution=isodate.parse_duration("P1M"),
        )


def test_minimum_formula_apply_picks_cheaper_formula() -> None:
    """When one formula is always cheaper, it should win in every period."""
    formula = MinimumFormula(
        period=isodate.parse_duration("P1M"),
        minimum=[
            IndexFormula(constant_cost=2.0),
            IndexFormula(constant_cost=5.0),
        ],
    )
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "value": [10.0, 10.0],
        }
    )

    out = formula.apply(data, resolution=isodate.parse_duration("P1M"))

    # IndexFormula(constant_cost=2.0) → 2*10=20 per month; the other → 5*10=50
    assert out["value"].tolist() == pytest.approx([20.0, 20.0])


def test_minimum_formula_apply_different_winner_per_period() -> None:
    """When consumption varies, a periodic (fixed) formula may win some months and a variable formula others."""
    formula = MinimumFormula(
        period=isodate.parse_duration("P1M"),
        minimum=[
            PeriodicFormula(period=isodate.parse_duration("P1M"), constant_cost=50.0),
            IndexFormula(constant_cost=10.0),
        ],
    )
    # Jan: 3 units  → periodic=50, index=30 → min=30 (index wins)
    # Feb: 8 units  → periodic=50, index=80 → min=50 (periodic wins)
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "value": [3.0, 8.0],
        }
    )

    out = formula.apply(data, resolution=isodate.parse_duration("P1M"))

    assert out["value"].tolist() == pytest.approx([30.0, 50.0])


def test_minimum_formula_apply_all_formulas_tie() -> None:
    """When all formulas produce the same cost, the minimum should be that cost."""
    formula = MinimumFormula(
        period=isodate.parse_duration("P1M"),
        minimum=[
            PeriodicFormula(period=isodate.parse_duration("P1M"), constant_cost=20.0),
            IndexFormula(constant_cost=10.0),
        ],
    )
    # Jan: 2 units  → periodic=20, index=20 → min=20
    # Feb: 5 units  → periodic=20, index=50 → min=20
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "value": [2.0, 5.0],
        }
    )

    out = formula.apply(data, resolution=isodate.parse_duration("P1M"))

    assert out["value"].tolist() == pytest.approx([20.0, 20.0])


def test_minimum_formula_apply_works_for_resolutions_smaller_than_period() -> None:
    """The apply method should work even if the input resolution is smaller than the minimum formula's period."""
    formula = MinimumFormula(
        period=isodate.parse_duration("PT1H"),
        minimum=[
            PeriodicFormula(period=isodate.parse_duration("PT1H"), constant_cost=30.0),
            IndexFormula(constant_cost=10.0),
        ],
    )
    # 00:00: 2 units → periodic=30, index=20 → min=20 (index wins)
    # 01:00: 5 units → periodic=15, index=50 → min=15 (periodic wins)
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 00:30", "2025-01-01 01:00"]),
            "value": [1.0, 1.0, 5.0],
        }
    )

    out = formula.apply(data, resolution=isodate.parse_duration("PT30M"))

    assert out["value"].tolist() == pytest.approx([10.0, 10.0, 15.0])


def test_minimum_formula_apply_works_for_resolutions_larger_than_period() -> None:
    """The apply method should work even if the input resolution is larger than the minimum formula's period."""
    formula = MinimumFormula(
        period=isodate.parse_duration("PT30M"),
        minimum=[
            PeriodicFormula(period=isodate.parse_duration("PT30M"), constant_cost=30.0),
            IndexFormula(constant_cost=10.0),
        ],
    )

    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 00:30"]),
            "value": [2.0, 5.0],
        }
    )

    out = formula.apply(data, resolution=isodate.parse_duration("PT1H"))

    assert out["value"].tolist()[0] == pytest.approx(50.0)  # 20 from index + 30 from periodic


def test_minimum_formula_apply_takes_first_formula_on_ties() -> None:
    """When formulas tie, the minimum should deterministically pick the first one (after sorting by formula ID) to ensure consistent results."""
    formula = MinimumFormula(
        period=isodate.parse_duration("PT1H"),
        minimum=[
            IndexFormula(constant_cost=10.0),
            PeriodicFormula(period=isodate.parse_duration("PT1H"), constant_cost=20.0),
        ],
    )
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 00:30"]),
            "value": [0.5, 1.5],
        }
    )

    out = formula.apply(data, resolution=isodate.parse_duration("PT30M"))

    # we get the 5, 15 from the index instead of the 10, 10 from the periodic formula
    assert out["value"].tolist() == pytest.approx([5.0, 15.0])

    reversed_formula = MinimumFormula(
        period=isodate.parse_duration("PT1H"),
        minimum=[
            PeriodicFormula(period=isodate.parse_duration("PT1H"), constant_cost=20.0),
            IndexFormula(constant_cost=10.0),
        ],
    )
    reversed_out = reversed_formula.apply(data, resolution=isodate.parse_duration("PT30M"))

    assert reversed_out["value"].tolist() == pytest.approx([10.0, 10.0])


def test_minimum_formula_apply_single_formula() -> None:
    """A minimum formula with a single child simply returns that formula's period sum."""
    formula = MinimumFormula(
        period=isodate.parse_duration("P1M"),
        minimum=[IndexFormula(constant_cost=3.0)],
    )
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01"]),
            "value": [4.0],
        }
    )

    out = formula.apply(data, resolution=isodate.parse_duration("P1M"))

    assert out["value"].tolist() == pytest.approx([12.0])


def test_minimum_formula_model_validate_from_dict() -> None:
    """A dict with a 'minimum' key should be coerced into a MinimumFormula via the discriminator."""
    raw = {
        "period": "P1M",
        "minimum": [
            {"constant_cost": 2.0},
            {"constant_cost": 5.0},
        ],
    }

    formula = MinimumFormula.model_validate(raw)

    assert isinstance(formula, MinimumFormula)
    assert len(formula.minimum) == 2
    assert all(isinstance(f, IndexFormula) for f in formula.minimum)


# ---------------------------------------------------------------------------
# MaximumFormula tests
# ---------------------------------------------------------------------------


def test_maximum_formula_get_values_raises_not_implemented() -> None:
    formula = MaximumFormula(
        period=isodate.parse_duration("P1M"),
        maximum=[IndexFormula(constant_cost=1.0)],
    )

    with pytest.raises(NotImplementedError):
        formula.get_values(
            start=dt.datetime(2025, 1, 1, 0, 0),
            end=dt.datetime(2025, 2, 1, 0, 0),
            output_resolution=isodate.parse_duration("P1M"),
        )


def test_maximum_formula_apply_picks_more_expensive_formula() -> None:
    """When one formula is always more expensive, it should win in every period."""
    formula = MaximumFormula(
        period=isodate.parse_duration("P1M"),
        maximum=[
            IndexFormula(constant_cost=2.0),
            IndexFormula(constant_cost=5.0),
        ],
    )
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "value": [10.0, 10.0],
        }
    )

    out = formula.apply(data, resolution=isodate.parse_duration("P1M"))

    # IndexFormula(constant_cost=5.0) → 5*10=50 per month; the other → 2*10=20
    assert out["value"].tolist() == pytest.approx([50.0, 50.0])


def test_maximum_formula_apply_different_winner_per_period() -> None:
    """When consumption varies, a periodic (fixed) formula may win some months and a variable formula others."""
    formula = MaximumFormula(
        period=isodate.parse_duration("P1M"),
        maximum=[
            PeriodicFormula(period=isodate.parse_duration("P1M"), constant_cost=50.0),
            IndexFormula(constant_cost=10.0),
        ],
    )
    # Jan: 3 units  → periodic=50, index=30 → max=50 (periodic wins)
    # Feb: 8 units  → periodic=50, index=80 → max=80 (index wins)
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01", "2025-02-01"]),
            "value": [3.0, 8.0],
        }
    )

    out = formula.apply(data, resolution=isodate.parse_duration("P1M"))

    assert out["value"].tolist() == pytest.approx([50.0, 80.0])


def test_maximum_formula_apply_all_formulas_tie() -> None:
    """When all formulas produce the same cost, the maximum should be that cost."""
    formula = MaximumFormula(
        period=isodate.parse_duration("P1M"),
        maximum=[
            PeriodicFormula(period=isodate.parse_duration("P1M"), constant_cost=20.0),
            IndexFormula(constant_cost=10.0),
        ],
    )
    # Jan: 2 units  → periodic=20, index=20 → max=20
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01"]),
            "value": [2.0],
        }
    )

    out = formula.apply(data, resolution=isodate.parse_duration("P1M"))

    assert out["value"].tolist() == pytest.approx([20.0])


def test_maximum_formula_apply_works_for_resolutions_smaller_than_period() -> None:
    """The apply method should work even if the input resolution is smaller than the maximum formula's period."""
    formula = MaximumFormula(
        period=isodate.parse_duration("PT1H"),
        maximum=[
            PeriodicFormula(period=isodate.parse_duration("PT1H"), constant_cost=30.0),
            IndexFormula(constant_cost=10.0),
        ],
    )
    # 00:00: 2 units → periodic=30, index=20 → max=30 (periodic wins)
    # 01:00: 5 units → periodic=15, index=50 → max=50 (index wins)
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 00:30", "2025-01-01 01:00"]),
            "value": [1.0, 1.0, 5.0],
        }
    )

    out = formula.apply(data, resolution=isodate.parse_duration("PT30M"))

    assert out["value"].tolist() == pytest.approx([15.0, 15.0, 50.0])


def test_maximum_formula_apply_takes_first_formula_on_ties() -> None:
    """When formulas tie, the maximum should deterministically pick the first one to ensure consistent results."""
    formula = MaximumFormula(
        period=isodate.parse_duration("PT1H"),
        maximum=[
            IndexFormula(constant_cost=10.0),
            PeriodicFormula(period=isodate.parse_duration("PT1H"), constant_cost=20.0),
        ],
    )
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00", "2025-01-01 00:30"]),
            "value": [0.5, 1.5],
        }
    )

    out = formula.apply(data, resolution=isodate.parse_duration("PT30M"))

    # index total = 0.5*10 + 1.5*10 = 20, periodic total = 20 → tie → first formula (index) wins
    # index gives 5.0 and 15.0 per slot; periodic gives 10.0 per slot
    assert out["value"].tolist() == pytest.approx([5.0, 15.0])

    reversed_formula = MaximumFormula(
        period=isodate.parse_duration("PT1H"),
        maximum=[
            PeriodicFormula(period=isodate.parse_duration("PT1H"), constant_cost=20.0),
            IndexFormula(constant_cost=10.0),
        ],
    )
    reversed_out = reversed_formula.apply(data, resolution=isodate.parse_duration("PT30M"))

    assert reversed_out["value"].tolist() == pytest.approx([10.0, 10.0])


def test_maximum_formula_apply_single_formula() -> None:
    """A maximum formula with a single child simply returns that formula's result."""
    formula = MaximumFormula(
        period=isodate.parse_duration("P1M"),
        maximum=[IndexFormula(constant_cost=3.0)],
    )
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01"]),
            "value": [4.0],
        }
    )

    out = formula.apply(data, resolution=isodate.parse_duration("P1M"))

    assert out["value"].tolist() == pytest.approx([12.0])


def test_maximum_formula_model_validate_from_dict() -> None:
    """A dict with a 'maximum' key should be coerced into a MaximumFormula via the discriminator."""
    raw = {
        "period": "P1M",
        "maximum": [
            {"constant_cost": 2.0},
            {"constant_cost": 5.0},
        ],
    }

    formula = MaximumFormula.model_validate(raw)

    assert isinstance(formula, MaximumFormula)
    assert len(formula.maximum) == 2
    assert all(isinstance(f, IndexFormula) for f in formula.maximum)
