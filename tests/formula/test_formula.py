import pandas as pd
import pytest
import yaml

from energy_cost.formula import Formula, IndexFormula, PeriodicFormula, ScheduledFormulas, TieredFormula


def test_tiered_formulas_are_correctly_coerced_by_model_validate() -> None:
    simple_yaml = """
bands:
  - up_to: 10.0
    formula:
      constant_cost: 100.0
  - formula:
      constant_cost: 100.0
"""
    formula_dict = yaml.safe_load(simple_yaml)
    formula = Formula.model_validate(formula_dict)

    assert isinstance(formula, TieredFormula)
    assert len(formula.bands) == 2

    explicit_yaml = """
kind: tiered
bands: []
"""
    formula_dict = yaml.safe_load(explicit_yaml)
    formula = Formula.model_validate(formula_dict)

    assert isinstance(formula, TieredFormula)
    assert len(formula.bands) == 0


def test_periodic_formulas_are_correctly_coerced_by_model_validate() -> None:
    simple_yaml = """
period: P1M
constant_cost: 100.0
"""
    formula_dict = yaml.safe_load(simple_yaml)
    formula = Formula.model_validate(formula_dict)

    assert isinstance(formula, PeriodicFormula)
    assert formula.constant_cost == 100.0


def test_index_formulas_are_correctly_coerced_by_model_validate() -> None:
    simple_yaml = """
constant_cost: 100.0
"""
    formula_dict = yaml.safe_load(simple_yaml)
    formula = Formula.model_validate(formula_dict)

    assert isinstance(formula, IndexFormula)


def test_scheduled_formulas_are_correctly_coerced_by_model_validate() -> None:
    simple_yaml = """
schedule:
    - when:
        - days: [wednesday]
          start: 09:00:00
          end: 17:00:00
      formula:
        constant_cost: 5.0
    - formula:
        constant_cost: 2.0
"""
    formula_dict = yaml.safe_load(simple_yaml)
    formula = Formula.model_validate(formula_dict)

    assert isinstance(formula, ScheduledFormulas)
    assert len(formula.schedule) == 2


def test_invalid_formulas_raise_on_model_validate() -> None:
    invalid_yaml = """
123
"""
    invalid_formula = yaml.safe_load(invalid_yaml)

    pytest.raises(ValueError, Formula.model_validate, invalid_formula)


def test_calling_apply_on_an_empty_dataframe_returns_empty_dataframe() -> None:
    formula = IndexFormula(constant_cost=100.0)
    empty_df = pd.DataFrame(columns=["timestamp", "value"])
    result = formula.apply(empty_df)
    assert result.empty


def test_apply_produces_correct_values_when_data_has_non_default_integer_index() -> None:
    """Regression: Formula.apply must not NaN-out values when data has a non-zero integer index.

    This happens when data is a boolean-filtered slice of a larger DataFrame, leaving the
    original positional index intact (e.g. 35136, 35137, ...).  The merge inside apply()
    produces a zero-based result index; assigning back via result["value"] = merged[...]
    then misaligns by index and fills with NaN unless the index is reset first.
    """
    formula = IndexFormula(constant_cost=90.0)

    # Build a large frame and slice a window from it — this leaves a non-zero-based integer index
    timestamps = pd.date_range("2025-01-01T00:00:00+01:00", periods=16, freq="15min")
    large = pd.DataFrame({"timestamp": timestamps, "value": 0.0002})
    data = large.iloc[8:].copy()  # index starts at 8, not 0

    assert data.index[0] == 8, "Pre-condition: index must NOT start at 0"

    result = formula.apply(data)

    assert result["value"].notna().all(), "Formula.apply must not produce NaN values for non-zero-based indexes"
    expected = 0.0002 * 90.0
    assert result["value"].tolist() == pytest.approx([expected] * len(data))
