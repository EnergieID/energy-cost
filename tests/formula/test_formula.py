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
period: monthly
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
