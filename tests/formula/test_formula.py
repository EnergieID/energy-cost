import datetime as dt

import pandas as pd
import pytest
import yaml
from pydantic import TypeAdapter, ValidationError

from energy_cost.formula import Formula, IndexFormula, PeriodicFormula, ScheduledFormulas, TieredFormula
from energy_cost.meter import Meter, TimeseriesFrame

_formula_adapter = TypeAdapter(Formula)


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
    formula = _formula_adapter.validate_python(formula_dict)

    assert isinstance(formula, TieredFormula)
    assert len(formula.bands) == 2

    explicit_yaml = """
kind: tiered
bands: []
"""
    formula_dict = yaml.safe_load(explicit_yaml)
    formula = _formula_adapter.validate_python(formula_dict)

    assert isinstance(formula, TieredFormula)
    assert len(formula.bands) == 0


def test_periodic_formulas_are_correctly_coerced_by_model_validate() -> None:
    simple_yaml = """
period: P1M
constant_cost: 100.0
"""
    formula_dict = yaml.safe_load(simple_yaml)
    formula = _formula_adapter.validate_python(formula_dict)

    assert isinstance(formula, PeriodicFormula)
    assert formula.constant_cost == 100.0


def test_index_formulas_are_correctly_coerced_by_model_validate() -> None:
    simple_yaml = """
constant_cost: 100.0
"""
    formula_dict = yaml.safe_load(simple_yaml)
    formula = _formula_adapter.validate_python(formula_dict)

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
    formula = _formula_adapter.validate_python(formula_dict)

    assert isinstance(formula, ScheduledFormulas)
    assert len(formula.schedule) == 2


def test_invalid_formulas_raise_on_model_validate() -> None:
    invalid_yaml = """
123
"""
    invalid_formula = yaml.safe_load(invalid_yaml)

    pytest.raises(ValidationError, _formula_adapter.validate_python, invalid_formula)


def test_formula_json_schema_generates_without_recursion() -> None:
    """Regression: generating JSON schema for Formula must not recurse infinitely.

    TierBand.formula and ScheduledFormula.formula are both typed as Formula,
    which previously caused infinite recursion when generating the OpenAPI schema.
    """
    schema = _formula_adapter.json_schema()

    # Should be a oneOf/anyOf with refs for each of the four concrete subtypes
    assert any(k in schema for k in ("oneOf", "anyOf", "$defs"))


def test_formula_json_schema_covers_all_subtypes() -> None:
    from pydantic import TypeAdapter

    schema = TypeAdapter(Formula).json_schema(mode="serialization")
    schema_str = str(schema)

    for subtype in ("IndexFormula", "PeriodicFormula", "ScheduledFormulas", "TieredFormula"):
        assert subtype in schema_str, f"{subtype} missing from Formula JSON schema"


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

    meter = Meter(power=TimeseriesFrame(data))
    result = formula.apply(meter, meter.power.start, meter.power.end, output_resolution=dt.timedelta(minutes=15))

    assert result["value"].notna().all(), "Formula.apply must not produce NaN values for non-zero-based indexes"
    expected = 0.0002 * 90.0
    assert result["value"].tolist() == pytest.approx([expected] * len(data))
