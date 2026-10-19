from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from pydantic import TypeAdapter, ValidationError

from energy_cost import Tariff
from energy_cost.formula import Formula, IndexAdder, IndexFormula
from energy_cost.index import DataFrameIndex, Index
from energy_cost.meter import Meter, TimeseriesFrame


def test_index_formula_merges_with_timezone_conversion():
    # Create index data in Europe/Brussels timezone using ZoneInfo
    z = ZoneInfo("Europe/Brussels")
    timestamps_brussels = pd.date_range(dt.datetime(2025, 1, 1, 1, 0, tzinfo=z), periods=3, freq="15min")
    index_df = pd.DataFrame({"timestamp": timestamps_brussels, "value": [1.0, 2.0, 3.0]})
    Index.register("tztest", DataFrameIndex(index_df))
    # Request with UTC datetimes
    start_utc = dt.datetime(2025, 1, 1, 0, 0, tzinfo=dt.UTC)
    end_utc = dt.datetime(2025, 1, 1, 0, 45, tzinfo=dt.UTC)
    formula = IndexFormula(constant_cost=1.0, variable_costs=[IndexAdder(index="tztest", scalar=1.0)])
    out = formula.get_values(start=start_utc, end=end_utc, output_resolution=dt.timedelta(minutes=15))
    # All timestamps should be UTC
    assert all(ts.tzinfo == dt.UTC for ts in out["timestamp"])
    # Values should be correct (constant + index values)
    assert out["value"].tolist() == [2.0, 3.0, 4.0]


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
        output_resolution=dt.timedelta(minutes=15),
    )

    assert out["value"].tolist() == [1.0, 2.0]
    assert base["value"].tolist() == [10.0, 20.0]


def test_index_formula_constant_only() -> None:
    formula = IndexFormula(constant_cost=1.5)

    out = formula.get_values(
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 0, 45),
        output_resolution=dt.timedelta(minutes=15),
    )

    assert out["timestamp"].tolist() == list(pd.date_range("2025-01-01", periods=3, freq="15min", tz=dt.UTC))
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
        output_resolution=dt.timedelta(minutes=15),
    )

    assert out["value"].tolist() == [2.5, 3.5, 6.5]


def test_index_formula_apply_multiplies_input_dataframe() -> None:
    formula = IndexFormula(constant_cost=2.0)
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="15min", tz=dt.UTC),
            "value": [1.0, 2.0, 3.0],
        }
    )
    meter = Meter(measurements=TimeseriesFrame(data))

    out = formula.apply(
        meter, meter.measurements.start, meter.measurements.end, output_resolution=dt.timedelta(minutes=15)
    )

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
        output_resolution=dt.timedelta(minutes=15),
    )

    assert out["value"].isna().tolist() == [True, True, False, False, True, True]


def test_index_formula_with_only_constant_value_returns_valid_for_all_timestamps() -> None:
    formula = IndexFormula(constant_cost=5.0)

    out = formula.get_values(
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 1, 0),
        output_resolution=dt.timedelta(minutes=15),
    )

    assert out["value"].tolist() == [5.0] * 4


def test_capacity_based_formula_raises_when_meter_has_no_capacity() -> None:
    """FormulaBase.apply raises ValueError when capacity_based=True but meter.capacity is None (base.py line 40)."""
    formula = IndexFormula(constant_cost=1.0, capacity_based=True)
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=2, freq="15min", tz=dt.UTC),
            "value": [1.0, 1.0],
        }
    )
    meter = Meter(measurements=TimeseriesFrame(data))  # no capacity

    with pytest.raises(ValueError, match="Capacity is required"):
        formula.apply(
            meter, meter.measurements.start, meter.measurements.end, output_resolution=dt.timedelta(minutes=15)
        )


def _two_quarter_index() -> DataFrameIndex:
    return DataFrameIndex(
        pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=2, freq="15min"), "value": 1.0})
    )


def test_index_adder_accepts_registered_index() -> None:
    Index.register("spot", _two_quarter_index())

    assert IndexAdder(index="spot", scalar=1.0).index == "spot"


def test_index_adder_serializes_only_the_index_name() -> None:
    Index.register("spot", _two_quarter_index())

    assert IndexAdder(index="spot", scalar=1.0).model_dump() == {"index": "spot", "scalar": 1.0}


def test_index_adder_rejects_unknown_index() -> None:
    with pytest.raises(ValidationError, match="unknown index 'spot'"):
        IndexAdder(index="spot", scalar=1.0)


@pytest.mark.parametrize(
    "formula",
    [
        pytest.param({"variable_costs": [{"index": "missing", "scalar": 1.0}]}, id="index"),
        pytest.param(
            {
                "bands": [
                    {"up_to": 1.0, "formula": {"constant_cost": 1.0}},
                    {"formula": {"variable_costs": [{"index": "missing", "scalar": 1.0}]}},
                ]
            },
            id="tiered",
        ),
        pytest.param(
            {
                "schedule": [
                    {
                        "when": [{"start": "06:00:00"}],
                        "formula": {"variable_costs": [{"index": "missing", "scalar": 1.0}]},
                    }
                ]
            },
            id="scheduled",
        ),
        pytest.param(
            {
                "period": "P1M",
                "minimum": [{"constant_cost": 1.0}, {"variable_costs": [{"index": "missing", "scalar": 1.0}]}],
            },
            id="minimum",
        ),
        pytest.param(
            {
                "period": "P1M",
                "maximum": [{"constant_cost": 1.0}, {"variable_costs": [{"index": "missing", "scalar": 1.0}]}],
            },
            id="maximum",
        ),
        pytest.param(
            {"by_meter_type": {"default": {"variable_costs": [{"index": "missing", "scalar": 1.0}]}}},
            id="meter_type",
        ),
    ],
)
def test_nested_formula_rejects_unknown_index(formula: dict) -> None:
    with pytest.raises(ValidationError, match="unknown index 'missing'"):
        TypeAdapter(Formula).validate_python(formula)


def test_tariff_rejects_unknown_index_in_later_version() -> None:
    Index.register("spot", _two_quarter_index())
    adder = {"variable_costs": [{"index": "spot", "scalar": 1.0}]}

    with pytest.raises(ValidationError, match="unknown index 'missing'"):
        Tariff.model_validate(
            [
                {"start": "2025-01-01T00:00:00+01:00", "consumption": adder},
                {
                    "start": "2026-01-01T00:00:00+01:00",
                    "injection": {"variable_costs": [{"index": "missing", "scalar": 1.0}]},
                },
            ]
        )
