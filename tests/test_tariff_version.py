from __future__ import annotations

import datetime as dt

import isodate
import pandas as pd
import pytest

from energy_cost.formula import Formula, IndexFormula, PeriodicFormula, ScheduledFormulas
from energy_cost.meter import CostGroup, Meter, TimeseriesFrame
from energy_cost.resolution import Resolution
from energy_cost.tariff_version import TariffVersion, _coerce_named_formulas


def _constant_cost(formula: Formula) -> float:
    assert isinstance(formula, IndexFormula)
    return formula.constant_cost


def test_get_values_returns_one_column_per_cost_type() -> None:
    segment = TariffVersion(
        start=dt.datetime(2025, 1, 1, 0, 0),
        consumption={
            "energy": IndexFormula(constant_cost=10.0),
            "chp_certificates": IndexFormula(constant_cost=2.0),
            "renewable_certificates": IndexFormula(constant_cost=3.0),
        },
    )

    out = segment.get_values(
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 1, 0),
        output_resolution=dt.timedelta(minutes=15),
        cost_group=CostGroup.CONSUMPTION,
    )

    assert out is not None
    assert set(out.columns) == {"timestamp", "energy", "chp_certificates", "renewable_certificates", "total"}
    assert out["energy"].tolist() == [10.0, 10.0, 10.0, 10.0]
    assert out["chp_certificates"].tolist() == [2.0, 2.0, 2.0, 2.0]
    assert out["renewable_certificates"].tolist() == [3.0, 3.0, 3.0, 3.0]
    assert out["total"].tolist() == [15.0, 15.0, 15.0, 15.0]


def test_get_values_returns_injection_cost_types() -> None:
    segment = TariffVersion(
        start=dt.datetime(2025, 1, 1, 0, 0),
        injection={"energy": IndexFormula(constant_cost=-5.0)},
    )

    out = segment.get_values(
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 1, 0),
        output_resolution=dt.timedelta(minutes=15),
        cost_group=CostGroup.INJECTION,
    )

    assert out is not None
    assert out["energy"].tolist() == [-5.0, -5.0, -5.0, -5.0]


def test_fixed_costs_returns_prorated_cost_for_each_entry() -> None:
    segment = TariffVersion(
        start=dt.datetime(2025, 1, 1, 0, 0),
        fixed={
            "admin": PeriodicFormula(period=isodate.parse_duration("P1D"), constant_cost=24.0),
            "billing": PeriodicFormula(period=isodate.parse_duration("P1D"), constant_cost=12.0),
        },
    )
    ts = pd.date_range("2025-01-01", "2025-01-02", freq="h", tz=dt.UTC, inclusive="left")
    meter = Meter(power=TimeseriesFrame(pd.DataFrame({"timestamp": ts, "value": 1.0})))
    start = dt.datetime(2025, 1, 1, 0, 0, tzinfo=dt.UTC)
    end = dt.datetime(2025, 1, 2, 0, 0, tzinfo=dt.UTC)

    result = segment.apply(
        consumption=meter,
        injection=None,
        start=start,
        end=end,
        output_resolution=dt.timedelta(hours=1),
    )

    assert result is not None
    assert result[(CostGroup.FIXED, "admin")].sum() == pytest.approx(24.0)
    assert result[(CostGroup.FIXED, "billing")].sum() == pytest.approx(12.0)


def test_model_validation_treats_bare_consumption_formula_as_total() -> None:
    segment = TariffVersion.model_validate(
        {
            "start": "2025-01-01T00:00:00",
            "consumption": {"constant_cost": 1.0},
        }
    )

    assert _constant_cost(segment.consumption["total"]) == 1.0


def test_model_validation_treats_cost_type_map_as_named_formulas() -> None:
    segment = TariffVersion.model_validate(
        {
            "start": "2025-01-01T00:00:00",
            "consumption": {
                "energy": {"constant_cost": 1.0},
                "chp_certificates": {"constant_cost": 2.0},
            },
        }
    )

    assert _constant_cost(segment.consumption["energy"]) == 1.0
    assert _constant_cost(segment.consumption["chp_certificates"]) == 2.0


def test_model_validation_treats_scheduled_formula_dict_as_total() -> None:
    segment = TariffVersion.model_validate(
        {
            "start": "2025-01-01T00:00:00+01:00",
            "consumption": {
                "kind": "scheduled",
                "schedule": [
                    {
                        "when": [{"days": ["monday"], "start": "06:00:00", "end": "10:00:00"}],
                        "formula": {"constant_cost": 300.0},
                    },
                    {"formula": {"constant_cost": 100.0}},
                ],
            },
        }
    )

    formula = segment.consumption["total"]
    assert isinstance(formula, ScheduledFormulas)

    out = formula.get_values(
        dt.datetime.fromisoformat("2025-01-06T05:00:00+01:00"),
        dt.datetime.fromisoformat("2025-01-06T07:00:00+01:00"),
        dt.timedelta(hours=1),
        timezone=dt.timezone(dt.timedelta(hours=1)),
    )
    assert out["value"].tolist() == [100.0, 300.0]


def test_coerce_named_formulas_wraps_bare_formula_as_total() -> None:
    bare_formula = {"constant_cost": 1.0}

    result = _coerce_named_formulas(bare_formula)

    assert list(result.keys()) == ["total"]
    assert result["total"] == IndexFormula(constant_cost=1.0)


def test_get_values_returns_none_when_no_formulas_configured() -> None:
    segment = TariffVersion(
        start=dt.datetime(2025, 1, 1, 0, 0),
        injection={"energy": IndexFormula(constant_cost=-5.0)},
    )

    result = segment.get_values(
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 1, 0),
        output_resolution=dt.timedelta(minutes=15),
        cost_group=CostGroup.CONSUMPTION,
    )
    assert result is None


def test_get_values_returns_none_when_formulas_return_empty_series() -> None:
    class EmptyIndexFormula(IndexFormula):
        def get_values(
            self, start: dt.datetime, end: dt.datetime, output_resolution: Resolution, timezone: dt.tzinfo = dt.UTC
        ) -> pd.DataFrame:
            return pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns]"), "value": pd.Series(dtype=float)})

    segment = TariffVersion(
        start=dt.datetime(2025, 1, 1, 0, 0),
        consumption={"energy": EmptyIndexFormula()},
    )

    result = segment.get_values(
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 1, 0),
        output_resolution=dt.timedelta(minutes=15),
        cost_group=CostGroup.CONSUMPTION,
    )
    assert result is None


def test_apply_returns_none_when_data_is_outside_start_end() -> None:
    """Data that falls entirely outside [start, end) becomes empty after slicing -> all NaN energy values."""
    segment = TariffVersion(
        start=dt.datetime(2025, 1, 1, 0, 0),
        consumption={"energy": IndexFormula(constant_cost=10.0)},
    )

    ts = pd.date_range("2025-03-01", periods=4, freq="15min", tz=dt.UTC)
    meter = Meter(power=TimeseriesFrame(pd.DataFrame({"timestamp": ts, "value": [1.0] * 4})))

    result = segment.apply(
        consumption=meter,
        injection=None,
        start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
        end=dt.datetime(2025, 2, 1, tzinfo=dt.UTC),
        output_resolution=dt.timedelta(minutes=15),
    )

    # All energy values are NaN since data is outside the billed range.
    assert result is None or result[(CostGroup.CONSUMPTION, "energy")].isna().all()


def test_apply_returns_none_when_no_formulas_match_cost_group() -> None:
    segment = TariffVersion(
        start=dt.datetime(2025, 1, 1, 0, 0),
        injection={"energy": IndexFormula(constant_cost=-5.0)},
    )

    ts = pd.date_range("2025-01-01", periods=1, freq="h", tz=dt.UTC)
    meter = Meter(power=TimeseriesFrame(pd.DataFrame({"timestamp": ts, "value": [10.0]})))

    result = segment.apply(
        consumption=meter,
        injection=None,
        start=dt.datetime(2025, 1, 1),
        end=dt.datetime(2025, 2, 1),
        output_resolution=dt.timedelta(hours=1),
    )

    assert result is None


def test_fixed_costs_returns_none_when_output_range_is_empty() -> None:
    """When end < start the output timestamp grid is empty and the result is None."""
    segment = TariffVersion(
        start=dt.datetime(2025, 1, 1, 0, 0),
        fixed={"admin": PeriodicFormula(period=isodate.parse_duration("P1D"), constant_cost=24.0)},
    )
    ts = pd.date_range("2025-01-01", periods=2, freq="h", tz=dt.UTC)
    meter = Meter(power=TimeseriesFrame(pd.DataFrame({"timestamp": ts, "value": 1.0})))

    result = segment.apply(
        consumption=meter,
        injection=None,
        start=dt.datetime(2025, 1, 2, 0, 0, tzinfo=dt.UTC),
        end=dt.datetime(2025, 1, 1, 0, 0, tzinfo=dt.UTC),  # end before start -> empty grid
        output_resolution=dt.timedelta(hours=1),
    )

    assert result is None
