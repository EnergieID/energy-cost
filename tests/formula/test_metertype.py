from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from energy_cost.formula import IndexFormula
from energy_cost.formula.metertype import MeterTypeFormula
from energy_cost.meter import Meter, MeterType, TimeseriesFrame


def _meter(value: float = 1.0, meter_type: MeterType = MeterType.TOU_PEAK) -> Meter:
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=2, freq="15min", tz=dt.UTC),
            "value": [value, value],
        }
    )
    return Meter(power=TimeseriesFrame(data), type=meter_type)


def test_meter_type_formula_routes_to_matching_formula() -> None:
    formula = MeterTypeFormula(
        by_meter_type={
            MeterType.TOU_PEAK: IndexFormula(constant_cost=10.0),
            MeterType.TOU_OFFPEAK: IndexFormula(constant_cost=5.0),
        }
    )
    meter = _meter(value=2.0, meter_type=MeterType.TOU_PEAK)

    out = formula.apply(meter, meter.power.start, meter.power.end, output_resolution=dt.timedelta(minutes=15))

    assert out["value"].tolist() == pytest.approx([20.0, 20.0])


def test_meter_type_formula_falls_back_to_default_when_no_match() -> None:
    formula = MeterTypeFormula(
        by_meter_type={
            MeterType.TOU_PEAK: IndexFormula(constant_cost=10.0),
            "default": IndexFormula(constant_cost=3.0),
        }
    )
    meter = _meter(value=2.0, meter_type=MeterType.TOU_OFFPEAK)

    out = formula.apply(meter, meter.power.start, meter.power.end, output_resolution=dt.timedelta(minutes=15))

    assert out["value"].tolist() == pytest.approx([6.0, 6.0])


def test_meter_type_formula_raises_when_no_match_and_no_default() -> None:
    """When meter type has no configured formula and no 'default' key, ValueError is raised (metertype.py line 35)."""
    formula = MeterTypeFormula(
        by_meter_type={
            MeterType.TOU_PEAK: IndexFormula(constant_cost=10.0),
        }
    )
    meter = _meter(value=1.0, meter_type=MeterType.TOU_OFFPEAK)

    with pytest.raises(ValueError, match="No formula configured for meter type"):
        formula.apply(meter, meter.power.start, meter.power.end, output_resolution=dt.timedelta(minutes=15))
