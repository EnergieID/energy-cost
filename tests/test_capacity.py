from __future__ import annotations

import datetime as dt

import isodate
import pandas as pd
import pytest

from energy_cost.capacity import CapacityRule
from energy_cost.meter import Meter, TimeseriesFrame


def _power_meter(values: list[float], freq: str = "15min") -> Meter:
    timestamps = pd.date_range("2025-01-01", periods=len(values), freq=freq, tz=dt.UTC)
    return Meter(power=TimeseriesFrame(pd.DataFrame({"timestamp": timestamps, "value": values})))


def test_capacity_rule_uses_existing_capacity_data() -> None:
    """When the meter already has capacity data, it is used directly (line 24: capacity branch)."""
    cap_rule = CapacityRule(
        measurement_period=isodate.parse_duration("PT15M"),
        billing_period=isodate.parse_duration("P1M"),
    )
    # Build a meter that already has capacity set (two monthly rows).
    cap_df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01", "2025-02-01"], utc=True),
            "value": [3.0, 5.0],
        }
    )
    power_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=4, freq="15min", tz=dt.UTC),
            "value": [1.0] * 4,
        }
    )
    meter = Meter(
        power=TimeseriesFrame(power_df),
        capacity=TimeseriesFrame(cap_df, resolution=isodate.parse_duration("P1M")),
    )

    result = cap_rule.apply(meter)

    # Capacity already monthly → billing_period (P1M) resampling gives max per month.
    assert result.capacity is not None
    assert result.capacity["value"].tolist() == pytest.approx([3.0, 5.0])


def test_capacity_rule_raises_when_power_resolution_not_divisor_of_measurement_period() -> None:
    """Power data resolution must evenly divide measurement_period (line 34)."""
    cap_rule = CapacityRule(
        measurement_period=isodate.parse_duration("PT10M"),
        billing_period=isodate.parse_duration("P1M"),
    )
    meter = _power_meter([1.0, 2.0, 3.0, 4.0], freq="15min")  # 15-min data

    with pytest.raises(ValueError, match="divisor of measurement period"):
        cap_rule.apply(meter)


def test_capacity_rule_raises_when_capacity_resolution_not_divisor_of_billing_period() -> None:
    """Capacity data resolution must evenly divide billing_period (line 43)."""
    cap_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=4, freq="30min", tz=dt.UTC),
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    power_df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=4, freq="30min", tz=dt.UTC),
            "value": [1.0] * 4,
        }
    )
    cap_rule2 = CapacityRule(
        measurement_period=isodate.parse_duration("PT20M"),
        billing_period=isodate.parse_duration("PT40M"),  # 40 min; 30 min is not a divisor
    )
    meter = Meter(
        power=TimeseriesFrame(power_df),
        capacity=TimeseriesFrame(cap_df, resolution=isodate.parse_duration("PT30M")),
    )

    with pytest.raises(ValueError, match="divisor of billing period"):
        cap_rule2.apply(meter)
