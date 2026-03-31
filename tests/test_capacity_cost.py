from __future__ import annotations

import datetime as dt

import isodate
import pandas as pd

from energy_cost.capacity_cost import CapacityComponent
from energy_cost.formula import (
    DayOfWeek,
    IndexFormula,
    PeriodicFormula,
    ScheduledFormula,
    ScheduledFormulas,
    TierBand,
    TieredFormula,
    WhenClause,
)
from energy_cost.fractional_periods import Period
from energy_cost.tariff import Tariff
from energy_cost.tariff_version import TariffVersion


def test_capacity_component_applies_index_formula_to_billing_values() -> None:
    component = CapacityComponent(
        measurement_period=isodate.parse_duration("P1M"),
        billing_period=isodate.parse_duration("P1M"),
        formula=IndexFormula(constant_cost=10.0),
    )
    capacity_data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00:00", "2025-02-01 00:00:00"]),
            "value": [5.0, 7.0],
        }
    )

    out = component.apply(capacity_data)

    assert out["timestamp"].tolist() == list(pd.to_datetime(["2025-01-01 00:00:00", "2025-02-01 00:00:00"]))
    assert out["value"].tolist() == [50.0, 70.0]


def test_capacity_component_applies_banded_tiered_formula() -> None:
    component = CapacityComponent(
        measurement_period=isodate.parse_duration("P1M"),
        billing_period=isodate.parse_duration("P1M"),
        formula=TieredFormula(
            bands=[
                TierBand(up_to=10.0, formula=PeriodicFormula(period=Period.MONTHLY, constant_cost=100.0)),
                TierBand(formula=PeriodicFormula(period=Period.MONTHLY, constant_cost=180.0)),
            ]
        ),
    )
    capacity_data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00:00", "2025-02-01 00:00:00"]),
            "value": [9.0, 15.0],
        }
    )

    out = component.apply(capacity_data)

    assert out["value"].tolist() == [100.0, 180.0]


def test_capacity_component_supports_scheduled_formulas_inside_bands() -> None:
    component = CapacityComponent(
        measurement_period=isodate.parse_duration("P1M"),
        billing_period=isodate.parse_duration("P1M"),
        formula=TieredFormula(
            bands=[
                TierBand(
                    formula=ScheduledFormulas(
                        schedule=[
                            ScheduledFormula(
                                when=[WhenClause(days=[DayOfWeek.WEDNESDAY])],
                                formula=IndexFormula(constant_cost=5.0),
                            ),
                            ScheduledFormula(formula=IndexFormula(constant_cost=2.0)),
                        ]
                    )
                )
            ]
        ),
    )
    capacity_data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00:00", "2025-02-01 00:00:00"]),
            "value": [8.0, 8.0],
        }
    )

    out = component.apply(capacity_data)

    assert out["value"].tolist() == [40.0, 16.0]


def test_capacity_component_applies_rolling_average_window() -> None:
    component = CapacityComponent(
        measurement_period=isodate.parse_duration("P1M"),
        billing_period=isodate.parse_duration("P1M"),
        window_periods=2,
        formula=IndexFormula(constant_cost=10.0),
    )
    capacity_data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00:00", "2025-02-01 00:00:00", "2025-03-01 00:00:00"]),
            "value": [4.0, 8.0, 10.0],
        }
    )

    out = component.apply(capacity_data)

    assert out["value"].tolist() == [40.0, 60.0, 90.0]


def test_tariff_applies_capacity_cost_across_version_boundary() -> None:
    tariff = Tariff(
        versions=[
            TariffVersion(
                start=dt.datetime(2025, 1, 1, 0, 0),
                capacity=CapacityComponent(
                    measurement_period=isodate.parse_duration("P1M"),
                    billing_period=isodate.parse_duration("P1M"),
                    formula=IndexFormula(constant_cost=10.0),
                ),
            ),
            TariffVersion(
                start=dt.datetime(2025, 2, 1, 0, 0),
                capacity=CapacityComponent(
                    measurement_period=isodate.parse_duration("P1M"),
                    billing_period=isodate.parse_duration("P1M"),
                    formula=IndexFormula(constant_cost=20.0),
                ),
            ),
        ]
    )
    capacity_data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00:00", "2025-02-01 00:00:00"]),
            "value": [5.0, 5.0],
        }
    )

    out = tariff.apply_capacity_cost(capacity_data)

    assert out["timestamp"].tolist() == list(pd.to_datetime(["2025-01-01 00:00:00", "2025-02-01 00:00:00"]))
    assert out["value"].tolist() == [50.0, 100.0]
