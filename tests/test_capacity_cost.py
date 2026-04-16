from __future__ import annotations

import datetime as dt

import isodate
import pandas as pd
import pytest

from energy_cost.capacity_cost import CapacityComponent
from energy_cost.formula import (
    DayOfWeek,
    IndexFormula,
    PeriodicFormula,
    ScheduledFormula,
    ScheduledFormulas,
    TierBand,
    TieredFormula,
    TieringMode,
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

    out = component.apply(capacity_data, unit="MW")

    assert out["timestamp"].tolist() == list(pd.to_datetime(["2025-01-01 00:00:00", "2025-02-01 00:00:00"], utc=True))
    assert out["value"].tolist() == [50.0, 70.0]


def test_capacity_component_applies_banded_tiered_formula() -> None:
    component = CapacityComponent(
        measurement_period=isodate.parse_duration("P1M"),
        billing_period=isodate.parse_duration("P1M"),
        formula=TieredFormula(
            mode=TieringMode.BANDED,
            bands=[
                TierBand(up_to=10.0, formula=PeriodicFormula(period=Period.MONTHLY, constant_cost=100.0)),
                TierBand(formula=PeriodicFormula(period=Period.MONTHLY, constant_cost=180.0)),
            ],
        ),
    )
    capacity_data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00:00", "2025-02-01 00:00:00"]),
            "value": [9.0, 15.0],
        }
    )

    out = component.apply(capacity_data, unit="MW")

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

    out = component.apply(capacity_data, unit="MW")

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

    out = component.apply(capacity_data, unit="MW")

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

    out = tariff.apply_capacity_cost(capacity_data, unit="MW")

    assert out is not None
    assert out["timestamp"].tolist() == list(pd.to_datetime(["2025-01-01 00:00:00", "2025-02-01 00:00:00"], utc=True))
    assert out["value"].tolist() == [50.0, 100.0]


def test_data_frame_resolution_should_be_divisor_of_billing_period() -> None:
    component = CapacityComponent(
        measurement_period=isodate.parse_duration("P1M"),
        billing_period=isodate.parse_duration("P1M"),
        formula=IndexFormula(constant_cost=10.0),
    )
    capacity_data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00:00", "2025-03-01 00:00:00"]),
            "value": [5.0, 7.0],
        }
    )

    pytest.raises(ValueError, component.apply, capacity_data)


def test_mw_data_resolution_must_be_divisor_of_billing_period() -> None:
    """unit=MW data whose resolution is not a divisor of billing_period raises ValueError."""
    component = CapacityComponent(
        measurement_period=isodate.parse_duration("P1M"),
        billing_period=isodate.parse_duration("P1M"),
        formula=IndexFormula(constant_cost=10.0),
    )
    # 2-monthly resolution is not a divisor of 1-month billing period
    capacity_data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00:00", "2025-03-01 00:00:00"]),
            "value": [5.0, 7.0],
        }
    )

    with pytest.raises(ValueError):
        component.apply(capacity_data, unit="MW")


def test_capacity_component_converts_mwh_to_mw_for_15min_data() -> None:
    """15-min MWh consumption is correctly converted to MW capacity and then billed."""
    # measurement_period == data resolution == PT15M, billing_period == P1M
    # Each slot: 2.0 MWh / 0.25h = 8.0 MW; max over January = 8.0 MW; cost = 8.0 * 10.0 = 80.0
    component = CapacityComponent(
        measurement_period=isodate.parse_duration("PT15M"),
        billing_period=isodate.parse_duration("P1M"),
        formula=IndexFormula(constant_cost=10.0),
    )
    timestamps = pd.date_range("2025-01-01 00:00:00", periods=4, freq="15min")
    capacity_data = pd.DataFrame({"timestamp": timestamps, "value": 2.0})

    out = component.apply(capacity_data)

    assert len(out) == 1
    assert out["value"].iloc[0] == pytest.approx(80.0)


def test_capacity_component_aggregates_15min_mwh_to_hourly_measurement_period() -> None:
    """15-min MWh data is summed to 1-hour measurement periods before MW conversion."""
    # 4 slots × 1.0 MWh = 4.0 MWh per hour → 4.0 MWh / 1.0h = 4.0 MW
    # Two hours in January: max = 4.0 MW; cost = 4.0 * 5.0 = 20.0
    component = CapacityComponent(
        measurement_period=isodate.parse_duration("PT1H"),
        billing_period=isodate.parse_duration("P1M"),
        formula=IndexFormula(constant_cost=5.0),
    )
    timestamps = pd.date_range("2025-01-01 00:00:00", periods=8, freq="15min")
    capacity_data = pd.DataFrame({"timestamp": timestamps, "value": 1.0})

    out = component.apply(capacity_data)

    assert len(out) == 1
    assert out["value"].iloc[0] == pytest.approx(20.0)


def test_capacity_component_mwh_resolution_must_be_divisor_of_measurement_period() -> None:
    """MWh data whose resolution is not a divisor of measurement_period raises ValueError."""
    component = CapacityComponent(
        measurement_period=isodate.parse_duration("PT1H"),
        billing_period=isodate.parse_duration("P1M"),
        formula=IndexFormula(constant_cost=10.0),
    )
    # 45-min resolution is not a divisor of 1h
    timestamps = pd.date_range("2025-01-01 00:00:00", periods=4, freq="45min")
    capacity_data = pd.DataFrame({"timestamp": timestamps, "value": 1.0})

    with pytest.raises(ValueError):
        component.apply(capacity_data)


def test_tariff_version_apply_capacity_cost_accepts_unit_param() -> None:
    """TariffVersion.apply_capacity_cost passes the unit param to the capacity component."""
    version = TariffVersion(
        start=dt.datetime(2025, 1, 1, 0, 0),
        capacity=CapacityComponent(
            measurement_period=isodate.parse_duration("P1M"),
            billing_period=isodate.parse_duration("P1M"),
            formula=IndexFormula(constant_cost=10.0),
        ),
    )
    capacity_data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00:00", "2025-02-01 00:00:00"]),
            "value": [3.0, 6.0],
        }
    )

    out = version.apply_capacity_cost(capacity_data, unit="MW")

    assert out is not None
    assert out["value"].tolist() == [30.0, 60.0]


def test_tariff_apply_capacity_cost_accepts_unit_param() -> None:
    """Tariff.apply_capacity_cost passes unit=MW to each version's capacity component."""
    tariff = Tariff(
        versions=[
            TariffVersion(
                start=dt.datetime(2025, 1, 1, 0, 0),
                capacity=CapacityComponent(
                    measurement_period=isodate.parse_duration("P1M"),
                    billing_period=isodate.parse_duration("P1M"),
                    formula=IndexFormula(constant_cost=10.0),
                ),
            )
        ]
    )
    capacity_data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00:00", "2025-02-01 00:00:00"]),
            "value": [4.0, 7.0],
        }
    )

    out = tariff.apply_capacity_cost(capacity_data, unit="MW")

    assert out is not None
    assert out["value"].tolist() == [40.0, 70.0]
