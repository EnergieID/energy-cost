from __future__ import annotations

import datetime as dt

import isodate
import pandas as pd

from energy_cost.formula import (
    IndexFormula,
    PeriodicFormula,
    ScheduledFormula,
    ScheduledFormulas,
    TierBand,
    TieredFormula,
)
from energy_cost.formula.scheduled import DayOfWeek, WhenClause
from energy_cost.fractional_periods import Period


def test_tier_band_coerces_nested_formula_dict() -> None:
    band = TierBand.model_validate({"up_to": 10.0, "formula": {"constant_cost": 4.0}})

    assert isinstance(band.formula, IndexFormula)
    assert band.formula.constant_cost == 4.0


def test_tier_band_matches_values_up_to_threshold() -> None:
    band = TierBand(up_to=10.0, formula=IndexFormula(constant_cost=1.0))
    values = pd.Series([5.0, 10.0, 12.0])

    assert band.matches(values).tolist() == [True, True, False]


def test_tiered_formula_get_values_returns_one_column_per_band() -> None:
    formula = TieredFormula(
        bands=[
            TierBand(up_to=10.0, formula=IndexFormula(constant_cost=4.0)),
            TierBand(formula=IndexFormula(constant_cost=6.0)),
        ]
    )

    out = formula.get_values(
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 0, 30),
        resolution=dt.timedelta(minutes=15),
    )

    assert list(out.columns) == ["timestamp", "tier_1", "tier_2"]
    assert out["tier_1"].tolist() == [4.0, 4.0]
    assert out["tier_2"].tolist() == [6.0, 6.0]


def test_tiered_formula_apply_uses_first_matching_band() -> None:
    formula = TieredFormula(
        bands=[
            TierBand(up_to=10.0, formula=IndexFormula(constant_cost=4.0)),
            TierBand(formula=IndexFormula(constant_cost=6.0)),
        ]
    )
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00:00", "2025-02-01 00:00:00"]),
            "value": [9.0, 15.0],
        }
    )

    out = formula.apply(data, resolution=isodate.parse_duration("P1M"))

    assert out["value"].tolist() == [36.0, 90.0]


def test_tiered_formula_apply_supports_fixed_periodic_band() -> None:
    formula = TieredFormula(
        bands=[
            TierBand(up_to=10.0, formula=PeriodicFormula(period=Period.MONTHLY, constant_cost=100.0)),
            TierBand(formula=PeriodicFormula(period=Period.MONTHLY, constant_cost=180.0)),
        ]
    )
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00:00", "2025-02-01 00:00:00"]),
            "value": [9.0, 15.0],
        }
    )

    out = formula.apply(data, resolution=isodate.parse_duration("P1M"))

    assert out["value"].tolist() == [100.0, 180.0]


def test_tiered_formula_apply_supports_scheduled_formula_band() -> None:
    formula = TieredFormula(
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
    )
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00:00", "2025-02-01 00:00:00"]),
            "value": [3.0, 8.0],
        }
    )

    out = formula.apply(data, resolution=isodate.parse_duration("P1M"))

    assert out["value"].tolist() == [15.0, 16.0]


def test_get_values_returns_empty_dataframe_for_tiered_formula_with_no_bands() -> None:
    formula = TieredFormula(bands=[])

    out = formula.get_values(
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 0, 30),
        resolution=dt.timedelta(minutes=15),
    )

    assert out.empty
