from __future__ import annotations

import datetime as dt

import isodate
import pandas as pd
import pytest

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


def test_tiered_formula_get_values_raises_not_implemented() -> None:
    formula = TieredFormula(
        bands=[
            TierBand(up_to=10.0, formula=IndexFormula(constant_cost=4.0)),
            TierBand(formula=IndexFormula(constant_cost=6.0)),
        ]
    )

    with pytest.raises(NotImplementedError):
        formula.get_values(
            start=dt.datetime(2025, 1, 1, 0, 0),
            end=dt.datetime(2025, 1, 1, 0, 30),
            resolution=dt.timedelta(minutes=15),
        )


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


def test_tiered_formula_get_values_raises_not_implemented_with_no_bands() -> None:
    formula = TieredFormula(bands=[])

    with pytest.raises(NotImplementedError):
        formula.get_values(
            start=dt.datetime(2025, 1, 1, 0, 0),
            end=dt.datetime(2025, 1, 1, 0, 30),
            resolution=dt.timedelta(minutes=15),
        )


def test_tiered_formula_apply_skips_band_that_matches_no_rows() -> None:
    """When a band's threshold matches none of the data rows the band is skipped
    and remaining rows continue to the next band."""
    formula = TieredFormula(
        bands=[
            # This band only covers values <= 5; all test values are > 5, so it is skipped.
            TierBand(up_to=5.0, formula=IndexFormula(constant_cost=1.0)),
            TierBand(formula=IndexFormula(constant_cost=10.0)),
        ]
    )
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00:00", "2025-02-01 00:00:00"]),
            "value": [8.0, 12.0],
        }
    )

    out = formula.apply(data, resolution=isodate.parse_duration("P1M"))

    # Both rows fall through to the catch-all band (constant_cost=10).
    assert out["value"].tolist() == [80.0, 120.0]


def test_tiered_formula_band_resolution_selects_band_by_annual_sum() -> None:
    """When band_resolution is set, the band is chosen based on the total consumption
    aggregated over that resolution period, not the per-row value."""
    formula = TieredFormula(
        band_period=isodate.parse_duration("P1Y"),
        bands=[
            TierBand(up_to=15.0, formula=IndexFormula(constant_cost=2.0)),
            TierBand(formula=IndexFormula(constant_cost=5.0)),
        ],
    )
    # 12 monthly rows summing to 12 MWh/year — below the 15 MWh threshold → band 1
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=12, freq="MS"),
            "value": [1.0] * 12,
        }
    )

    out = formula.apply(data, resolution=isodate.parse_duration("P1M"))

    assert out["value"].tolist() == [2.0] * 12


def test_tiered_formula_band_resolution_extrapolates_incomplete_period() -> None:
    """An incomplete period is scaled to a full period before band matching.
    Two months of 5 MWh each extrapolates to ~62 MWh/year, pushing into the higher band
    even though the raw two-month sum (10 MWh) would have fallen below the threshold."""
    formula = TieredFormula(
        band_period=isodate.parse_duration("P1Y"),
        bands=[
            TierBand(up_to=20.0, formula=IndexFormula(constant_cost=2.0)),
            TierBand(formula=IndexFormula(constant_cost=5.0)),
        ],
    )
    # Only Jan–Feb 2025 — extrapolated annual sum ≈ 10 × (365/59) ≈ 62 MWh > 20 → band 2
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=2, freq="MS"),
            "value": [5.0, 5.0],
        }
    )

    out = formula.apply(data, resolution=isodate.parse_duration("P1M"))

    assert out["value"].tolist() == [25.0, 25.0]


def test_tiered_formula_band_resolution_handles_multiple_periods() -> None:
    """Each band_resolution period is evaluated independently: the same formula can
    select different bands in different years based on that year's consumption."""
    formula = TieredFormula(
        band_period=isodate.parse_duration("P1Y"),
        bands=[
            TierBand(up_to=15.0, formula=IndexFormula(constant_cost=2.0)),
            TierBand(formula=IndexFormula(constant_cost=5.0)),
        ],
    )
    # 2025: 12 × 1 MWh = 12 MWh/year → band 1 (cost 2.0)
    # 2026: 12 × 2 MWh = 24 MWh/year → band 2 (cost 5.0)
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=24, freq="MS"),
            "value": [1.0] * 12 + [2.0] * 12,
        }
    )

    out = formula.apply(data, resolution=isodate.parse_duration("P1M"))

    assert out["value"].tolist() == [2.0] * 12 + [10.0] * 12
