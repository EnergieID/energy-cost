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
    TieringMode,
)
from energy_cost.formula.scheduled import DayOfWeek, WhenClause


def test_tier_band_coerces_nested_formula_dict() -> None:
    band = TierBand.model_validate({"up_to": 10.0, "formula": {"constant_cost": 4.0}})

    assert isinstance(band.formula, IndexFormula)
    assert band.formula.constant_cost == 4.0


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
        mode=TieringMode.BANDED,
        bands=[
            TierBand(up_to=10.0, formula=IndexFormula(constant_cost=4.0)),
            TierBand(formula=IndexFormula(constant_cost=6.0)),
        ],
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
        mode=TieringMode.BANDED,
        bands=[
            TierBand(up_to=10.0, formula=PeriodicFormula(period=isodate.parse_duration("P1M"), constant_cost=100.0)),
            TierBand(formula=PeriodicFormula(period=isodate.parse_duration("P1M"), constant_cost=180.0)),
        ],
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
        mode=TieringMode.BANDED,
        bands=[
            # This band only covers values <= 5; all test values are > 5, so it is skipped.
            TierBand(up_to=5.0, formula=IndexFormula(constant_cost=1.0)),
            TierBand(formula=IndexFormula(constant_cost=10.0)),
        ],
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
        mode=TieringMode.BANDED,
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
        mode=TieringMode.BANDED,
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


def test_tiered_formula_progressive_is_default() -> None:
    """TieredFormula defaults to progressive mode without needing an explicit mode argument."""
    formula = TieredFormula(
        bands=[
            TierBand(up_to=10.0, formula=IndexFormula(constant_cost=2.0)),
            TierBand(formula=IndexFormula(constant_cost=6.0)),
        ]
    )
    # value 10 is fully within the first band → fraction = 1.0 → same in both modes
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00:00"]),
            "value": [10.0],
        }
    )

    out = formula.apply(data, resolution=isodate.parse_duration("P1M"))

    assert out["value"].tolist() == [20.0]


def test_tiered_formula_progressive_rowwise_splits_across_bands() -> None:
    """In progressive mode (no band_period), a row's value spanning multiple bands is
    split proportionally: each band's formula is applied to the full row value and
    weighted by that band's fraction of the total.

    value=20 with bands up_to 5 (€2/MWh) and catch-all (€6/MWh):
      fraction_A = 5/20 = 0.25  →  20 * 2.0 * 0.25 = 10.0
      fraction_B = 15/20 = 0.75 →  20 * 6.0 * 0.75 = 90.0
      total cost = 100.0
    """
    formula = TieredFormula(
        bands=[
            TierBand(up_to=5.0, formula=IndexFormula(constant_cost=2.0)),
            TierBand(formula=IndexFormula(constant_cost=6.0)),
        ]
    )
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-01-01 00:00:00"]),
            "value": [20.0],
        }
    )

    out = formula.apply(data, resolution=isodate.parse_duration("P1M"))

    assert out["value"].tolist() == [100.0]


def test_tiered_formula_progressive_with_band_period_splits_across_bands() -> None:
    """In progressive mode with band_period, the period total determines band fractions
    and those fractions are applied uniformly to every timestamp in the period.

    12 MWh/year total with bands up_to 5 (€2), up_to 10 (€4), catch-all (€6):
      fraction_A = 5/12,  fraction_B = 5/12,  fraction_C = 2/12
      blended rate per MWh = (5*2 + 5*4 + 2*6) / 12 = (10 + 20 + 12) / 12 = 42/12 = 3.5
      each monthly row of 1 MWh → cost = 3.5
    """
    formula = TieredFormula(
        band_period=isodate.parse_duration("P1Y"),
        bands=[
            TierBand(up_to=5.0, formula=IndexFormula(constant_cost=2.0)),
            TierBand(up_to=10.0, formula=IndexFormula(constant_cost=4.0)),
            TierBand(formula=IndexFormula(constant_cost=6.0)),
        ],
    )
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=12, freq="MS"),
            "value": [1.0] * 12,  # 12 MWh/year total
        }
    )

    out = formula.apply(data, resolution=isodate.parse_duration("P1M"))

    expected = pytest.approx([3.5] * 12)
    assert out["value"].tolist() == expected


def test_tiered_formula_progressive_with_band_period_notebook_scenario() -> None:
    """Regression: 1 MWh/month = 12 MWh/year should be priced progressively across
    all three bands (up_to 3 @ €5, up_to 5 @ €7, catch-all @ €10), not locked into
    the first band just because 1 MWh < 3 MWh.

      fraction_A = 3/12 = 0.25   → 1 * 5.0 * 0.25  = 1.25
      fraction_B = 2/12 = 1/6    → 1 * 7.0 * (1/6) ≈ 1.1667
      fraction_C = 7/12          → 1 * 10.0 * (7/12) ≈ 5.8333
      total per row ≈ 8.25
    """
    formula = TieredFormula(
        band_period=isodate.parse_duration("P1Y"),
        bands=[
            TierBand(up_to=3.0, formula=IndexFormula(constant_cost=5.0)),
            TierBand(up_to=5.0, formula=IndexFormula(constant_cost=7.0)),
            TierBand(formula=IndexFormula(constant_cost=10.0)),
        ],
    )
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=12, freq="MS"),
            "value": [1.0] * 12,
        }
    )

    out = formula.apply(data, resolution=isodate.parse_duration("P1M"))

    # blended rate = (3*5 + 2*7 + 7*10) / 12 = (15 + 14 + 70) / 12 = 99/12 = 8.25
    assert out["value"].tolist() == pytest.approx([99 / 12] * 12)


def test_banded_tiers_raises_when_no_band_matches_estimated_total() -> None:
    """In banded mode, if all bands have an up_to threshold and the period total exceeds
    the last one, _apply_banded_group has no matching band and must raise ValueError."""
    formula = TieredFormula(
        mode=TieringMode.BANDED,
        band_period=isodate.parse_duration("P1Y"),
        bands=[
            TierBand(up_to=5.0, formula=IndexFormula(constant_cost=2.0)),
            TierBand(up_to=10.0, formula=IndexFormula(constant_cost=4.0)),
            # no catch-all band
        ],
    )
    # 12 MWh/year — exceeds the highest up_to of 10 MWh with no catch-all to fall back on
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=12, freq="MS"),
            "value": [1.0] * 12,
        }
    )

    with pytest.raises(ValueError, match="No tier band matches"):
        formula.apply(data, resolution=isodate.parse_duration("P1M"))
