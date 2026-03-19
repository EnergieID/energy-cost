from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from energy_cost.fractional_periods import Period
from energy_cost.periodic_cost import PeriodicCost
from energy_cost.price_formula import PriceFormula
from energy_cost.scheduled_formula import ScheduledPriceFormulas
from energy_cost.tariff import CostType, MeterType, PowerDirection, Tariff, TariffSegment


def _constant_cost(f: PriceFormula | ScheduledPriceFormulas) -> float:
    assert isinstance(f, PriceFormula)
    return f.constant_cost


def test_tariff_from_yaml_versioned_segments(tmp_path: Path) -> None:
    """A YAML list is loaded as multiple segments sorted by start date."""
    path = tmp_path / "tariff.yml"
    path.write_text(
        "- start: 2026-01-01T00:00:00\n"
        "  consumption:\n"
        "    all:\n"
        "      energy:\n"
        "        constant_cost: 2.0\n"
        "- start: 2025-01-01T00:00:00\n"
        "  consumption:\n"
        "    all:\n"
        "      energy:\n"
        "        constant_cost: 1.0\n",
        encoding="utf-8",
    )

    tariff = Tariff.from_yaml(path)

    assert len(tariff.segments) == 2
    assert tariff.segments[0].start == dt.datetime(2025, 1, 1, 0, 0)
    assert tariff.segments[1].start == dt.datetime(2026, 1, 1, 0, 0)


def test_get_cost_uses_correct_segment_for_time_range() -> None:
    """Values from the active segment are used; the boundary switches formulas correctly."""
    tariff = Tariff(
        segments=[
            TariffSegment(
                start=dt.datetime(2025, 1, 1, 0, 0),
                consumption={"all": {CostType.ENERGY: PriceFormula(constant_cost=1.0)}},
            ),
            TariffSegment(
                start=dt.datetime(2025, 1, 1, 0, 30),
                consumption={"all": {CostType.ENERGY: PriceFormula(constant_cost=2.0)}},
            ),
        ]
    )

    out = tariff.get_cost(
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 1, 0),
        resolution=dt.timedelta(minutes=15),
    )

    assert out["timestamp"].tolist() == list(pd.date_range("2025-01-01", periods=4, freq="15min"))
    assert out["energy"].tolist() == [1.0, 1.0, 2.0, 2.0]
    assert out["total"].tolist() == [1.0, 1.0, 2.0, 2.0]


def test_get_cost_raises_when_no_formulas_found() -> None:
    tariff = Tariff(segments=[TariffSegment(start=dt.datetime(2025, 1, 1, 0, 0))])

    with pytest.raises(
        ValueError, match="No formulas for meter type 'single_rate' and direction 'consumption' found in tariff"
    ):
        tariff.get_cost(
            start=dt.datetime(2025, 1, 1, 0, 0),
            end=dt.datetime(2025, 1, 1, 1, 0),
        )


def test_all_key_applies_to_all_meter_types() -> None:
    """Formulas under ``all`` are visible regardless of the meter type queried."""
    segment = TariffSegment(
        start=dt.datetime(2025, 1, 1, 0, 0),
        injection={"all": {CostType.ENERGY: PriceFormula(constant_cost=-5.0)}},
    )

    single = segment.resolve_cost_formulas(MeterType.SINGLE_RATE, PowerDirection.INJECTION)
    tou = segment.resolve_cost_formulas(MeterType.TOU_PEAK, PowerDirection.INJECTION)

    assert _constant_cost(single[CostType.ENERGY]) == -5.0
    assert _constant_cost(tou[CostType.ENERGY]) == -5.0


def test_specific_meter_type_overrides_all() -> None:
    """A meter-type-specific formula takes precedence over the ``all`` formula for the same cost type."""
    segment = TariffSegment(
        start=dt.datetime(2025, 1, 1, 0, 0),
        consumption={
            "all": {CostType.ENERGY: PriceFormula(constant_cost=1.0)},
            "single_rate": {CostType.ENERGY: PriceFormula(constant_cost=99.0)},
        },
    )

    single = segment.resolve_cost_formulas(MeterType.SINGLE_RATE, PowerDirection.CONSUMPTION)
    tou = segment.resolve_cost_formulas(MeterType.TOU_PEAK, PowerDirection.CONSUMPTION)

    assert _constant_cost(single[CostType.ENERGY]) == 99.0
    assert _constant_cost(tou[CostType.ENERGY]) == 1.0


def test_get_cost_returns_column_per_cost_type() -> None:
    tariff = Tariff(
        segments=[
            TariffSegment(
                start=dt.datetime(2025, 1, 1, 0, 0),
                consumption={
                    "all": {
                        CostType.ENERGY: PriceFormula(constant_cost=10.0),
                        CostType.CHP_CERTIFICATES: PriceFormula(constant_cost=2.0),
                        CostType.RENEWABLE_CERTIFICATES: PriceFormula(constant_cost=3.0),
                    }
                },
            )
        ]
    )

    out = tariff.get_cost(
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 1, 0),
    )

    assert set(out.columns) == {"timestamp", "energy", "chp_certificates", "renewable_certificates", "total"}
    assert out["energy"].tolist() == [10.0, 10.0, 10.0, 10.0]
    assert out["chp_certificates"].tolist() == [2.0, 2.0, 2.0, 2.0]
    assert out["renewable_certificates"].tolist() == [3.0, 3.0, 3.0, 3.0]
    assert out["total"].tolist() == [15.0, 15.0, 15.0, 15.0]


def test_get_periodic_cost_spans_multiple_segments() -> None:
    """Periodic costs are summed across segment boundaries within the queried interval."""
    tariff = Tariff(
        segments=[
            TariffSegment(
                start=dt.datetime(2025, 1, 1, 0, 0),
                periodic={"admin": PeriodicCost(period=Period.DAILY, constant_cost=24.0)},
            ),
            TariffSegment(
                start=dt.datetime(2025, 1, 1, 12, 0),
                periodic={"admin": PeriodicCost(period=Period.DAILY, constant_cost=48.0)},
            ),
        ]
    )

    costs = tariff.get_periodic_cost(
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 2, 0, 0),
    )

    # First 12 h: 24 * (12/24) = 12.0; second 12 h: 48 * (12/24) = 24.0; total: 36.0
    assert costs == pytest.approx({"admin": 36.0})
