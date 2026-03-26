from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import pytest

from energy_cost.fractional_periods import Period
from energy_cost.periodic_cost import PeriodicCost
from energy_cost.price_formula import PriceFormula
from energy_cost.tariff import Tariff
from energy_cost.tariff_version import CostType, TariffVersion


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

    assert len(tariff.versions) == 2
    assert tariff.versions[0].start == dt.datetime(2025, 1, 1, 0, 0)
    assert tariff.versions[1].start == dt.datetime(2026, 1, 1, 0, 0)


def test_tariff_from_yaml_supports_scheduled_formula_list_shorthand() -> None:
    tariff = Tariff.from_yaml("data/tariffs/scheduled.yml")

    out = tariff.get_cost(
        start=dt.datetime.fromisoformat("2025-01-06T05:00:00+01:00"),
        end=dt.datetime.fromisoformat("2025-01-06T11:00:00+01:00"),
        resolution=dt.timedelta(hours=1),
    )

    assert out["energy"].tolist() == [100.0, 300.0, 300.0, 300.0, 300.0, 150.0]
    assert out["total"].tolist() == [100.0, 300.0, 300.0, 300.0, 300.0, 150.0]


def test_get_cost_uses_correct_segment_for_time_range() -> None:
    """Values from the active segment are used; the boundary switches formulas correctly."""
    tariff = Tariff(
        versions=[
            TariffVersion(
                start=dt.datetime(2025, 1, 1, 0, 0),
                consumption={"all": {CostType.ENERGY: PriceFormula(constant_cost=1.0)}},
            ),
            TariffVersion(
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
    tariff = Tariff(versions=[TariffVersion(start=dt.datetime(2025, 1, 1, 0, 0))])

    with pytest.raises(ValueError, match="No formulas for meter type 'single_rate' and direction 'consumption'"):
        tariff.get_cost(
            start=dt.datetime(2025, 1, 1, 0, 0),
            end=dt.datetime(2025, 1, 1, 1, 0),
        )


def test_get_cost_raises_when_no_versions_overlap_interval() -> None:
    tariff = Tariff(
        versions=[
            TariffVersion(
                start=dt.datetime(2025, 1, 2, 0, 0),
                consumption={"all": {CostType.ENERGY: PriceFormula(constant_cost=1.0)}},
            )
        ]
    )

    with pytest.raises(ValueError, match="No active versions with formulas"):
        tariff.get_cost(
            start=dt.datetime(2025, 1, 1, 0, 0),
            end=dt.datetime(2025, 1, 1, 1, 0),
        )


def test_get_periodic_cost_spans_multiple_segments() -> None:
    """Periodic costs are summed across segment boundaries within the queried interval."""
    tariff = Tariff(
        versions=[
            TariffVersion(
                start=dt.datetime(2025, 1, 1, 0, 0),
                periodic={"admin": PeriodicCost(period=Period.DAILY, constant_cost=24.0)},
            ),
            TariffVersion(
                start=dt.datetime(2025, 1, 1, 12, 0),
                periodic={"admin": PeriodicCost(period=Period.DAILY, constant_cost=48.0)},
            ),
        ]
    )

    costs = tariff.get_periodic_cost(
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 2, 0, 0),
    )

    assert costs == pytest.approx({"admin": 36.0})
