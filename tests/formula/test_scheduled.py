from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest
from pydantic import ValidationError

from energy_cost.formula import DayOfWeek, IndexFormula, ScheduledFormula, ScheduledFormulas, WhenClause


def test_when_clause_rejects_start_after_end() -> None:
    with pytest.raises(ValidationError, match="strictly before"):
        WhenClause(start=dt.time(22, 0), end=dt.time(6, 0))


def test_when_clause_rejects_equal_start_end() -> None:
    with pytest.raises(ValidationError, match="strictly before"):
        WhenClause(start=dt.time(9, 0), end=dt.time(9, 0))


def test_when_clause_matches_days_and_half_open_time_range() -> None:
    clause = WhenClause(days=[DayOfWeek.MONDAY], start=dt.time(9, 0), end=dt.time(17, 0))
    ts = pd.Series(pd.to_datetime(["2026-03-16 08:59", "2026-03-16 09:00", "2026-03-16 16:59", "2026-03-16 17:00"]))

    assert clause.matches(ts).tolist() == [False, True, True, False]


def test_scheduled_formula_coerces_nested_formula_dict() -> None:
    scheduled = ScheduledFormula.model_validate(
        {
            "when": [{"days": ["monday"]}],
            "formula": {"constant_cost": 10.0},
        }
    )

    assert isinstance(scheduled.formula, IndexFormula)
    assert scheduled.formula.constant_cost == 10.0


def test_scheduled_formulas_first_match_wins() -> None:
    scheduled = ScheduledFormulas(
        schedule=[
            ScheduledFormula(
                when=[WhenClause(days=[DayOfWeek.MONDAY])],
                formula=IndexFormula(constant_cost=10.0),
            ),
            ScheduledFormula(
                when=[WhenClause(days=list(DayOfWeek))],
                formula=IndexFormula(constant_cost=99.0),
            ),
            ScheduledFormula(formula=IndexFormula(constant_cost=1.0)),
        ]
    )

    out = scheduled.get_values(dt.datetime(2026, 3, 16), dt.datetime(2026, 3, 16, 1), dt.timedelta(hours=1))

    assert out["value"].tolist() == [10.0]


def test_scheduled_formulas_fallback_covers_unmatched_timestamps() -> None:
    scheduled = ScheduledFormulas(
        schedule=[
            ScheduledFormula(
                when=[WhenClause(days=[DayOfWeek.MONDAY])],
                formula=IndexFormula(constant_cost=10.0),
            ),
            ScheduledFormula(formula=IndexFormula(constant_cost=50.0)),
        ]
    )

    out = scheduled.get_values(dt.datetime(2026, 3, 21), dt.datetime(2026, 3, 22), dt.timedelta(hours=1))

    assert (out["value"] == 50.0).all()


def test_scheduled_formulas_support_multiple_masks() -> None:
    scheduled = ScheduledFormulas(
        schedule=[
            ScheduledFormula(
                when=[
                    WhenClause(
                        days=[
                            DayOfWeek.MONDAY,
                            DayOfWeek.TUESDAY,
                            DayOfWeek.WEDNESDAY,
                            DayOfWeek.THURSDAY,
                            DayOfWeek.FRIDAY,
                        ],
                        start=dt.time(6, 0),
                        end=dt.time(10, 0),
                    ),
                    WhenClause(
                        days=[DayOfWeek.SATURDAY, DayOfWeek.SUNDAY],
                        start=dt.time(7, 0),
                        end=dt.time(19, 0),
                    ),
                ],
                formula=IndexFormula(constant_cost=300.0),
            ),
            ScheduledFormula(
                when=[
                    WhenClause(start=dt.time(10, 0), end=dt.time(13, 0)),
                    WhenClause(start=dt.time(18, 0), end=dt.time(22, 0)),
                ],
                formula=IndexFormula(constant_cost=150.0),
            ),
            ScheduledFormula(formula=IndexFormula(constant_cost=100.0)),
        ]
    )

    out = scheduled.get_values(
        dt.datetime(2026, 3, 16, 0, 0),
        dt.datetime(2026, 3, 17, 0, 0),
        dt.timedelta(hours=1),
    )

    assert out.iloc[6]["value"] == 300.0
    assert out.iloc[10]["value"] == 150.0
    assert out.iloc[13]["value"] == 100.0
    assert out.iloc[18]["value"] == 150.0


def test_scheduled_formulas_apply_multiplies_matching_formula_values() -> None:
    formula = ScheduledFormulas(
        schedule=[
            ScheduledFormula(
                when=[WhenClause(days=[DayOfWeek.MONDAY])],
                formula=IndexFormula(constant_cost=5.0),
            ),
            ScheduledFormula(formula=IndexFormula(constant_cost=2.0)),
        ]
    )
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-03-16 00:00:00", "2026-03-17 00:00:00"]),
            "value": [3.0, 4.0],
        }
    )

    out = formula.apply(data, resolution=dt.timedelta(days=1))

    assert out["value"].tolist() == [15.0, 8.0]
