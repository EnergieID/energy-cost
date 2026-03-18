from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest
from pydantic import ValidationError

from energy_cost.scheduled_formula import (
    DayOfWeek,
    ScheduledPriceFormula,
    ScheduledPriceFormulas,
    WhenClause,
)

# ---------------------------------------------------------------------------
# WhenClause
# ---------------------------------------------------------------------------


def test_when_clause_rejects_start_after_end() -> None:
    with pytest.raises(ValidationError, match="strictly before"):
        WhenClause(start=dt.time(22, 0), end=dt.time(6, 0))


def test_when_clause_rejects_equal_start_end() -> None:
    with pytest.raises(ValidationError, match="strictly before"):
        WhenClause(start=dt.time(9, 0), end=dt.time(9, 0))


def test_when_clause_no_end_matches_rest_of_day() -> None:
    # No end → matches from start until end of day
    clause = WhenClause(start=dt.time(9, 0))
    ts = pd.Series(pd.to_datetime(["2026-03-16 08:59", "2026-03-16 09:00", "2026-03-16 23:00"]))
    assert clause.matches(ts).tolist() == [False, True, True]


def test_when_clause_half_open_interval() -> None:
    # Matches [09:00, 17:00)
    clause = WhenClause(start=dt.time(9, 0), end=dt.time(17, 0))
    ts = pd.Series(pd.to_datetime(["2026-03-16 08:59", "2026-03-16 09:00", "2026-03-16 16:59", "2026-03-16 17:00"]))
    assert clause.matches(ts).tolist() == [False, True, True, False]


# ---------------------------------------------------------------------------
# WhenClause — days and time filtering
# ---------------------------------------------------------------------------


def test_when_clause_days_only() -> None:
    # 2026-03-16 Mon, 2026-03-17 Tue, 2026-03-18 Wed
    clause = WhenClause(days=[DayOfWeek.MONDAY, DayOfWeek.TUESDAY])
    ts = pd.Series(pd.to_datetime(["2026-03-16", "2026-03-17", "2026-03-18"]))
    assert clause.matches(ts).tolist() == [True, True, False]


def test_when_clause_days_and_time_range() -> None:
    # 2026-03-16 is a Monday
    clause = WhenClause(days=[DayOfWeek.MONDAY], start=dt.time(9, 0), end=dt.time(17, 0))
    ts = pd.Series(pd.to_datetime(["2026-03-16 08:45", "2026-03-16 09:00", "2026-03-16 16:59", "2026-03-16 17:00"]))
    assert clause.matches(ts).tolist() == [False, True, True, False]


def test_when_clause_default_days_covers_full_week() -> None:
    # Omitting ``days`` → all seven days match
    clause = WhenClause(start=dt.time(9, 0), end=dt.time(17, 0))
    # One noon timestamp per day for a full week starting Monday 2026-03-16
    ts = pd.Series(pd.date_range("2026-03-16 12:00", periods=7, freq="D"))
    assert clause.matches(ts).all()


def test_when_clause_wrong_day_does_not_match() -> None:
    # 2026-03-17 is Tuesday — should not match even though time is in range
    clause = WhenClause(days=[DayOfWeek.MONDAY], start=dt.time(9, 0), end=dt.time(17, 0))
    ts = pd.Series(pd.to_datetime(["2026-03-17 12:00"]))
    assert clause.matches(ts).tolist() == [False]


# ---------------------------------------------------------------------------
# ScheduledPriceFormula — get_values
# ---------------------------------------------------------------------------


def test_first_match_wins() -> None:
    """When two schedules both match a timestamp, the first one is used."""
    scheduled = ScheduledPriceFormulas(
        [
            ScheduledPriceFormula(when=[WhenClause(days=[DayOfWeek.MONDAY])], constant_cost=10.0),
            ScheduledPriceFormula(when=[WhenClause(days=list(DayOfWeek))], constant_cost=99.0),
            ScheduledPriceFormula(constant_cost=1.0),  # fallback
        ]
    )
    # 2026-03-16 is Monday → first schedule wins
    out = scheduled.get_values(dt.datetime(2026, 3, 16), dt.datetime(2026, 3, 16, 1), dt.timedelta(hours=1))
    assert out["value"].tolist() == [10.0]


def test_fallback_covers_unmatched_timestamps() -> None:
    scheduled = ScheduledPriceFormulas(
        [
            ScheduledPriceFormula(when=[WhenClause(days=[DayOfWeek.MONDAY])], constant_cost=10.0),
            ScheduledPriceFormula(constant_cost=50.0),  # fallback
        ]
    )
    # 2026-03-21 is Saturday → fallback
    out = scheduled.get_values(dt.datetime(2026, 3, 21), dt.datetime(2026, 3, 22), dt.timedelta(hours=1))
    assert (out["value"] == 50.0).all()


def test_weekday_weekend_split() -> None:
    """Formula A on weekdays, formula B (fallback) on weekends."""
    scheduled = ScheduledPriceFormulas(
        [
            ScheduledPriceFormula(
                when=[
                    WhenClause(
                        days=[
                            DayOfWeek.MONDAY,
                            DayOfWeek.TUESDAY,
                            DayOfWeek.WEDNESDAY,
                            DayOfWeek.THURSDAY,
                            DayOfWeek.FRIDAY,
                        ]
                    )
                ],
                constant_cost=100.0,
            ),
            ScheduledPriceFormula(constant_cost=50.0),  # B fallback (weekends)
        ]
    )
    # Monday full day → all A
    weekday = scheduled.get_values(dt.datetime(2026, 3, 16), dt.datetime(2026, 3, 17), dt.timedelta(hours=1))
    assert (weekday["value"] == 100.0).all()

    # Saturday full day → all B
    weekend = scheduled.get_values(dt.datetime(2026, 3, 21), dt.datetime(2026, 3, 22), dt.timedelta(hours=1))
    assert (weekend["value"] == 50.0).all()


def test_time_of_day_split_on_weekday() -> None:
    """Formula A between 09:00–21:00 on weekdays, B (fallback) otherwise."""
    scheduled = ScheduledPriceFormulas(
        [
            ScheduledPriceFormula(
                when=[
                    WhenClause(
                        days=[
                            DayOfWeek.MONDAY,
                            DayOfWeek.TUESDAY,
                            DayOfWeek.WEDNESDAY,
                            DayOfWeek.THURSDAY,
                            DayOfWeek.FRIDAY,
                        ],
                        start=dt.time(9, 0),
                        end=dt.time(21, 0),
                    )
                ],
                constant_cost=200.0,
            ),
            ScheduledPriceFormula(constant_cost=50.0),  # B fallback
        ]
    )
    # Monday 08:00-12:00 (hourly): 08:00→B, 09:00→A, 10:00→A, 11:00→A
    out = scheduled.get_values(dt.datetime(2026, 3, 16, 8), dt.datetime(2026, 3, 16, 12), dt.timedelta(hours=1))
    assert out["value"].tolist() == [50.0, 200.0, 200.0, 200.0]


def test_three_way_abc_scenario() -> None:
    """Full A/B/C scenario from the feature spec.

    A (300.0): weekdays 06:00–10:00, weekends 07:00–19:00
    C (150.0): weekdays 10:00–13:00 and 18:00–22:00 only
    B (100.0): fallback (all remaining times)
    """
    scheduled = ScheduledPriceFormulas(
        [
            # A — weekday mornings and weekend daytime combined via multiple when clauses
            ScheduledPriceFormula(
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
                constant_cost=300.0,
            ),
            # C — weekday peak windows
            ScheduledPriceFormula(
                when=[
                    WhenClause(
                        days=[
                            DayOfWeek.MONDAY,
                            DayOfWeek.TUESDAY,
                            DayOfWeek.WEDNESDAY,
                            DayOfWeek.THURSDAY,
                            DayOfWeek.FRIDAY,
                        ],
                        start=dt.time(10, 0),
                        end=dt.time(13, 0),
                    ),
                    WhenClause(
                        days=[
                            DayOfWeek.MONDAY,
                            DayOfWeek.TUESDAY,
                            DayOfWeek.WEDNESDAY,
                            DayOfWeek.THURSDAY,
                            DayOfWeek.FRIDAY,
                        ],
                        start=dt.time(18, 0),
                        end=dt.time(22, 0),
                    ),
                ],
                constant_cost=150.0,
            ),
            # B — fallback
            ScheduledPriceFormula(constant_cost=100.0),
        ]
    )

    # Probe Monday 2026-03-16 at specific hours using hourly resolution over the full day.
    out = scheduled.get_values(
        dt.datetime(2026, 3, 16, 0, 0),
        dt.datetime(2026, 3, 17, 0, 0),
        dt.timedelta(hours=1),
    )
    # hour → expected formula
    expectations = {
        5: 100.0,  # B  — before 06:00
        6: 300.0,  # A  — [06:00, 10:00)
        9: 300.0,  # A  — still in [06:00, 10:00)
        10: 150.0,  # C  — [10:00, 13:00), not A (A ends at 10:00)
        12: 150.0,  # C
        13: 100.0,  # B  — C ends at 13:00
        17: 100.0,  # B  — between C windows
        18: 150.0,  # C  — [18:00, 22:00)
        21: 150.0,  # C
        22: 100.0,  # B  — C ends at 22:00
    }
    for hour, expected in expectations.items():
        actual = out.iloc[hour]["value"]
        assert actual == expected, f"hour={hour}: expected {expected}, got {actual}"

    # Probe Saturday 2026-03-21 — C never applies on weekends.
    out_sat = scheduled.get_values(
        dt.datetime(2026, 3, 21, 0, 0),
        dt.datetime(2026, 3, 22, 0, 0),
        dt.timedelta(hours=1),
    )
    sat_expectations = {
        6: 100.0,  # B  — A only starts at 07:00 on weekends
        7: 300.0,  # A
        12: 300.0,  # A
        18: 300.0,  # A  — C never applies on weekends
        19: 100.0,  # B  — A ends at 19:00 on weekends
    }
    for hour, expected in sat_expectations.items():
        actual = out_sat.iloc[hour]["value"]
        assert actual == expected, f"Saturday hour={hour}: expected {expected}, got {actual}"


def test_unmatched_timestamps_are_nan() -> None:
    """Timestamps not covered by any schedule produce NaN values."""
    # Only Monday is covered; Saturday timestamps should be NaN.
    scheduled = ScheduledPriceFormulas(
        [
            ScheduledPriceFormula(
                when=[WhenClause(days=[DayOfWeek.MONDAY])],
                constant_cost=42.0,
            ),
        ]
    )
    # 2026-03-16 Mon 00:00, 2026-03-21 Sat 00:00
    out = scheduled.get_values(dt.datetime(2026, 3, 16), dt.datetime(2026, 3, 16, 2), dt.timedelta(hours=1))
    assert out["value"].tolist() == [42.0, 42.0]

    out_sat = scheduled.get_values(dt.datetime(2026, 3, 21), dt.datetime(2026, 3, 21, 2), dt.timedelta(hours=1))
    import math

    assert all(math.isnan(v) for v in out_sat["value"].tolist())


def test_output_has_correct_timestamps() -> None:
    scheduled = ScheduledPriceFormulas([ScheduledPriceFormula(constant_cost=1.0)])
    start = dt.datetime(2026, 3, 16, 0, 0)
    end = dt.datetime(2026, 3, 16, 1, 0)
    out = scheduled.get_values(start, end, dt.timedelta(minutes=15))
    expected_ts = list(pd.date_range(start=start, end=end, freq="15min", inclusive="left"))
    assert out["timestamp"].tolist() == expected_ts
    assert out["value"].tolist() == [1.0, 1.0, 1.0, 1.0]
