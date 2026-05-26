from __future__ import annotations

import datetime as dt

import isodate
import pandas as pd
import pytest
from pydantic import ValidationError

from energy_cost.formula import DayOfWeek, IndexFormula, ScheduledFormula, ScheduledFormulas, WhenClause
from energy_cost.formula.scheduled import maximal_resolution
from energy_cost.meter import Meter, TimeseriesFrame


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
            "timestamp": pd.to_datetime(["2026-03-16 00:00:00", "2026-03-17 00:00:00"], utc=True),
            "value": [3.0, 4.0],
        }
    )
    meter = Meter(power=TimeseriesFrame(data))

    out = formula.apply(meter, meter.power.start, meter.power.end, output_resolution=dt.timedelta(days=1))

    assert out["value"].tolist() == [15.0, 8.0]


def test_scheduled_formulas_apply_schedule_before_aggregating_to_output_resolution() -> None:
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
            "timestamp": pd.date_range("2026-03-16", periods=48, freq="h", tz=dt.UTC),
            "value": 1.0,
        }
    )
    meter = Meter(power=TimeseriesFrame(data))

    out = formula.apply(meter, meter.power.start, meter.power.end, output_resolution=dt.timedelta(days=2))

    assert out["value"].tolist() == [24 * 5.0 + 24 * 2.0]
    formula = ScheduledFormulas(
        schedule=[
            ScheduledFormula(
                when=[WhenClause(days=[DayOfWeek.MONDAY], start=dt.time(1, 30), end=dt.time(13, 30))],
                formula=IndexFormula(constant_cost=5.0),
            ),
            ScheduledFormula(formula=IndexFormula(constant_cost=2.0)),
        ]
    )
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-03-16", periods=192, freq="15min", tz=dt.UTC),
            "value": 0.25,
        }
    )
    meter = Meter(power=TimeseriesFrame(data))

    out = formula.apply(meter, meter.power.start, meter.power.end, output_resolution=dt.timedelta(days=2))

    assert out["value"].tolist() == [12 * 5.0 + 36 * 2.0]


def test_scheduled_formulas_spread_out_to_resolution_to_culculate_partial_matches() -> None:
    formula = ScheduledFormulas(
        schedule=[
            ScheduledFormula(
                when=[WhenClause(days=[DayOfWeek.THURSDAY])],
                formula=IndexFormula(constant_cost=5.0),
            ),
            ScheduledFormula(formula=IndexFormula(constant_cost=2.0)),
        ]
    )
    data = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01 00:00:00", "2026-02-01 00:00:00"], utc=True),
            "value": [7.0, 9.0],
        }
    )

    meter = Meter(power=TimeseriesFrame(data))
    out = formula.apply(meter, meter.power.start, meter.power.end, output_resolution=isodate.parse_duration("P1M"))

    # January has 5 thursdays, February has 4
    assert out["value"].tolist() == pytest.approx(
        [5 / 31 * 5.0 * 7.0 + 26 / 31 * 2.0 * 7.0, 4 / 28 * 5.0 * 9.0 + 24 / 28 * 2.0 * 9.0]
    )


@pytest.mark.parametrize(
    ("time", "expected"),
    [
        (dt.time(0, 0, 0), dt.timedelta(days=1)),
        (dt.time(12, 0, 0), dt.timedelta(hours=12)),
        (dt.time(18, 0, 0), dt.timedelta(hours=6)),
        (dt.time(1, 0, 0), dt.timedelta(hours=1)),
        (dt.time(8, 30, 0), dt.timedelta(minutes=30)),
        (dt.time(2, 45, 0), dt.timedelta(minutes=15)),
        (dt.time(3, 55, 0), dt.timedelta(minutes=5)),
        (dt.time(9, 7, 0), dt.timedelta(minutes=1)),
        (dt.time(15, 0, 17), dt.timedelta(seconds=1)),
    ],
)
def test_maximal_resolution(time: dt.time, expected: dt.timedelta) -> None:
    assert maximal_resolution(time) == expected


def test_scheduled_formulas_maximal_resolution_returns_none_for_empty_schedule() -> None:
    """ """
    assert ScheduledFormulas(schedule=[]).maximal_resolution() is None
