from __future__ import annotations

import datetime as dt

import pytest

from energy_cost.fractional_periods import MonthPeriod, Period, YearPeriod


def test_hourly_and_daily_periods_use_simple_time_fractions() -> None:
    start = dt.datetime(2025, 1, 1, 0, 0)

    assert Period.HOURLY.fractional_periods(start, start + dt.timedelta(minutes=90)) == pytest.approx(1.5)
    assert Period.DAILY.fractional_periods(start, start + dt.timedelta(hours=36)) == pytest.approx(1.5)


def test_month_period_helpers_and_fractional_periods() -> None:
    period = MonthPeriod()
    start = dt.datetime(2025, 1, 15, 12, 0)

    assert period.period_start(start) == dt.datetime(2025, 1, 1, 0, 0)
    assert period.next_period(start) == dt.datetime(2025, 2, 1, 0, 0)
    assert period.complete_periods_between(dt.datetime(2025, 2, 1), dt.datetime(2025, 4, 1)) == 2

    assert period.fractional_periods(start, start) == 0.0
    assert period.fractional_periods(start, start - dt.timedelta(days=1)) == 0.0

    same_month_end = dt.datetime(2025, 1, 20, 12, 0)
    assert period.fractional_periods(start, same_month_end) == pytest.approx(5 / 31)

    cross_month_start = dt.datetime(2025, 1, 16, 0, 0)
    cross_month_end = dt.datetime(2025, 3, 16, 0, 0)
    assert period.fractional_periods(cross_month_start, cross_month_end) == pytest.approx(2.0)


def test_year_period_helpers_and_fractional_periods() -> None:
    period = YearPeriod()
    leap_year_start = dt.datetime(2024, 7, 2, 0, 0)

    assert period.period_start(leap_year_start) == dt.datetime(2024, 1, 1, 0, 0)
    assert period.next_period(leap_year_start) == dt.datetime(2025, 1, 1, 0, 0)
    assert period.complete_periods_between(dt.datetime(2025, 1, 1), dt.datetime(2028, 1, 1)) == 3

    same_year_start = dt.datetime(2025, 4, 1, 0, 0)
    same_year_end = dt.datetime(2025, 10, 1, 0, 0)
    assert period.fractional_periods(same_year_start, same_year_end) == pytest.approx(183 / 365)

    assert period.fractional_periods(dt.datetime(2024, 1, 1), dt.datetime(2026, 1, 1)) == pytest.approx(2.0)
