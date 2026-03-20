from __future__ import annotations

import datetime as dt

import pytest

from energy_cost.fractional_periods import MonthPeriod, Period, YearPeriod


def test_hourly_period_returns_fraction_for_partial_hours() -> None:
    start = dt.datetime(2025, 1, 1, 0, 0)

    assert Period.HOURLY.fractional_periods(start, start + dt.timedelta(minutes=90)) == pytest.approx(1.5)


def test_daily_period_returns_fraction_for_partial_days() -> None:
    start = dt.datetime(2025, 1, 1, 0, 0)

    assert Period.DAILY.fractional_periods(start, start + dt.timedelta(hours=36)) == pytest.approx(1.5)


def test_month_period_start_truncates_datetime_to_first_day_of_month() -> None:
    period = MonthPeriod()

    assert period.period_start(dt.datetime(2025, 1, 15, 12, 0)) == dt.datetime(2025, 1, 1, 0, 0)


def test_month_period_next_period_returns_first_moment_of_following_month() -> None:
    period = MonthPeriod()

    assert period.next_period(dt.datetime(2025, 1, 15, 12, 0)) == dt.datetime(2025, 2, 1, 0, 0)


def test_month_period_counts_complete_months_between_month_boundaries() -> None:
    period = MonthPeriod()

    assert period.complete_periods_between(dt.datetime(2025, 2, 1), dt.datetime(2025, 4, 1)) == 2


def test_month_period_returns_zero_for_empty_interval() -> None:
    period = MonthPeriod()
    start = dt.datetime(2025, 1, 15, 12, 0)

    assert period.fractional_periods(start, start) == 0.0


def test_month_period_returns_zero_for_interval_with_end_before_start() -> None:
    period = MonthPeriod()
    start = dt.datetime(2025, 1, 15, 12, 0)

    assert period.fractional_periods(start, start - dt.timedelta(days=1)) == 0.0


def test_month_period_correctly_handles_periods_starting_and_ending_in_same_month() -> None:
    period = MonthPeriod()
    start = dt.datetime(2025, 1, 15, 12, 0)
    end = dt.datetime(2025, 1, 20, 12, 0)

    assert period.fractional_periods(start, end) == pytest.approx(5 / 31)


def test_month_period_correctly_detects_fractions_in_the_starting_and_ending_months() -> None:
    period = MonthPeriod()
    start = dt.datetime(2025, 1, 16, 0, 0)
    end = dt.datetime(2025, 3, 16, 0, 0)

    assert period.fractional_periods(start, end) == pytest.approx(2.0)


def test_year_period_start_truncates_datetime_to_first_day_of_year() -> None:
    period = YearPeriod()

    assert period.period_start(dt.datetime(2024, 7, 2, 0, 0)) == dt.datetime(2024, 1, 1, 0, 0)


def test_year_period_next_period_returns_first_moment_of_following_year() -> None:
    period = YearPeriod()

    assert period.next_period(dt.datetime(2024, 7, 2, 0, 0)) == dt.datetime(2025, 1, 1, 0, 0)


def test_year_period_counts_complete_years_between_year_boundaries() -> None:
    period = YearPeriod()

    assert period.complete_periods_between(dt.datetime(2025, 1, 1), dt.datetime(2028, 1, 1)) == 3


def test_year_period_correctly_handles_periods_starting_and_ending_in_same_year() -> None:
    period = YearPeriod()
    start = dt.datetime(2025, 4, 1, 0, 0)
    end = dt.datetime(2025, 10, 1, 0, 0)

    assert period.fractional_periods(start, end) == pytest.approx(183 / 365)


def test_year_period_correctly_detects_complete_years_between_boundaries() -> None:
    period = YearPeriod()

    assert period.fractional_periods(dt.datetime(2024, 1, 1), dt.datetime(2026, 1, 1)) == pytest.approx(2.0)
