"""Time-based fixed costs for energy tariffs.

Fixed costs are periodic fees (hourly, daily, monthly, yearly) that do not depend on
energy consumption.  Partial periods are prorated: the number of periods in the queried
range is computed as a float and multiplied by the per-period amount.

In a time-series the total cost is spread equally across all resolution steps.
"""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import UTC, datetime
from enum import StrEnum

from dateutil.relativedelta import relativedelta


class PeriodUnit(ABC):
    @abstractmethod
    def period_start(self, dt: datetime) -> datetime:
        """Truncate dt to the start of its containing period."""

    @abstractmethod
    def next_period(self, dt: datetime) -> datetime:
        """First moment of the period after the one containing dt."""

    @abstractmethod
    def complete_periods_between(self, start: datetime, end: datetime) -> int:
        """Number of complete periods between two period-start datetimes."""

    def period_seconds(self, dt: datetime) -> float:
        """Total seconds in the period containing dt."""
        return elapsed_seconds(self.period_start(dt), self.next_period(dt))

    def fractional_periods(self, start: datetime, end: datetime, timezone: dt.tzinfo = UTC) -> float:
        """Calendar-aware fractional periods for a [start, end) interval.

        Naive datetimes are localized to *timezone* (defaults to UTC).
        """
        # Localize naive datetimes to the specified timezone
        if not start.tzinfo:
            start = start.replace(tzinfo=timezone)
        if not end.tzinfo:
            end = end.replace(tzinfo=timezone)

        if start >= end:
            return 0.0

        period_start_of_start = self.period_start(start)
        period_start_of_end = self.period_start(end)

        if period_start_of_start == period_start_of_end:
            return elapsed_seconds(start, end) / self.period_seconds(start)

        next_p = self.next_period(start)
        frac_start = elapsed_seconds(start, next_p) / self.period_seconds(start)
        complete = self.complete_periods_between(next_p, period_start_of_end)
        frac_end = elapsed_seconds(period_start_of_end, end) / self.period_seconds(end)

        return frac_start + complete + frac_end


def elapsed_seconds(start: datetime, end: datetime) -> float:
    """Return elapsed seconds, using UTC for timezone-aware datetimes."""
    start = start.astimezone(UTC)
    end = end.astimezone(UTC)

    return (end - start).total_seconds()


class MonthPeriod(PeriodUnit):
    def period_start(self, dt: datetime) -> datetime:
        return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    def next_period(self, dt: datetime) -> datetime:
        return self.period_start(dt) + relativedelta(months=1)

    def complete_periods_between(self, start: datetime, end: datetime) -> int:
        delta = relativedelta(end, start)
        return delta.years * 12 + delta.months


class YearPeriod(PeriodUnit):
    def period_start(self, dt: datetime) -> datetime:
        return dt.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)

    def next_period(self, dt: datetime) -> datetime:
        return self.period_start(dt) + relativedelta(years=1)

    def complete_periods_between(self, start: datetime, end: datetime) -> int:
        return relativedelta(end, start).years


class DayPeriod(PeriodUnit):
    def period_start(self, dt: datetime) -> datetime:
        return dt.replace(hour=0, minute=0, second=0, microsecond=0)

    def next_period(self, dt: datetime) -> datetime:
        return self.period_start(dt) + relativedelta(days=1)

    def complete_periods_between(self, start: datetime, end: datetime) -> int:
        return (end.date() - start.date()).days


class HourPeriod(PeriodUnit):
    def period_start(self, dt: datetime) -> datetime:
        return dt.replace(minute=0, second=0, microsecond=0)

    def next_period(self, dt: datetime) -> datetime:
        return self.period_start(dt) + relativedelta(hours=1)

    def complete_periods_between(self, start: datetime, end: datetime) -> int:
        return int(elapsed_seconds(start, end) // 3600)


class Period(StrEnum):
    HOURLY = "hourly"
    DAILY = "daily"
    MONTHLY = "monthly"
    YEARLY = "yearly"

    def fractional_periods(self, start: datetime, end: datetime, timezone: dt.tzinfo = UTC) -> float:
        """Calendar-aware fractional periods for a [start, end) interval."""
        return PERIOD_FRACTION_FUNCTIONS[self](start, end, timezone)


PERIOD_FRACTION_FUNCTIONS: dict[Period, Callable[[datetime, datetime, dt.tzinfo], float]] = {
    Period.HOURLY: HourPeriod().fractional_periods,
    Period.DAILY: DayPeriod().fractional_periods,
    Period.MONTHLY: MonthPeriod().fractional_periods,
    Period.YEARLY: YearPeriod().fractional_periods,
}
