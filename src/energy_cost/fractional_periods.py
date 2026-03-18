"""Time-based fixed costs for energy tariffs.

Fixed costs are periodic fees (hourly, daily, monthly, yearly) that do not depend on
energy consumption.  Partial periods are prorated: the number of periods in the queried
range is computed as a float and multiplied by the per-period amount.

In a time-series the total cost is spread equally across all resolution steps.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime
from enum import StrEnum

from dateutil.relativedelta import relativedelta


class Period(StrEnum):
    HOURLY = "hourly"
    DAILY = "daily"
    MONTHLY = "monthly"
    YEARLY = "yearly"


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
        return (self.next_period(dt) - self.period_start(dt)).total_seconds()

    def fractional_periods(self, start: datetime, end: datetime) -> float:
        """Calendar-aware fractional periods for a [start, end) interval."""
        if start >= end:
            return 0.0

        period_start_of_start = self.period_start(start)
        period_start_of_end = self.period_start(end)

        if period_start_of_start == period_start_of_end:
            return (end - start).total_seconds() / self.period_seconds(start)

        next_p = self.next_period(start)
        frac_start = (next_p - start).total_seconds() / self.period_seconds(start)
        complete = self.complete_periods_between(next_p, period_start_of_end)
        frac_end = (end - period_start_of_end).total_seconds() / self.period_seconds(end)

        return frac_start + complete + frac_end


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


PERIOD_FRACTION_FUNCTIONS: dict[Period, Callable[[datetime, datetime], float]] = {
    Period.HOURLY: lambda start, end: (end - start).total_seconds() / 3600,
    Period.DAILY: lambda start, end: (end - start).total_seconds() / 86400,
    Period.MONTHLY: MonthPeriod().fractional_periods,
    Period.YEARLY: YearPeriod().fractional_periods,
}


def fractional_periods(start: datetime, end: datetime, period: Period) -> float:
    """Calendar-aware fractional periods for a [start, end) interval."""
    return PERIOD_FRACTION_FUNCTIONS[period](start, end)
