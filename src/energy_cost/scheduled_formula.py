import datetime as dt
from enum import StrEnum

import pandas as pd
from pydantic import BaseModel, Field, RootModel, model_validator

from energy_cost.price_formula import PriceFormula


class DayOfWeek(StrEnum):
    MONDAY = "monday"
    TUESDAY = "tuesday"
    WEDNESDAY = "wednesday"
    THURSDAY = "thursday"
    FRIDAY = "friday"
    SATURDAY = "saturday"
    SUNDAY = "sunday"


# Maps DayOfWeek to the integer used by pandas/Python's weekday() (Mon=0 … Sun=6).
_PANDAS_DAYOFWEEK: dict[DayOfWeek, int] = {
    DayOfWeek.MONDAY: 0,
    DayOfWeek.TUESDAY: 1,
    DayOfWeek.WEDNESDAY: 2,
    DayOfWeek.THURSDAY: 3,
    DayOfWeek.FRIDAY: 4,
    DayOfWeek.SATURDAY: 5,
    DayOfWeek.SUNDAY: 6,
}


class TimeRange(BaseModel):
    """A half-open time interval ``[start, end)`` within a single calendar day.

    Midnight-spanning ranges (e.g. 22:00 → 06:00) are **not** supported; use
    two separate :class:`TimeRange` objects instead.
    """

    start: dt.time
    end: dt.time

    @model_validator(mode="after")
    def _no_midnight_spanning(self) -> "TimeRange":
        if self.start >= self.end:
            raise ValueError(
                f"TimeRange start ({self.start}) must be strictly before end ({self.end}). "
                "Midnight-spanning ranges are not supported; split into two ranges instead."
            )
        return self

    def matches(self, times: pd.Series) -> pd.Series:
        """Return a boolean Series: ``True`` where *times* falls in ``[start, end)``."""
        return (times >= self.start) & (times < self.end)


class WeekSchedule(BaseModel):
    """Describes which timestamps belong to a particular pricing window.

    Parameters
    ----------
    days:
        Weekdays on which this schedule is active.  Defaults to all seven
        days when omitted.
    time_ranges:
        One or more ``[start, end)`` windows within the day.  When the list
        is empty (the default) **all hours** of the specified days match.
        When multiple ranges are given, any match is sufficient (OR logic).
    """

    days: list[DayOfWeek] = Field(default_factory=lambda: list(DayOfWeek))
    time_ranges: list[TimeRange] = Field(default_factory=list)

    def matches(self, timestamps: pd.Series) -> pd.Series:
        """Return a boolean Series for *timestamps* that satisfy this schedule."""
        day_ints = {_PANDAS_DAYOFWEEK[d] for d in self.days}
        day_mask: pd.Series = timestamps.dt.dayofweek.isin(day_ints)

        if not self.time_ranges:
            return day_mask

        times: pd.Series = timestamps.dt.time
        time_mask = pd.Series(False, index=timestamps.index)
        for time_range in self.time_ranges:
            time_mask = time_mask | time_range.matches(times)

        return day_mask & time_mask


class ScheduledPriceFormula(PriceFormula):
    """A :class:`~energy_cost.price_formula.PriceFormula` with an optional
    :class:`WeekSchedule` that describes when it applies.
    """

    when: WeekSchedule | None = None

    def get_values(self, start: dt.datetime, end: dt.datetime, resolution: dt.timedelta) -> pd.DataFrame:
        """Get cost values masked by ``when``; non-matching timestamps become NaN."""
        df = super().get_values(start, end, resolution)
        if self.when is not None:
            mask = self.when.matches(df["timestamp"])
            df["value"] = df["value"].where(mask)
        return df


class ScheduledPriceFormulas(RootModel[list[ScheduledPriceFormula]]):
    """A list of :class:`ScheduledPriceFormula` entries coalesced into a single price
    formula that varies by day-of-week and/or time-of-day.

    Behaves as a list: iterate, index, and measure length directly on the instance.
    Each :class:`FormulaSchedule` contributes values only for its matching
    timestamps; the first defined value per timestamp wins (coalesce semantics).
    """

    def __iter__(self):  # type: ignore[override]
        return iter(self.root)

    def __len__(self) -> int:
        return len(self.root)

    def __getitem__(self, index: int) -> ScheduledPriceFormula:
        return self.root[index]

    def get_values(self, start: dt.datetime, end: dt.datetime, resolution: dt.timedelta) -> pd.DataFrame:
        """Get the cost values for the given time range and resolution in €/MWh.

        Each :class:`FormulaSchedule` contributes only its matching timestamps
        (non-matching slots are NaN in its output); the first defined value per
        timestamp wins (coalesce semantics).
        """
        timestamps = pd.date_range(start=start, end=end, freq=resolution, inclusive="left")
        result: pd.Series = pd.Series(float("nan"), index=timestamps, dtype=float)
        for schedule in self.root:
            values = schedule.get_values(start, end, resolution).set_index("timestamp")["value"]
            result = result.combine_first(values)
        return pd.DataFrame({"timestamp": timestamps, "value": result.to_numpy()})
