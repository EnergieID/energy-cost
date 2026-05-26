from __future__ import annotations

import datetime as dt
from collections.abc import Callable
from datetime import UTC
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from energy_cost.meter import Meter
from energy_cost.resolution import (
    Resolution,
    align_datetime_to_tz,
    find_common_divisor,
    redistribute_to_resolution,
    to_pandas_freq,
)

from .base import FormulaBase

if TYPE_CHECKING:
    from .formula import Formula


class DayOfWeek(StrEnum):
    MONDAY = "monday"
    TUESDAY = "tuesday"
    WEDNESDAY = "wednesday"
    THURSDAY = "thursday"
    FRIDAY = "friday"
    SATURDAY = "saturday"
    SUNDAY = "sunday"


_PANDAS_DAYOFWEEK: dict[DayOfWeek, int] = {
    DayOfWeek.MONDAY: 0,
    DayOfWeek.TUESDAY: 1,
    DayOfWeek.WEDNESDAY: 2,
    DayOfWeek.THURSDAY: 3,
    DayOfWeek.FRIDAY: 4,
    DayOfWeek.SATURDAY: 5,
    DayOfWeek.SUNDAY: 6,
}


_CANDIDATE_RESOLUTIONS = [
    dt.timedelta(days=1),
    dt.timedelta(hours=12),
    dt.timedelta(hours=6),
    dt.timedelta(hours=1),
    dt.timedelta(minutes=30),
    dt.timedelta(minutes=15),
    dt.timedelta(minutes=5),
    dt.timedelta(minutes=1),
    dt.timedelta(seconds=1),
]


def maximal_resolution(time: dt.time) -> dt.timedelta:
    total_seconds = time.hour * 3600 + time.minute * 60 + time.second
    return next(r for r in _CANDIDATE_RESOLUTIONS if total_seconds % int(r.total_seconds()) == 0)


class WhenClause(BaseModel):
    days: list[DayOfWeek] = Field(default_factory=lambda: list(DayOfWeek))
    start: dt.time = dt.time(0, 0, 0)
    end: dt.time | None = None

    @model_validator(mode="after")
    def _validate_time_range(self) -> WhenClause:
        self.start = self.start.replace(tzinfo=None)
        self.end = self.end.replace(tzinfo=None) if self.end is not None else None
        if self.end is not None and self.start >= self.end:
            raise ValueError(
                f"WhenClause start ({self.start}) must be strictly before end ({self.end}). "
                "Midnight-spanning ranges are not supported; split into two clauses instead."
            )
        return self

    def matches(self, timestamps: pd.Series) -> pd.Series:
        day_ints = {_PANDAS_DAYOFWEEK[d] for d in self.days}
        day_mask: pd.Series = timestamps.dt.dayofweek.isin(day_ints)

        times: pd.Series = timestamps.dt.time
        time_mask: pd.Series = times >= self.start
        if self.end is not None:
            time_mask = time_mask & (times < self.end)

        return day_mask & time_mask

    def maximal_resolution(self) -> dt.timedelta:
        """Return the maximal resolution allowed by this clause's time range."""
        time_res = maximal_resolution(self.start)
        if self.end is not None:
            time_res = min(time_res, maximal_resolution(self.end))
        return time_res


class ScheduledFormula(FormulaBase):
    when: list[WhenClause] | None = None
    formula: Formula

    def _filter_by_when(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.when is None:
            return df

        mask = pd.Series(False, index=df.index)
        for clause in self.when:
            mask = mask | clause.matches(df["timestamp"])
        df["value"] = df["value"].where(mask)
        return df

    def maximal_resolution(self) -> dt.timedelta | None:
        """Return the maximal resolution allowed by this formula's schedule."""
        if self.when is None:
            return None
        return min(clause.maximal_resolution() for clause in self.when)

    def get_values(
        self,
        start: dt.datetime,
        end: dt.datetime,
        output_resolution: Resolution,
        timezone: dt.tzinfo = UTC,
    ) -> pd.DataFrame:
        values = self.formula.get_values(start, end, output_resolution, timezone)
        return self._filter_by_when(values)

    def apply(
        self,
        meter: Meter,
        start: dt.datetime,
        end: dt.datetime,
        output_resolution: Resolution,
        timezone: dt.tzinfo = UTC,
        binning_anchor: dt.datetime | None = None,
    ) -> pd.DataFrame:
        values = self.formula.apply(meter, start, end, output_resolution, timezone, binning_anchor)
        return self._filter_by_when(values)


class ScheduledFormulas(FormulaBase):
    kind: Literal["scheduled"] = "scheduled"
    schedule: list[ScheduledFormula] = Field(default_factory=list)

    def maximal_resolution(self) -> dt.timedelta | None:
        """Return the maximal resolution allowed by this formula's schedule."""
        if not self.schedule:
            return None
        resolutions = [formula.maximal_resolution() for formula in self.schedule]
        resolutions = [r for r in resolutions if r is not None]
        return min(resolutions) if resolutions else None

    def _combine(
        self,
        start: dt.datetime,
        end: dt.datetime,
        output_resolution: Resolution,
        timezone: dt.tzinfo,
        func: Callable[[ScheduledFormula, Resolution], pd.DataFrame],
        binning_anchor: dt.datetime | None = None,
    ) -> pd.DataFrame:
        start = align_datetime_to_tz(start, timezone)
        end = align_datetime_to_tz(end, timezone)

        intermediate_resolution = self.maximal_resolution() or output_resolution
        if intermediate_resolution.total_seconds() > output_resolution.total_seconds():
            intermediate_resolution = find_common_divisor(intermediate_resolution, output_resolution)
        timestamps = pd.date_range(start=start, end=end, freq=to_pandas_freq(intermediate_resolution), inclusive="left")
        result: pd.Series = pd.Series(float("nan"), index=timestamps, dtype=float)
        for schedule in self.schedule:
            values = func(schedule, intermediate_resolution).set_index("timestamp")["value"]
            result = result.combine_first(values)
        merged = pd.DataFrame({"timestamp": timestamps, "value": result.to_numpy()})
        return redistribute_to_resolution(
            merged, intermediate_resolution, output_resolution, start, end, binning_anchor=binning_anchor
        )

    def get_values(
        self,
        start: dt.datetime,
        end: dt.datetime,
        output_resolution: Resolution,
        timezone: dt.tzinfo = UTC,
    ) -> pd.DataFrame:
        return self._combine(
            start,
            end,
            output_resolution,
            timezone,
            lambda schedule, intermediate_resolution: schedule.get_values(
                start, end, intermediate_resolution, timezone
            ),
        )

    def apply(
        self,
        meter: Meter,
        start: dt.datetime,
        end: dt.datetime,
        output_resolution: Resolution,
        timezone: dt.tzinfo = UTC,
        binning_anchor: dt.datetime | None = None,
    ) -> pd.DataFrame:
        return self._combine(
            start,
            end,
            output_resolution,
            timezone,
            lambda schedule, intermediate_resolution: schedule.apply(
                meter, start, end, intermediate_resolution, timezone, binning_anchor
            ),
            binning_anchor,
        )
