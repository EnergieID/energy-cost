from __future__ import annotations

import datetime as dt
from enum import StrEnum

import pandas as pd
from pydantic import BaseModel, Field, model_validator

from energy_cost.resolution import Resolution, to_pandas_freq

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


class WhenClause(BaseModel):
    days: list[DayOfWeek] = Field(default_factory=lambda: list(DayOfWeek))
    start: dt.time = dt.time(0, 0, 0)
    end: dt.time | None = None

    @model_validator(mode="after")
    def _validate_time_range(self) -> WhenClause:
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


class ScheduledFormula(BaseModel):
    when: list[WhenClause] | None = None
    formula: Formula

    def get_values(
        self,
        start: dt.datetime,
        end: dt.datetime,
        resolution: Resolution,
    ) -> pd.DataFrame:
        df = self.formula.get_values(start, end, resolution)
        if self.when is not None:
            mask = pd.Series(False, index=df.index)
            for clause in self.when:
                mask = mask | clause.matches(df["timestamp"])
            df["value"] = df["value"].where(mask)
        return df


class ScheduledFormulas(BaseModel, Formula):
    kind: str = "scheduled"
    schedule: list[ScheduledFormula] = Field(default_factory=list)

    def get_values(
        self,
        start: dt.datetime,
        end: dt.datetime,
        resolution: Resolution,
    ) -> pd.DataFrame:
        timestamps = pd.date_range(start=start, end=end, freq=to_pandas_freq(resolution), inclusive="left")
        result: pd.Series = pd.Series(float("nan"), index=timestamps, dtype=float)
        for schedule in self.schedule:
            values = schedule.get_values(start, end, resolution).set_index("timestamp")["value"]
            result = result.combine_first(values)
        return pd.DataFrame({"timestamp": timestamps, "value": result.to_numpy()})
