import datetime as dt
from enum import StrEnum

import pandas as pd
from pydantic import BaseModel, Field, RootModel, model_validator

from energy_cost.price_formula import PriceFormula
from energy_cost.resolution import Resolution, to_pandas_freq


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


class WhenClause(BaseModel):
    """A single scheduling clause: matches timestamps on specified days within a time window.

    All fields are optional:

    - ``days`` defaults to all seven days.
    - ``start`` defaults to ``00:00:00`` (beginning of the day).
    - ``end`` defaults to ``None`` (no upper bound; matches until end of day).

    The time interval is half-open: ``[start, end)``.
    Midnight-spanning ranges (e.g. 22:00 → 06:00) are **not** supported; use two
    separate :class:`WhenClause` entries instead.
    """

    days: list[DayOfWeek] = Field(default_factory=lambda: list(DayOfWeek))
    start: dt.time = dt.time(0, 0, 0)
    end: dt.time | None = None  # None means no upper bound (rest of day)

    @model_validator(mode="after")
    def _validate_time_range(self) -> "WhenClause":
        if self.end is not None and self.start >= self.end:
            raise ValueError(
                f"WhenClause start ({self.start}) must be strictly before end ({self.end}). "
                "Midnight-spanning ranges are not supported; split into two clauses instead."
            )
        return self

    def matches(self, timestamps: pd.Series) -> pd.Series:
        """Return a boolean Series: ``True`` where *timestamps* match this clause."""
        day_ints = {_PANDAS_DAYOFWEEK[d] for d in self.days}
        day_mask: pd.Series = timestamps.dt.dayofweek.isin(day_ints)

        times: pd.Series = timestamps.dt.time
        time_mask: pd.Series = times >= self.start
        if self.end is not None:
            time_mask = time_mask & (times < self.end)

        return day_mask & time_mask


class ScheduledPriceFormula(PriceFormula):
    """A :class:`~energy_cost.price_formula.PriceFormula` with an optional list of
    :class:`WhenClause` entries that describes when it applies.

    A timestamp matches if it satisfies **any** of the clauses (OR logic).
    When ``when`` is ``None`` or omitted, the formula applies at all times.
    """

    when: list[WhenClause] | None = None

    def get_values(self, start: dt.datetime, end: dt.datetime, resolution: Resolution) -> pd.DataFrame:
        """Get cost values masked by ``when``; non-matching timestamps become NaN."""
        df = super().get_values(start, end, resolution)
        if self.when is not None:
            mask = pd.Series(False, index=df.index)
            for clause in self.when:
                mask = mask | clause.matches(df["timestamp"])
            df["value"] = df["value"].where(mask)
        return df


class ScheduledPriceFormulas(RootModel[list[ScheduledPriceFormula]]):
    """A list of :class:`ScheduledPriceFormula` entries coalesced into a single price
    formula that varies by day-of-week and/or time-of-day.

    Each :class:`ScheduledPriceFormula` contributes values only for its matching
    timestamps; the first defined value per timestamp wins (coalesce semantics).
    """

    def get_values(self, start: dt.datetime, end: dt.datetime, resolution: Resolution) -> pd.DataFrame:
        """Get the cost values for the given time range and resolution in €/MWh.

        Each :class:`ScheduledPriceFormula` contributes only its matching timestamps
        (non-matching slots are NaN in its output); the first defined value per
        timestamp wins (coalesce semantics).
        """
        timestamps = pd.date_range(start=start, end=end, freq=to_pandas_freq(resolution), inclusive="left")
        result: pd.Series = pd.Series(float("nan"), index=timestamps, dtype=float)
        for schedule in self.root:
            values = schedule.get_values(start, end, resolution).set_index("timestamp")["value"]
            result = result.combine_first(values)
        return pd.DataFrame({"timestamp": timestamps, "value": result.to_numpy()})
