import datetime as dt

from pydantic import BaseModel

from .fractional_periods import Period


class PeriodicCost(BaseModel):
    """A cost that applies for a fixed period of time, regardless of energy usage."""

    period: Period
    cost: float

    def get_cost_for_interval(self, start: dt.datetime, end: dt.datetime) -> float:
        """Get the cost for a given time interval, prorating for partial periods."""
        return self.cost * self.period.fractional_periods(start, end)
