from datetime import UTC, datetime, tzinfo
from typing import TYPE_CHECKING, Literal

from pandas.core.api import DataFrame as DataFrame
from pydantic import Field

from energy_cost.meter import Meter, MeterType
from energy_cost.resolution import Resolution

from .base import FormulaBase

if TYPE_CHECKING:
    from .formula import Formula


class MeterTypeFormula(FormulaBase):
    """Formula that applies a different formula based on the meter type."""

    kind: Literal["meter_type"] = "meter_type"
    by_meter_type: dict[MeterType | Literal["default"], Formula] = Field(default_factory=dict)

    def apply(
        self,
        meter: Meter,
        start: datetime,
        end: datetime,
        output_resolution: Resolution,
        timezone: tzinfo = UTC,
        binning_anchor: datetime | None = None,
    ) -> DataFrame:
        formula = self.by_meter_type.get(meter.type, self.by_meter_type.get("default"))
        if formula is None:
            raise ValueError(f"No formula configured for meter type {meter.type} and no default formula provided.")

        return formula.apply(meter, start, end, output_resolution, timezone, binning_anchor)
