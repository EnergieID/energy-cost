import bisect
import datetime as dt
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import yaml
from pydantic import BaseModel, BeforeValidator, Field

from .periodic_cost import PeriodicCost
from .price_formula import PriceFormula
from .scheduled_formula import ScheduledPriceFormulas


class MeterType(StrEnum):
    SINGLE_RATE = "single_rate"
    TOU_PEAK = "tou_peak"
    TOU_OFFPEAK = "tou_offpeak"
    NIGHT_ONLY = "night_only"


class PowerDirection(StrEnum):
    CONSUMPTION = "consumption"
    INJECTION = "injection"


class CostType(StrEnum):
    ENERGY = "energy"
    # Combined Heat and Power certificates, a type of subsidy for efficient cogeneration plants
    CHP_CERTIFICATES = "chp_certificates"
    # Renewable Energy certificates, a type of subsidy for renewable energy generation
    RENEWABLE_CERTIFICATES = "renewable_certificates"


_COST_TYPE_VALUES = {e.value for e in CostType}
_METER_KEYS = {"all"} | {e.value for e in MeterType}


def _coerce_cost_type_formulas(value: Any) -> Any:
    """Allow a bare PriceFormula dict to be shorthand for ``{energy: it}``.

    Detected when the dict's keys are not all recognised CostType values.
    """
    if isinstance(value, dict) and not value.keys() <= _COST_TYPE_VALUES:
        return {CostType.ENERGY: value}
    return value


CostTypeFormulas = Annotated[
    dict[CostType, PriceFormula | ScheduledPriceFormulas],
    BeforeValidator(_coerce_cost_type_formulas),
]


def _coerce_meter_formulas(value: Any) -> Any:
    """Coerce shorthand MeterFormulas values.

    - CostType-keyed dict → ``{"all": it}``
    - Dict with keys not recognised as meter keys → bare PriceFormula → ``{"all": {energy: it}}``
    """
    if not isinstance(value, dict):
        return value
    if value.keys() <= _COST_TYPE_VALUES:
        return {"all": value}
    if not value.keys() <= _METER_KEYS:
        return {"all": {CostType.ENERGY: value}}
    return value


MeterFormulas = Annotated[
    dict[str, CostTypeFormulas],
    BeforeValidator(_coerce_meter_formulas),
]


class TariffSegment(BaseModel):
    start: dt.datetime
    injection: MeterFormulas = Field(default_factory=dict)
    consumption: MeterFormulas = Field(default_factory=dict)
    periodic: dict[str, PeriodicCost] = Field(default_factory=dict)

    def resolve_cost_formulas(
        self,
        meter_type: MeterType,
        direction: PowerDirection,
    ) -> dict[CostType, PriceFormula | ScheduledPriceFormulas]:
        """Merge the ``all`` defaults with any meter-type-specific overrides for a direction.

        Per-meter-type entries take precedence over ``all`` entries for the same cost type.
        """
        direction_formulas: MeterFormulas = getattr(self, direction.value)
        result: dict[CostType, PriceFormula | ScheduledPriceFormulas] = {}
        if "all" in direction_formulas:
            result.update(direction_formulas["all"])
        if meter_type.value in direction_formulas:
            result.update(direction_formulas[meter_type.value])
        return result

    def get_cost(
        self,
        start: dt.datetime,
        end: dt.datetime,
        resolution: dt.timedelta,
        meter_type: MeterType,
        direction: PowerDirection,
    ) -> pd.DataFrame:
        """Compute cost time series per cost type for the given interval.

        Returns a mapping of cost type to a DataFrame with ``timestamp`` and ``value`` columns.
        """
        resolved = self.resolve_cost_formulas(meter_type, direction)
        result: pd.DataFrame | None = None
        for cost_type in resolved:
            df = resolved[cost_type].get_values(start, end, resolution)
            if df.empty:
                continue
            df = df.rename(columns={"value": cost_type.value})
            result = df if result is None else result.merge(df, on="timestamp", how="outer")

        if result is None:
            raise ValueError(f"No formulas for meter type '{meter_type}' and direction '{direction}' found in tariff.")

        cost_columns = [col for col in result.columns if col != "timestamp"]
        result["total"] = result[cost_columns].sum(axis=1)
        return result

    def get_periodic_cost(self, start: dt.datetime, end: dt.datetime) -> dict[str, float]:
        """Get the prorated periodic costs for the given interval."""
        return {name: entry.get_cost_for_interval(start, end) for name, entry in self.periodic.items()}


class Tariff(BaseModel):
    segments: list[TariffSegment]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Tariff":
        """Load a tariff definition from YAML."""
        with Path(path).open(encoding="utf-8") as file:
            raw_data = yaml.safe_load(file)

        tariff = cls.model_validate({"segments": raw_data})
        tariff.segments.sort(key=lambda s: (s.start is not None, s.start or dt.datetime.min))
        return tariff

    def _find_active_segments(
        self,
        start: dt.datetime,
        end: dt.datetime,
    ) -> list[tuple["TariffSegment", dt.datetime, dt.datetime]]:
        """Return each segment that overlaps ``[start, end)`` together with the effective sub-range."""
        start_index = max(0, bisect.bisect_right(self.segments, start, key=lambda c: c.start) - 1)
        end_index = bisect.bisect_right(self.segments, end, key=lambda c: c.start)
        segments = self.segments[start_index:end_index]

        if not segments:
            return []

        starts = [max(segment.start, start) for segment in segments]
        ends = [segment.start for segment in segments[1:]] + [end]
        return list(zip(segments, starts, ends, strict=True))

    def get_cost(
        self,
        start: dt.datetime,
        end: dt.datetime,
        resolution: dt.timedelta = dt.timedelta(minutes=15),
        meter_type: MeterType = MeterType.SINGLE_RATE,
        direction: PowerDirection = PowerDirection.CONSUMPTION,
    ) -> pd.DataFrame:
        """Get the cost values for the given meter type and time range at the given resolution in €/MWh.

        Returns a DataFrame with a column per active cost type and a ``total`` column.
        """
        result: pd.DataFrame | None = None
        for segment, seg_start, seg_end in self._find_active_segments(start, end):
            df = segment.get_cost(seg_start, seg_end, resolution, meter_type, direction)
            result = df if result is None else pd.concat([result, df]).groupby("timestamp", as_index=False).sum()

        if result is None:
            raise ValueError(
                f"No active segments with formulas for meter type '{meter_type}' and direction '{direction}' found in tariff for the given time range."
            )

        return result.sort_values("timestamp").reset_index(drop=True)

    def get_periodic_cost(self, start: dt.datetime, end: dt.datetime) -> dict[str, float]:
        """Get the prorated periodic (fixed) costs for the given time interval.

        Returns a mapping of cost name to the total prorated cost for ``[start, end)``.
        """
        totals: dict[str, float] = {}
        for segment, seg_start, seg_end in self._find_active_segments(start, end):
            for name, cost in segment.get_periodic_cost(seg_start, seg_end).items():
                totals[name] = totals.get(name, 0.0) + cost
        return totals
