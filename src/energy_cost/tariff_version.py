import datetime as dt
from collections.abc import Callable
from datetime import UTC
from typing import Annotated, Any

import pandas as pd
from pydantic import BeforeValidator, Field, TypeAdapter, field_validator

from energy_cost.versioning import Versioned

from .formula import Formula
from .meter import CostGroup, Meter, MeterType
from .resolution import Resolution, align_datetime_to_tz

_METER_KEYS = {e.value for e in MeterType}
_formula_adapter: TypeAdapter[Formula] = TypeAdapter(Formula)


def _coerce_named_formulas(value: Any) -> Any:
    """Allow a bare Formula dict to be shorthand for ``{total: it}``."""
    try:
        return {"total": _formula_adapter.validate_python(value)}
    except Exception:
        return value


NamedFormulas = Annotated[
    dict[str, Formula],
    BeforeValidator(_coerce_named_formulas),
]


class TariffVersion(Versioned):
    @field_validator("end")
    @classmethod
    def end_must_be_none(cls, v: dt.datetime | None) -> None:
        if v is not None:
            raise ValueError(
                "TariffVersion does not support an end date; It always applies until the next version starts."
            )
        return v

    injection: NamedFormulas = Field(default_factory=dict)
    consumption: NamedFormulas = Field(default_factory=dict)
    capacity: NamedFormulas = Field(default_factory=dict)
    fixed: NamedFormulas = Field(default_factory=dict)

    def _combine_energy_formulas(
        self,
        cost_group: CostGroup,
        get_df: Callable[[Formula], pd.DataFrame],
    ) -> pd.DataFrame | None:
        resolved = getattr(self, cost_group.value)
        series: dict[str, pd.Series] = {}
        for cost_type, formula in resolved.items():
            df = get_df(formula)
            if not df.empty:
                series[cost_type] = df.set_index("timestamp")["value"]

        if not series:
            return None

        result = pd.DataFrame(series)
        result.index.name = "timestamp"
        result = result.reset_index()

        cost_columns = [col for col in result.columns if col not in ("timestamp", "total")]
        if cost_columns:
            result["total"] = result[cost_columns].sum(axis=1, skipna=False)
        return result

    def get_values(
        self,
        start: dt.datetime,
        end: dt.datetime,
        output_resolution: Resolution,
        cost_group: CostGroup,
        timezone: dt.tzinfo = UTC,
    ) -> pd.DataFrame | None:
        """Get energy cost rates in €/MWh. Returns None if no formulas are configured."""
        return self._combine_energy_formulas(
            cost_group,
            lambda formula: formula.get_values(start, end, output_resolution, timezone),
        )

    def apply(
        self,
        consumption: Meter,
        injection: Meter | None,
        start: dt.datetime,
        end: dt.datetime,
        output_resolution: Resolution,
        timezone: dt.tzinfo = UTC,
        binning_anchor: dt.datetime | None = None,
    ) -> pd.DataFrame | None:
        start = align_datetime_to_tz(start, timezone)
        end = align_datetime_to_tz(end, timezone)

        results = []
        for cost_group in CostGroup:
            meter = injection if cost_group == CostGroup.INJECTION else consumption
            if meter is None:
                continue

            result = self._combine_energy_formulas(
                cost_group,
                lambda formula, meter=meter: formula.apply(
                    meter,
                    timezone=timezone,
                    start=start,
                    end=end,
                    output_resolution=output_resolution,
                    binning_anchor=binning_anchor,
                ),
            )
            if result is not None:
                result = result.set_index("timestamp")
                result.columns = pd.MultiIndex.from_tuples([(cost_group, c) for c in result.columns])
                results.append(result)

        if not results:
            return None

        result = pd.concat(results, axis=1, sort=True)
        result[("total", "total")] = result[[col for col in result.columns if col[-1] == "total"]].sum(
            axis=1, skipna=False
        )

        return result.reset_index()
