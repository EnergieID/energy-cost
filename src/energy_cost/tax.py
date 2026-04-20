import datetime as dt
from datetime import UTC
from pathlib import Path
from typing import Literal

import pandas as pd
import yaml
from pydantic import BaseModel, Field

from energy_cost.versioning import Versioned, VersionedCollection

from .meter import CostGroup, TariffCategory
from .resolution import Resolution, align_datetime_to_tz, align_timestamps_to_tz, detect_resolution_and_range

_WILDCARD: Literal["*"] = "*"

type ColumnPattern = tuple[TariffCategory | Literal["*"], CostGroup | Literal["*"], str]


def _matches_pattern(pattern: ColumnPattern, column: ColumnPattern) -> bool:
    return all(p in (_WILDCARD, c) for p, c in zip(pattern, column, strict=True))


def _specificity(pattern: ColumnPattern) -> int:
    return sum((1 << i) for i, p in enumerate(pattern) if p != _WILDCARD)


class TaxRule(BaseModel):
    rate: float = Field(ge=0.0, le=1.0)
    columns: list[ColumnPattern]


class TaxVersion(Versioned):
    default: float = Field(ge=0.0, le=1.0)
    rates: list[TaxRule] = Field(default_factory=list)

    @property
    def sorted_rates(self) -> list[tuple[float, ColumnPattern]]:
        rates = self.rates + [TaxRule(rate=self.default, columns=[(_WILDCARD, _WILDCARD, _WILDCARD)])]
        result: list[tuple[float, ColumnPattern]] = [(rule.rate, pattern) for rule in rates for pattern in rule.columns]
        return sorted(result, key=lambda x: _specificity(x[1]), reverse=True)

    def apply(self, data: pd.DataFrame, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        """Compute total tax for each row in *data* within [start, end).

        *data* has a ``timestamp`` column and a 3-level MultiIndex on the
        remaining columns: ``(TariffCategory, CostGroup, cost_type)``.

        Returns a DataFrame with a ``timestamp`` column and a
        ``("taxes", "total", "total")`` column.
        """
        data = data[(data["timestamp"] >= start) & (data["timestamp"] < end)].copy()
        # Work on a mutable copy of totals so we can subtract handled amounts
        remaining = data.copy().set_index("timestamp")
        tax = pd.Series(0.0, index=remaining.index)

        for rate, pattern in self.sorted_rates:
            total_pattern = _total_pattern(pattern)
            for col in remaining.columns:
                assert isinstance(col, tuple) and len(col) == 3
                if _matches_pattern(total_pattern, col):
                    amount = remaining[col]
                    tax = tax + amount * rate
                    for i in range(2, -1, -1):
                        if col[i] == "total":
                            # remove all cols that have been summed into this total column
                            remaining = remaining.loc[:, [c[:i] != col[:i] for c in remaining.columns]]
                        else:
                            # subtract the amount from wider totals it contributes to
                            total_col = col[:i] + ("total",) * (3 - i)  # type: ignore[tuple-item]
                            remaining[total_col] -= amount

        _col = (TariffCategory.TAXES, CostGroup.TOTAL, "total")
        return pd.DataFrame({"timestamp": tax.index, _col: tax.to_numpy()})


def _total_pattern(pattern: ColumnPattern) -> ColumnPattern:
    result = list(pattern)
    for i in range(len(result) - 1, -1, -1):
        if result[i] == _WILDCARD:
            result[i] = "total"
        else:
            break
    return tuple(result)  # type: ignore[return-value]


class Tax(VersionedCollection[TaxVersion]):
    @classmethod
    def from_yaml(cls, path: str | Path) -> "Tax":
        """Load a tax definition from YAML."""
        with Path(path).open(encoding="utf-8") as file:
            raw_data = yaml.safe_load(file)
        return cls.model_validate({"versions": raw_data})

    def apply(
        self,
        data: pd.DataFrame,
        start: dt.datetime | None = None,
        end: dt.datetime | None = None,
        resolution: Resolution | None = None,
        timezone: dt.tzinfo = UTC,
    ) -> pd.DataFrame | None:
        if start is None or end is None:
            detected_start, detected_end, _ = detect_resolution_and_range(data, resolution)
            start = align_datetime_to_tz(detected_start, timezone)
            end = align_datetime_to_tz(detected_end, timezone)

        data = align_timestamps_to_tz(data, timezone)

        return self.collect_version_frames(
            lambda version, seg_start, seg_end: version.apply(data, start=seg_start, end=seg_end), start, end, timezone
        )
