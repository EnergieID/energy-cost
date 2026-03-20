import bisect
import datetime as dt
from pathlib import Path

import pandas as pd
import yaml
from pydantic import BaseModel

from .tariff_version import CostType, MeterType, PowerDirection, TariffVersion


class Tariff(BaseModel):
    versions: list[TariffVersion]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Tariff":
        """Load a tariff definition from YAML."""
        with Path(path).open(encoding="utf-8") as file:
            raw_data = yaml.safe_load(file)

        tariff = cls.model_validate({"versions": raw_data})
        tariff.versions.sort(key=lambda s: (s.start is not None, s.start or dt.datetime.min))
        return tariff

    def _find_active_versions(
        self,
        start: dt.datetime,
        end: dt.datetime,
    ) -> list[tuple[TariffVersion, dt.datetime, dt.datetime]]:
        """Return each segment that overlaps ``[start, end)`` together with the effective sub-range."""
        start_index = max(0, bisect.bisect_right(self.versions, start, key=lambda c: c.start) - 1)
        end_index = bisect.bisect_right(self.versions, end, key=lambda c: c.start)
        segments = self.versions[start_index:end_index]

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
        for version, seg_start, seg_end in self._find_active_versions(start, end):
            df = version.get_cost(seg_start, seg_end, resolution, meter_type, direction)
            result = df if result is None else pd.concat([result, df]).groupby("timestamp", as_index=False).sum()

        if result is None:
            raise ValueError(
                f"No active versions with formulas for meter type '{meter_type}' and direction '{direction}' found in tariff for the given time range."
            )

        return result.sort_values("timestamp").reset_index(drop=True)

    def get_periodic_cost(self, start: dt.datetime, end: dt.datetime) -> dict[str, float]:
        """Get the prorated periodic (fixed) costs for the given time interval.

        Returns a mapping of cost name to the total prorated cost for ``[start, end)``.
        """
        totals: dict[str, float] = {}
        for version, seg_start, seg_end in self._find_active_versions(start, end):
            for name, cost in version.get_periodic_cost(seg_start, seg_end).items():
                totals[name] = totals.get(name, 0.0) + cost
        return totals


__all__ = ["CostType", "MeterType", "PowerDirection", "Tariff", "TariffVersion"]
