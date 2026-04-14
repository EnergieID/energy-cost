import bisect
import datetime as dt
from collections.abc import Callable
from datetime import UTC
from pathlib import Path
from typing import Literal

import isodate
import pandas as pd
import yaml
from pydantic import BaseModel

from .meter import CostGroup, Meter, MeterType, PowerDirection, as_single_meter
from .resolution import (
    Resolution,
    align_datetime_to_tz,
    align_timestamps_to_tz,
    detect_resolution_and_range,
    snap_billing_period,
    to_pandas_freq,
)
from .tariff_version import TariffVersion


class Tariff(BaseModel):
    versions: list[TariffVersion]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Tariff":
        """Load a tariff definition from YAML."""
        with Path(path).open(encoding="utf-8") as file:
            raw_data = yaml.safe_load(file)

        tariff = cls.model_validate({"versions": raw_data})
        return tariff

    def _find_active_versions(
        self,
        start: dt.datetime,
        end: dt.datetime,
        timezone: dt.tzinfo = UTC,
    ) -> list[tuple[TariffVersion, dt.datetime, dt.datetime]]:
        """Return each segment that overlaps ``[start, end)`` together with the effective sub-range."""

        def norm(v: TariffVersion) -> dt.datetime:
            return align_datetime_to_tz(v.start, timezone)

        start_index = max(0, bisect.bisect_right(self.versions, start, key=norm) - 1)
        end_index = bisect.bisect_right(self.versions, end, key=norm)
        segments = self.versions[start_index:end_index]
        if not segments:
            return []

        starts = [max(norm(segment), start) for segment in segments]
        ends = [norm(segment) for segment in segments[1:]] + [end]
        return list(zip(segments, starts, ends, strict=True))

    def get_energy_cost(
        self,
        start: dt.datetime,
        end: dt.datetime,
        resolution: Resolution = dt.timedelta(minutes=15),
        meter_type: MeterType = MeterType.SINGLE_RATE,
        direction: PowerDirection = PowerDirection.CONSUMPTION,
        timezone: dt.tzinfo = UTC,
    ) -> pd.DataFrame | None:
        """Get energy cost rates in €/MWh. Returns None if no active versions have formulas."""
        start = align_datetime_to_tz(start, timezone)
        end = align_datetime_to_tz(end, timezone)
        return self._collect_version_frames(
            self._find_active_versions(start, end, timezone),
            lambda version, seg_start, seg_end: version.get_energy_cost(
                seg_start, seg_end, resolution, meter_type, direction, timezone
            ),
        )

    def apply_capacity_cost(
        self,
        data: pd.DataFrame,
        start: dt.datetime | None = None,
        end: dt.datetime | None = None,
        timezone: dt.tzinfo = UTC,
        unit: Literal["MW", "MWh"] = "MWh",
    ) -> pd.DataFrame | None:
        """Apply capacity cost formulas across all active versions.  Returns None when unavailable."""
        detected_start, detected_end, _ = detect_resolution_and_range(data)
        start = align_datetime_to_tz(start if start is not None else detected_start, timezone)
        end = align_datetime_to_tz(end if end is not None else detected_end, timezone)
        return self._collect_version_frames(
            self._find_active_versions(start, end, timezone),
            lambda version, seg_start, seg_end: version.apply_capacity_cost(data, timezone, unit=unit),
        )

    def apply_energy_cost(
        self,
        data: pd.DataFrame,
        meter_type: MeterType = MeterType.SINGLE_RATE,
        direction: PowerDirection = PowerDirection.CONSUMPTION,
        start: dt.datetime | None = None,
        end: dt.datetime | None = None,
        timezone: dt.tzinfo = UTC,
    ) -> pd.DataFrame | None:
        """Apply energy cost formulas to quantity data across all active versions."""
        detected_start, detected_end, _ = detect_resolution_and_range(data)
        start = align_datetime_to_tz(start if start is not None else detected_start, timezone)
        end = align_datetime_to_tz(end if end is not None else detected_end, timezone)
        return self._collect_version_frames(
            self._find_active_versions(start, end, timezone),
            lambda version, seg_start, seg_end: version.apply_energy_cost(
                data,
                meter_type,
                direction,
                timezone,
            ),
        )

    @staticmethod
    def _collect_version_frames(
        segments: list[tuple[TariffVersion, dt.datetime, dt.datetime]],
        get_frame: "Callable[[TariffVersion, dt.datetime, dt.datetime], pd.DataFrame | None]",
    ) -> pd.DataFrame | None:
        frames = [
            df[(df["timestamp"] >= seg_start) & (df["timestamp"] < seg_end)]
            for version, seg_start, seg_end in segments
            if (df := get_frame(version, seg_start, seg_end)) is not None and not df.empty
        ]
        if not frames:
            return None
        return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    def get_periodic_cost(self, start: dt.datetime, end: dt.datetime, timezone: dt.tzinfo = UTC) -> dict[str, float]:
        start = align_datetime_to_tz(start, timezone)
        end = align_datetime_to_tz(end, timezone)
        totals: dict[str, float] = {}
        for version, seg_start, seg_end in self._find_active_versions(start, end, timezone):
            for name, cost in version.get_periodic_cost(seg_start, seg_end, timezone).items():
                totals[name] = totals.get(name, 0.0) + cost
        return totals

    def apply(
        self,
        meters: list[Meter],
        start: dt.datetime | None = None,
        end: dt.datetime | None = None,
        resolution: Resolution | None = None,
        timezone: dt.tzinfo = UTC,
    ) -> pd.DataFrame | None:
        if resolution is None:
            resolution = isodate.Duration(months=1)

        # used for start and end if not given, but also capacity cost, which don't differentiate by metertype
        combined_consumption = align_timestamps_to_tz(
            as_single_meter(meters, PowerDirection.CONSUMPTION).data, timezone
        )

        # Align start/end to the specified timezone
        if start is not None:
            start = align_datetime_to_tz(start, timezone)
        if end is not None:
            end = align_datetime_to_tz(end, timezone)

        billing_start: dt.datetime = start if start is not None else combined_consumption["timestamp"].min()
        if end is not None:
            billing_end: dt.datetime = end
        else:
            data_resolution = detect_resolution_and_range(combined_consumption)[2]
            billing_end = combined_consumption["timestamp"].max() + data_resolution
        output_freq = to_pandas_freq(resolution)

        billing_start, billing_end = snap_billing_period(billing_start, billing_end, output_freq)

        frames: list[pd.DataFrame] = []
        for meter in meters:
            aligned_data = align_timestamps_to_tz(meter.data, timezone)
            frame = self._apply_direction_costs(
                aligned_data, billing_start, billing_end, meter.type, meter.direction, output_freq, timezone
            )
            if frame is not None:
                frames.append(frame)

        for optional_frame in [
            self._apply_capacity_costs(combined_consumption, billing_start, billing_end, output_freq, timezone),
            self._apply_fixed_costs(billing_start, billing_end, output_freq, timezone),
        ]:
            if optional_frame is not None:
                frames.append(optional_frame)

        if not frames:
            return None

        result = pd.concat(frames, axis=1)
        total_cols = [c for c in result.columns if c[-1] == "total"]
        result[(CostGroup.TOTAL, MeterType.ALL, "total")] = result[total_cols].sum(axis=1)
        return result.reset_index()

    def _apply_direction_costs(
        self,
        data: pd.DataFrame,
        billing_start: dt.datetime,
        billing_end: dt.datetime,
        meter_type: MeterType,
        direction: PowerDirection,
        output_freq: str,
        timezone: dt.tzinfo = UTC,
    ) -> pd.DataFrame | None:
        sliced = data[(data["timestamp"] >= billing_start) & (data["timestamp"] < billing_end)].copy()
        costs = self.apply_energy_cost(sliced, meter_type, direction, billing_start, billing_end, timezone)
        if costs is None:
            return None
        agg = costs.set_index("timestamp").resample(output_freq).sum()
        cost_cols = [c for c in agg.columns if c != "total"]
        if cost_cols:
            agg["total"] = agg[cost_cols].sum(axis=1)
        cost_group = CostGroup.CONSUMPTION if direction == PowerDirection.CONSUMPTION else CostGroup.INJECTION
        agg.columns = pd.MultiIndex.from_tuples([(cost_group, meter_type, c) for c in agg.columns])
        return agg

    def _apply_capacity_costs(
        self,
        consumption: pd.DataFrame,
        billing_start: dt.datetime,
        billing_end: dt.datetime,
        output_freq: str,
        timezone: dt.tzinfo = UTC,
    ) -> pd.DataFrame | None:
        capacity_df = self.apply_capacity_cost(consumption, billing_start, billing_end, timezone)
        if capacity_df is None:
            return None
        filtered = capacity_df[(capacity_df["timestamp"] >= billing_start) & (capacity_df["timestamp"] < billing_end)]
        if filtered.empty:
            return None
        agg = filtered.set_index("timestamp").resample(output_freq).sum().rename(columns={"value": "total"})
        agg.columns = pd.MultiIndex.from_tuples([(CostGroup.CAPACITY, MeterType.ALL, c) for c in agg.columns])
        return agg

    def _apply_fixed_costs(
        self,
        billing_start: dt.datetime,
        billing_end: dt.datetime,
        output_freq: str,
        timezone: dt.tzinfo = UTC,
    ) -> pd.DataFrame | None:
        period_starts = pd.date_range(billing_start, billing_end, freq=output_freq, inclusive="left")
        period_ends_ts = list(period_starts[1:]) + [pd.Timestamp(billing_end)]
        rows: list[dict] = []
        names: set[str] = set()
        for ps, pe in zip(period_starts, period_ends_ts, strict=True):
            costs = self.get_periodic_cost(ps.to_pydatetime(), pe.to_pydatetime(), timezone)
            names.update(costs.keys())
            rows.append({"timestamp": ps, **costs})
        if not names:
            return None
        df = pd.DataFrame(rows).set_index("timestamp").fillna(0.0)
        df["total"] = df.sum(axis=1)
        df.columns = pd.MultiIndex.from_tuples([(CostGroup.FIXED, MeterType.ALL, c) for c in df.columns])
        return df
