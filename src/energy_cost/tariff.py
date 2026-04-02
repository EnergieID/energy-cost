import bisect
import datetime as dt
from collections.abc import Callable
from pathlib import Path

import isodate
import pandas as pd
import yaml
from pydantic import BaseModel

from .resolution import Resolution, detect_resolution_and_range, to_pandas_freq
from .tariff_version import MeterType, PowerDirection, TariffVersion


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

    def get_energy_cost(
        self,
        start: dt.datetime,
        end: dt.datetime,
        resolution: Resolution = dt.timedelta(minutes=15),
        meter_type: MeterType = MeterType.SINGLE_RATE,
        direction: PowerDirection = PowerDirection.CONSUMPTION,
    ) -> pd.DataFrame | None:
        """Get energy cost rates in €/MWh. Returns None if no active versions have formulas."""
        return self._collect_version_frames(
            self._find_active_versions(start, end),
            lambda version, seg_start, seg_end: version.get_energy_cost(
                seg_start, seg_end, resolution, meter_type, direction
            ),
        )

    def apply_capacity_cost(self, data: pd.DataFrame) -> pd.DataFrame | None:
        """Apply capacity cost formulas across all active versions.  Returns None when unavailable."""
        start, end, _ = detect_resolution_and_range(data)
        return self._collect_version_frames(
            self._find_active_versions(start, end),
            lambda version, seg_start, seg_end: version.apply_capacity_cost(data),
        )

    def apply_energy_cost(
        self,
        data: pd.DataFrame,
        meter_type: MeterType = MeterType.SINGLE_RATE,
        direction: PowerDirection = PowerDirection.CONSUMPTION,
    ) -> pd.DataFrame | None:
        """Apply energy cost formulas to quantity data across all active versions."""
        start, end, _ = detect_resolution_and_range(data)
        return self._collect_version_frames(
            self._find_active_versions(start, end),
            lambda version, seg_start, seg_end: version.apply_energy_cost(
                data[(data["timestamp"] >= seg_start) & (data["timestamp"] < seg_end)],
                meter_type,
                direction,
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

    def get_periodic_cost(self, start: dt.datetime, end: dt.datetime) -> dict[str, float]:
        totals: dict[str, float] = {}
        for version, seg_start, seg_end in self._find_active_versions(start, end):
            for name, cost in version.get_periodic_cost(seg_start, seg_end).items():
                totals[name] = totals.get(name, 0.0) + cost
        return totals

    def apply(
        self,
        consumption: pd.DataFrame,
        injection: pd.DataFrame | None = None,
        start: dt.datetime | None = None,
        end: dt.datetime | None = None,
        resolution: Resolution | None = None,
        meter_type: MeterType = MeterType.SINGLE_RATE,
    ) -> pd.DataFrame | None:
        """Apply the tariff to consumption (and optionally injection) data.

        Parameters
        ----------
        consumption:
            DataFrame with ``timestamp`` and ``value`` columns (quantity per interval, e.g. MWh).
            May extend beyond ``start``/``end`` to provide capacity-cost history (e.g. 12 months
            of prior data for the Flemish peak-demand calculation).
        injection:
            Optional DataFrame with the same ``timestamp``/``value`` schema for injection quantities.
        start:
            Start of the billing period (inclusive).  Defaults to the earliest timestamp in
            ``consumption``.
        end:
            End of the billing period (exclusive).  Defaults to one data-resolution step after the
            last timestamp in ``consumption``.
        resolution:
            Output resolution; costs are summed into buckets of this width.  Defaults to P1M
            (calendar-monthly).
        meter_type:
            Meter type used when looking up consumption/injection formulas.
        """
        if resolution is None:
            resolution = isodate.Duration(months=1)
        billing_start: dt.datetime = start if start is not None else consumption["timestamp"].min()
        if end is not None:
            billing_end: dt.datetime = end
        else:
            data_resolution = detect_resolution_and_range(consumption)[2]
            billing_end = consumption["timestamp"].max() + data_resolution
        output_freq = to_pandas_freq(resolution)

        frames: list[pd.DataFrame] = []
        for direction, direction_data in [
            (PowerDirection.CONSUMPTION, consumption),
            (PowerDirection.INJECTION, injection),
        ]:
            if direction_data is None:
                continue
            frame = self._apply_direction_costs(
                direction_data, billing_start, billing_end, meter_type, direction, output_freq
            )
            if frame is not None:
                frames.append(frame)

        for optional_frame in [
            self._apply_capacity_costs(consumption, billing_start, billing_end, output_freq),
            self._apply_fixed_costs(billing_start, billing_end, output_freq),
        ]:
            if optional_frame is not None:
                frames.append(optional_frame)

        if not frames:
            return None

        result = pd.concat(frames, axis=1)
        total_cols = [c for c in result.columns if c[-1] == "total"]
        result[("total", "total")] = result[total_cols].sum(axis=1)
        return result.reset_index()

    def _apply_direction_costs(
        self,
        data: pd.DataFrame,
        billing_start: dt.datetime,
        billing_end: dt.datetime,
        meter_type: MeterType,
        direction: PowerDirection,
        output_freq: str,
    ) -> pd.DataFrame | None:
        sliced = data[(data["timestamp"] >= billing_start) & (data["timestamp"] < billing_end)].copy()
        costs = self.apply_energy_cost(sliced, meter_type, direction)
        if costs is None:
            return None
        agg = costs.set_index("timestamp").resample(output_freq).sum()
        cost_cols = [c for c in agg.columns if c != "total"]
        agg[("total")] = agg[cost_cols].sum(axis=1)
        agg.columns = pd.MultiIndex.from_tuples([(direction.value, c) for c in agg.columns])
        return agg

    def _apply_capacity_costs(
        self,
        consumption: pd.DataFrame,
        billing_start: dt.datetime,
        billing_end: dt.datetime,
        output_freq: str,
    ) -> pd.DataFrame | None:
        capacity_df = self.apply_capacity_cost(consumption)
        if capacity_df is None:
            return None
        filtered = capacity_df[(capacity_df["timestamp"] >= billing_start) & (capacity_df["timestamp"] < billing_end)]
        if filtered.empty:
            return None
        agg = filtered.set_index("timestamp").resample(output_freq).sum().rename(columns={"value": "total"})
        agg.columns = pd.MultiIndex.from_tuples([("capacity", c) for c in agg.columns])
        return agg

    def _apply_fixed_costs(
        self,
        billing_start: dt.datetime,
        billing_end: dt.datetime,
        output_freq: str,
    ) -> pd.DataFrame | None:
        period_starts = pd.date_range(billing_start, billing_end, freq=output_freq, inclusive="left")
        period_ends_ts = list(period_starts[1:]) + [pd.Timestamp(billing_end)]
        rows: list[dict] = []
        names: set[str] = set()
        for ps, pe in zip(period_starts, period_ends_ts, strict=True):
            costs = self.get_periodic_cost(ps.to_pydatetime(), pe.to_pydatetime())
            names.update(costs.keys())
            rows.append({"timestamp": ps, **costs})
        if not names:
            return None
        df = pd.DataFrame(rows).set_index("timestamp").fillna(0.0)
        df["total"] = df.sum(axis=1)
        df.columns = pd.MultiIndex.from_tuples([("fixed", c) for c in df.columns])
        return df
