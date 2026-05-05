import datetime as dt
from datetime import UTC
from typing import Literal

import isodate
import pandas as pd

from energy_cost.versioning import VersionedCollection

from .meter import CostGroup, Meter, MeterType, PowerDirection, TariffCategory, as_single_meter
from .resolution import (
    Resolution,
    align_datetime_to_tz,
    align_timestamps_to_tz,
    detect_resolution_and_range,
    snap_billing_period,
    to_pandas_freq,
)
from .tariff_version import TariffVersion


class Tariff(VersionedCollection[TariffVersion]):
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
        return self.collect_version_frames(
            lambda version, seg_start, seg_end: version.get_energy_cost(
                seg_start, seg_end, resolution, meter_type, direction, timezone
            ),
            start,
            end,
            timezone,
        )

    def apply_capacity_cost(
        self,
        data: pd.DataFrame,
        start: dt.datetime | None = None,
        end: dt.datetime | None = None,
        timezone: dt.tzinfo = UTC,
        unit: Literal["MW", "MWh"] = "MWh",
        output_resolution: Resolution | None = None,
    ) -> pd.DataFrame | None:
        """Apply capacity cost formulas across all active versions.  Returns None when unavailable."""
        if start is None or end is None:
            detected_start, detected_end, _ = detect_resolution_and_range(data)
            if start is None:
                start = detected_start
            if end is None:
                end = detected_end
        start = align_datetime_to_tz(start, timezone)
        end = align_datetime_to_tz(end, timezone)
        return self.collect_version_frames(
            lambda version, seg_start, seg_end: version.apply_capacity_cost(
                data,
                start=seg_start,
                end=seg_end,
                timezone=timezone,
                unit=unit,
                output_resolution=output_resolution,
            ),
            start,
            end,
            timezone,
        )

    def apply_energy_cost(
        self,
        data: pd.DataFrame,
        meter_type: MeterType = MeterType.SINGLE_RATE,
        direction: PowerDirection = PowerDirection.CONSUMPTION,
        start: dt.datetime | None = None,
        end: dt.datetime | None = None,
        timezone: dt.tzinfo = UTC,
        output_resolution: Resolution | None = None,
        input_resolution: Resolution | None = None,
    ) -> pd.DataFrame | None:
        """Apply energy cost formulas to quantity data across all active versions."""
        if start is None or end is None or input_resolution is None:
            detected_start, detected_end, detected_resolution = detect_resolution_and_range(data)
            if start is None:
                start = detected_start
            if end is None:
                end = detected_end
            if input_resolution is None:
                input_resolution = detected_resolution
        start = align_datetime_to_tz(start, timezone)
        end = align_datetime_to_tz(end, timezone)
        return self.collect_version_frames(
            lambda version, seg_start, seg_end: version.apply_energy_cost(
                data,
                meter_type,
                direction,
                timezone=timezone,
                start=seg_start,
                end=seg_end,
                input_resolution=input_resolution,
                output_resolution=output_resolution,
            ),
            start,
            end,
            timezone,
        )

    def apply_periodic_costs(
        self,
        start: dt.datetime,
        end: dt.datetime,
        output_resolution: Resolution,
        timezone: dt.tzinfo = UTC,
    ) -> pd.DataFrame | None:
        """Apply periodic cost formulas across all active versions, returning a DataFrame with a column per named cost."""
        start = align_datetime_to_tz(start, timezone)
        end = align_datetime_to_tz(end, timezone)
        return self.collect_version_frames(
            lambda version, seg_start, seg_end: version.apply_periodic_costs(
                seg_start, seg_end, output_resolution, timezone
            ),
            start,
            end,
            timezone,
        )

    def apply(
        self,
        meters: list[Meter],
        start: dt.datetime | None = None,
        end: dt.datetime | None = None,
        resolution: Resolution | None = None,
        timezone: dt.tzinfo = UTC,
        include_meter_type: bool = False,
        tariff_category: TariffCategory | None = None,
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

        detected_start, detected_end, data_resolution = detect_resolution_and_range(combined_consumption)
        billing_start: dt.datetime = start if start is not None else detected_start
        billing_end: dt.datetime = end if end is not None else detected_end
        output_freq = to_pandas_freq(resolution)

        billing_start, billing_end = snap_billing_period(billing_start, billing_end, output_freq)

        frames: list[pd.DataFrame] = []
        for meter in meters:
            aligned_data = align_timestamps_to_tz(meter.data, timezone)
            frame = self._apply_direction_costs(
                aligned_data,
                billing_start,
                billing_end,
                meter.type,
                meter.direction,
                resolution,
                timezone,
                input_resolution=data_resolution,
            )
            if frame is not None:
                frames.append(frame)

        for optional_frame in [
            self._apply_capacity_costs(combined_consumption, billing_start, billing_end, resolution, timezone),
            self._apply_fixed_costs(billing_start, billing_end, resolution, timezone),
        ]:
            if optional_frame is not None:
                frames.append(optional_frame)

        if not frames:
            return None

        result = pd.concat(frames, axis=1, sort=False)
        total_cols = [c for c in result.columns if c[-1] == "total"]
        result[(CostGroup.TOTAL, MeterType.ALL, "total")] = result[total_cols].sum(axis=1)
        if not include_meter_type:
            result = _collapse_meter_type(result)
        if tariff_category is not None:
            result.columns = pd.MultiIndex.from_tuples([(tariff_category,) + col for col in result.columns])
            if include_meter_type:
                result[(TariffCategory.TOTAL, CostGroup.TOTAL, MeterType.ALL, "total")] = result[
                    (tariff_category, CostGroup.TOTAL, MeterType.ALL, "total")
                ]
            else:
                result[(TariffCategory.TOTAL, CostGroup.TOTAL, "total")] = result[
                    (tariff_category, CostGroup.TOTAL, "total")
                ]
        return result.reset_index()

    def _apply_direction_costs(
        self,
        data: pd.DataFrame,
        billing_start: dt.datetime,
        billing_end: dt.datetime,
        meter_type: MeterType,
        direction: PowerDirection,
        resolution: Resolution,
        timezone: dt.tzinfo = UTC,
        input_resolution: Resolution | None = None,
    ) -> pd.DataFrame | None:
        costs = self.apply_energy_cost(
            data,
            meter_type,
            direction,
            billing_start,
            billing_end,
            timezone,
            output_resolution=resolution,
            input_resolution=input_resolution,
        )
        if costs is None:
            return None
        agg = costs.set_index("timestamp")
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
        resolution: Resolution,
        timezone: dt.tzinfo = UTC,
    ) -> pd.DataFrame | None:
        capacity_df = self.apply_capacity_cost(
            consumption,
            billing_start,
            billing_end,
            timezone,
            output_resolution=resolution,
        )
        if capacity_df is None or capacity_df.empty:
            return None
        agg = capacity_df.set_index("timestamp").rename(columns={"value": "total"}).fillna(0.0)
        agg.columns = pd.MultiIndex.from_tuples([(CostGroup.CAPACITY, MeterType.ALL, c) for c in agg.columns])
        return agg

    def _apply_fixed_costs(
        self,
        billing_start: dt.datetime,
        billing_end: dt.datetime,
        resolution: Resolution,
        timezone: dt.tzinfo = UTC,
    ) -> pd.DataFrame | None:
        fixed_df = self.apply_periodic_costs(
            billing_start, billing_end, output_resolution=resolution, timezone=timezone
        )
        if fixed_df is None or fixed_df.empty:
            return None
        agg = fixed_df.set_index("timestamp").fillna(0.0)
        cost_cols = [c for c in agg.columns if c != "total"]
        if cost_cols:
            agg["total"] = agg[cost_cols].sum(axis=1)
        agg.columns = pd.MultiIndex.from_tuples([(CostGroup.FIXED, MeterType.ALL, c) for c in agg.columns])
        return agg


def _collapse_meter_type(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse (CostGroup, MeterType, cost_type) → (CostGroup, cost_type) by summing across MeterType."""
    collapsed = df.T.groupby(level=[0, 2]).sum().T
    collapsed.columns = pd.MultiIndex.from_tuples(collapsed.columns)
    return collapsed
