import datetime as dt
from datetime import UTC

import isodate
import pandas as pd

from energy_cost.versioning import VersionedCollection

from .meter import CostGroup, Meter
from .resolution import (
    Resolution,
    align_datetime_to_tz,
    snap_billing_period,
    to_pandas_freq,
)
from .tariff_version import TariffVersion


class Tariff(VersionedCollection[TariffVersion]):
    def get_values(
        self,
        start: dt.datetime,
        end: dt.datetime,
        output_resolution: Resolution = dt.timedelta(minutes=15),
        cost_group: CostGroup = CostGroup.CONSUMPTION,
        timezone: dt.tzinfo = UTC,
    ) -> pd.DataFrame | None:
        start = align_datetime_to_tz(start, timezone)
        end = align_datetime_to_tz(end, timezone)
        return self.collect_version_frames(
            lambda version, seg_start, seg_end: version.get_values(
                seg_start, seg_end, output_resolution, cost_group, timezone
            ),
            start,
            end,
            timezone,
        )

    def apply(
        self,
        consumption: Meter,
        injection: Meter | None = None,
        start: dt.datetime | None = None,
        end: dt.datetime | None = None,
        output_resolution: Resolution | None = None,
        timezone: dt.tzinfo = UTC,
        binning_anchor: dt.datetime | None = None,
    ) -> pd.DataFrame | None:
        if output_resolution is None:
            output_resolution = isodate.Duration(months=1)
        if start is None:
            start = consumption.measurements.start
        if end is None:
            end = consumption.measurements.end

        consumption = consumption.align_to_timezone(timezone)
        if injection is not None:
            injection = injection.align_to_timezone(timezone)

        start = align_datetime_to_tz(start, timezone)
        end = align_datetime_to_tz(end, timezone)
        output_freq = to_pandas_freq(output_resolution)

        start, end = snap_billing_period(start, end, output_freq, anchor=binning_anchor)

        binning_anchor = start if binning_anchor is None else align_datetime_to_tz(binning_anchor, timezone)

        result = self.collect_version_frames(
            lambda version, seg_start, seg_end: version.apply(
                consumption,
                injection,
                seg_start,
                seg_end,
                output_resolution,
                timezone,
                binning_anchor=binning_anchor,
            ),
            start,
            end,
            timezone,
        )
        return result
