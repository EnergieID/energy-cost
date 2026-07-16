import datetime as dt
import zoneinfo

import pandas as pd
from isodate import Duration

from energy_cost.index import CachedIndex, EntsoeDayAheadIndex, Index, LoadProfileIndex

from ..synergrid_load_profile_index import SynergridLoadProfileIndex

OVERWRITES = [
    ("RLP0N", "flanders", 2025, 3, 1, 94.61),
]


class BelpexLoadProfile(LoadProfileIndex):
    def __init__(
        self, profile: str, region: str, entsoe_api_key: str | None = None, belpex15min: Index | None = None
    ) -> None:
        self.region = region.upper()
        self.profile = profile.lower()
        load_profile_index = SynergridLoadProfileIndex(profile=self.profile, region=self.region)
        if belpex15min is not None:
            data_profile_index = belpex15min
        elif entsoe_api_key is not None:
            data_profile_index = CachedIndex(
                EntsoeDayAheadIndex("BE", entsoe_api_key, resolution=dt.timedelta(minutes=15)), file_name="belpex15min"
            )
        else:
            raise ValueError("Either entsoe_api_key or belpex15min must be provided.")
        super().__init__(load_profile_index, data_profile_index, resolution=Duration(months=1))

    def _get_values(self, start: dt.datetime, end: dt.datetime, timezone: dt.tzinfo) -> pd.DataFrame:
        result = super()._get_values(start, end, timezone)

        for profile, region, year, month, day, expected in OVERWRITES:
            if profile == self.profile and region == self.region:
                result.loc[
                    (result["timestamp"].dt.year == year)
                    & (result["timestamp"].dt.month == month)
                    & (result["timestamp"].dt.day == day),
                    "value",
                ] = expected

        return result


class BelpexRLP0N(CachedIndex):
    def __init__(
        self, entsoe_api_key: str | None = None, region: str = "belgium", belpex15min: Index | None = None
    ) -> None:
        index = BelpexLoadProfile("RLP0N", region, entsoe_api_key, belpex15min)
        super().__init__(
            index,
            f"belpex_rlp0n_{region.lower()}",
            old_threshold=dt.timedelta(days=32),
            cache_timezone=zoneinfo.ZoneInfo("Europe/Brussels"),
        )


class BelpexSPP(CachedIndex):
    def __init__(
        self, entsoe_api_key: str | None = None, region: str = "belgium", belpex15min: Index | None = None
    ) -> None:
        index = BelpexLoadProfile("SPP", region, entsoe_api_key, belpex15min)
        super().__init__(
            index,
            f"belpex_spp_{region.lower()}",
            old_threshold=dt.timedelta(days=32),
            cache_timezone=zoneinfo.ZoneInfo("Europe/Brussels"),
        )
