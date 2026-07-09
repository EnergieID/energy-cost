import datetime as dt
import zoneinfo

import pandas as pd
from isodate import Duration

from energy_cost.index import CachedIndex, EntsoeDayAheadIndex, LoadProfileIndex

from ..synergrid_load_profile_index import SynergridLoadProfileIndex

OVERWRITES = [
    ("RLP0N", "flanders", 2024, 6, 1, 60.27),
    ("RLP0N", "flanders", 2025, 3, 1, 94.61),
    ("SPP", "flanders", 2024, 6, 1, 32.52),
]


class BelpexLoadProfile(LoadProfileIndex):
    def __init__(self, entsoe_api_key: str, profile: str, region: str) -> None:
        self.region = region
        self.profile = profile
        load_profile_index = SynergridLoadProfileIndex(profile=profile, region=region)
        data_profile_index = CachedIndex(
            EntsoeDayAheadIndex("BE", entsoe_api_key, resolution=dt.timedelta(minutes=15)), file_name="belpex15min"
        )
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
    def __init__(self, entsoe_api_key: str, region: str) -> None:
        index = BelpexLoadProfile(entsoe_api_key, "RLP0N", region)
        super().__init__(
            index,
            f"belpex_rlp0n_{region}",
            old_threshold=dt.timedelta(days=32),
            cache_timezone=zoneinfo.ZoneInfo("Europe/Brussels"),
        )


class BelpexSPP(CachedIndex):
    def __init__(self, entsoe_api_key: str, region: str) -> None:
        index = BelpexLoadProfile(entsoe_api_key, "SPP", region)
        super().__init__(
            index,
            f"belpex_spp_{region}",
            old_threshold=dt.timedelta(days=32),
            cache_timezone=zoneinfo.ZoneInfo("Europe/Brussels"),
        )
