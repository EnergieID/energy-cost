import datetime as dt

from isodate import Duration

from energy_cost.index import CachedIndex, EntsoeDayAheadIndex, LoadProfileIndex

from ..synergrid_load_profile_index import SynergridLoadProfileIndex


class BelpexRLP0N(LoadProfileIndex):
    def __init__(self, entsoe_api_key: str) -> None:
        load_profile_index = CachedIndex(
            SynergridLoadProfileIndex(profile="RLP0N", resolution=dt.timedelta(minutes=15))
        )
        data_profile_index = CachedIndex(EntsoeDayAheadIndex("BE", entsoe_api_key, resolution=dt.timedelta(minutes=15)))
        super().__init__(load_profile_index, data_profile_index, resolution=Duration(months=1))


class BelpexSPP(LoadProfileIndex):
    def __init__(self, entsoe_api_key: str) -> None:
        load_profile_index = CachedIndex(SynergridLoadProfileIndex(profile="SPP", resolution=dt.timedelta(minutes=15)))
        data_profile_index = CachedIndex(EntsoeDayAheadIndex("BE", entsoe_api_key, resolution=dt.timedelta(minutes=15)))
        super().__init__(load_profile_index, data_profile_index, resolution=Duration(months=1))
