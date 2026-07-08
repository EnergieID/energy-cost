import datetime as dt

from isodate import Duration

from energy_cost.index import CachedIndex, EntsoeDayAheadIndex, LoadProfileIndex

from ..synergrid_load_profile_index import SynergridLoadProfileIndex


class BelpexRLP0N(LoadProfileIndex):
    def __init__(self, entsoe_api_key: str, region: str) -> None:
        load_profile_index = SynergridLoadProfileIndex(profile="RLP0N", region=region)
        data_profile_index = CachedIndex(EntsoeDayAheadIndex("BE", entsoe_api_key, resolution=dt.timedelta(minutes=15)))
        super().__init__(load_profile_index, data_profile_index, resolution=Duration(months=1))


class BelpexSPP(LoadProfileIndex):
    def __init__(self, entsoe_api_key: str, region: str) -> None:
        load_profile_index = SynergridLoadProfileIndex(profile="SPP", region=region)
        data_profile_index = CachedIndex(EntsoeDayAheadIndex("BE", entsoe_api_key, resolution=dt.timedelta(minutes=15)))
        super().__init__(load_profile_index, data_profile_index, resolution=Duration(months=1))


class CachedBelpexRLP0N(CachedIndex):
    def __init__(self, entsoe_api_key: str, region: str) -> None:
        index = BelpexRLP0N(entsoe_api_key, region)
        super().__init__(index, f"belpex_rlp0n_{region}", old_threshold=dt.timedelta(days=32))


class CachedBelpexSPP(CachedIndex):
    def __init__(self, entsoe_api_key: str, region: str) -> None:
        index = BelpexSPP(entsoe_api_key, region)
        super().__init__(index, f"belpex_spp_{region}", old_threshold=dt.timedelta(days=32))
