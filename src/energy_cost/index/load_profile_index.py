import datetime as dt

import pandas as pd

from energy_cost.resolution import Resolution, redistribute_to_resolution

from .index import Index


class LoadProfileIndex(Index):
    """
    This represent an index that is avaraged out allong a load profile.
    Eg. we have a profile of avarage houshold consumption every 15 minutes
    We can use this profile to calculate the monthly enegry cost by multiplying the load profile with the energy price index.
    We then divide the sum of the multiplied values by the sum of the load profile to get the avarage energy cost for that month.
    """

    def __init__(
        self, load_profile: Index, data_profile: Index, resolution: Resolution, forward_fill: bool = False
    ) -> None:
        self.load_profile = load_profile
        self.data_profile = data_profile
        if not load_profile.resolution == data_profile.resolution:
            raise ValueError("Load profile and data profile must have the same resolution.")
        self.source_resolution = load_profile.resolution

        super().__init__(resolution, forward_fill=forward_fill)

    def _get_values(self, start: dt.datetime, end: dt.datetime, timezone: dt.tzinfo) -> pd.DataFrame:
        load_profile_values = self.load_profile.get_values(start, end, self.source_resolution, timezone)
        data_profile_values = self.data_profile.get_values(start, end, self.source_resolution, timezone)

        # Merge the two profiles on the timestamp
        merged = pd.merge(
            load_profile_values, data_profile_values, on="timestamp", how="outer", suffixes=("_load", "_data")
        )
        # apply index for each value in the load profile
        merged["multiplied"] = merged["value_load"] * merged["value_data"]

        # redistribute the values to the target resolution
        redistributed = redistribute_to_resolution(merged, self.source_resolution, self.resolution, start, end)
        # divide the multiplied values by the load profile values to get the avarage index for that period according to the load profile
        redistributed["value"] = redistributed["multiplied"] / redistributed["value_load"]

        return redistributed[["timestamp", "value"]]
