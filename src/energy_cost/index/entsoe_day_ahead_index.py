import datetime as dt

import pandas as pd
from entsoe.entsoe import EntsoePandasClient

from energy_cost.resolution import align_timestamps_to_tz

from .index import Index


class EntsoeDayAheadIndex(Index):
    """An ENTSO-E day-ahead index for a given country."""

    def __init__(self, country_code: str, api_key: str, resolution: dt.timedelta = dt.timedelta(minutes=15)) -> None:
        self.client = EntsoePandasClient(api_key=api_key)
        self.country_code = country_code
        super().__init__(resolution=resolution)

    def _get_values(self, start: pd.Timestamp, end: pd.Timestamp, timezone: dt.tzinfo) -> pd.DataFrame:
        """Get the index values for the given time range in €/MWh."""

        df = (
            self.client.query_day_ahead_prices(
                country_code=self.country_code,
                start=start,
                end=end,
            )
            .to_frame()
            .reset_index()
            .rename(columns={"index": "timestamp", 0: "value"})
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return align_timestamps_to_tz(df, timezone)
