import datetime as dt

import pandas as pd
from entsoe.entsoe import EntsoePandasClient

from .index import Index


class EntsoeDayAheadIndex(Index):
    """An ENTSO-E day-ahead index for a given country."""

    def __init__(self, country_code: str, api_key: str):
        self.client = EntsoePandasClient(api_key=api_key)
        self.country_code = country_code

    def get_values(self, start: dt.datetime, end: dt.datetime, resolution: dt.timedelta) -> pd.DataFrame:
        """Get the index values for the given time range and resolution."""
        if resolution != dt.timedelta(minutes=15):
            raise ValueError("EntsoeDayAheadIndex only supports 15 minute resolution.")

        df = (
            self.client.query_day_ahead_prices(
                country_code=self.country_code,
                start=pd.Timestamp(start),
                end=pd.Timestamp(end),
            )
            .resample("15min")
            .ffill()
            .to_frame()
            .reset_index()
            .rename(columns={"index": "timestamp", 0: "value"})
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return df
