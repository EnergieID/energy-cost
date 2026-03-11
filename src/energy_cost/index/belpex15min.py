import datetime as dt

import narwhals as nw
import pandas as pd
from entsoe.entsoe import EntsoePandasClient

from .index import Index


class Belpex15min(Index):
    """The Belpex 15-minute index."""

    name = "Belpex15min"

    def get_values(self, start: dt.datetime, end: dt.datetime, resolution: dt.timedelta) -> nw.DataFrame:
        """Get the index values for the given time range and resolution."""
        if resolution != dt.timedelta(minutes=15):
            raise ValueError("Belpex15min index only supports 15 minute resolution.")

        client = EntsoePandasClient()
        df = (
            client.query_day_ahead_prices(
                country_code="BE",
                start=pd.Timestamp(start),
                end=pd.Timestamp(end),
            )
            .resample("15T")
            .ffill()
            .to_frame()
            .reset_index()
            .rename(columns={"index": "timestamp", 0: "value"})
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        return nw.from_native(df)
