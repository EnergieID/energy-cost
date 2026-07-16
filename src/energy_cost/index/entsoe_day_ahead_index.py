import datetime as dt
from importlib import resources

import pandas as pd
from entsoe.entsoe import EntsoePandasClient

from energy_cost.resolution import align_timestamps_to_tz

from .index import Index

OVERWRITES: dict[str, list[pd.DataFrame]] = {
    "BE": [
        pd.read_csv(
            str(resources.files("energy_cost.data.be").joinpath("EPEX_DA_20240626.csv")), parse_dates=["timestamp"]
        )
    ],
}

# Precompute bounds for each overwrite so the hot path is a cheap two-timestamp comparison.
_OVERWRITE_BOUNDS: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, pd.DataFrame]]] = {
    country: [(df["timestamp"].min(), df["timestamp"].max(), df) for df in dfs] for country, dfs in OVERWRITES.items()
}


class EntsoeDayAheadIndex(Index):
    """An ENTSO-E day-ahead index for a given country."""

    def __init__(
        self,
        country_code: str,
        api_key: str,
        resolution: dt.timedelta = dt.timedelta(minutes=15),
    ) -> None:
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
        df = align_timestamps_to_tz(df, timezone)

        # Apply country-specific overwrites when the requested range overlaps the overwrite period.
        if self.country_code in _OVERWRITE_BOUNDS:
            for ow_min, ow_max, overwrite_df in _OVERWRITE_BOUNDS[self.country_code]:
                if ow_max >= start and ow_min <= end:
                    overwrite = align_timestamps_to_tz(overwrite_df.copy(), timezone)
                    df = df.set_index("timestamp")
                    df.update(overwrite.set_index("timestamp"))
                    df = df.reset_index()

        return df
