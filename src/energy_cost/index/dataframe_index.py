import datetime as dt

import pandas as pd

from .index import Index


class DataFrameIndex(Index):
    def __init__(self, df: pd.DataFrame):
        if "timestamp" not in df.columns or "value" not in df.columns:
            raise ValueError("DataFrame must contain 'timestamp' and 'value' columns.")
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        self.df = df.copy().sort_values("timestamp")

    def get_values(self, start: dt.datetime, end: dt.datetime, resolution: dt.timedelta) -> pd.DataFrame:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        target_index = pd.date_range(start=start_ts, end=end_ts, freq=resolution, inclusive="left")

        return pd.merge_asof(
            left=pd.DataFrame({"timestamp": target_index}),
            right=self.df,
            on="timestamp",
            direction="backward",
        )


class CSVIndex(DataFrameIndex):
    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path, parse_dates=["timestamp"])
        super().__init__(df)


class YAMLIndex(DataFrameIndex):
    def __init__(self, yaml_path: str):
        import yaml

        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        df = pd.DataFrame(data)
        super().__init__(df)
