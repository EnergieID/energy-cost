import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from energy_cost.index.dataframe_index import DataFrameIndex
from energy_cost.resolution import Resolution


def display_yaml(path_str: str) -> None:
    yaml_text = Path(path_str).read_text(encoding="utf-8")
    display(Markdown(f"```yaml\n{yaml_text}\n```"))


class MockEntsoeDayAheadIndex(DataFrameIndex):
    """A mock day-ahead index that generates random 15-minute price data.

    Drop-in replacement for EntsoeDayAheadIndex / CachedEntsoeDayAheadIndex
    so notebooks can run without an ENTSO-E API key.
    """

    def __init__(
        self,
        country_code: str = "BE",
        resolution: Resolution = dt.timedelta(minutes=15),
        seed: int = 42,
    ) -> None:
        rng = np.random.default_rng(seed)
        timestamps = pd.date_range(start="2024-01-01", end="2027-01-01", freq="15min", tz="UTC")
        values = rng.uniform(low=20.0, high=150.0, size=len(timestamps))
        df = pd.DataFrame({"timestamp": timestamps, "value": values})
        super().__init__(df, resolution=resolution)
