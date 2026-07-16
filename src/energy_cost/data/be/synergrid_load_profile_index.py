import datetime as dt
from importlib import resources
from pathlib import Path

import pandas as pd

from energy_cost.index import DataFrameIndex

DEFAULT_RESOLUTION = dt.timedelta(minutes=15)
VALID_PROFILES = {"RLP0N", "SPP"}
VALID_REGIONS = {"belgium", "flanders", "wallonia", "brussels"}


class SynergridLoadProfileIndex(DataFrameIndex):
    """
    Index for Synergrid load profiles.

    This index fetches load profile data from the Synergrid Excel files and provides it in a standardized format.
    """

    def __init__(
        self,
        profile: str,
        region: str = "belgium",
        csv_path: str | Path | None = None,
    ) -> None:
        profile = profile.upper()
        region = region.lower()
        if profile not in VALID_PROFILES:
            raise ValueError(f"Unsupported profile '{profile}'. Expected one of {sorted(VALID_PROFILES)}.")
        if region not in VALID_REGIONS:
            raise ValueError(f"Unsupported region '{region}'. Expected one of {sorted(VALID_REGIONS)}.")

        resolved_csv_path: str | Path
        if csv_path is None:
            resolved_csv_path = str(resources.files("energy_cost.data.be").joinpath(f"synergrid_{profile.lower()}.csv"))
        else:
            resolved_csv_path = csv_path

        raw = pd.read_csv(resolved_csv_path, parse_dates=["timestamp"])
        if region not in raw.columns:
            raise ValueError(f"CSV file {resolved_csv_path} has no '{region}' column.")

        df = raw[["timestamp", region]].rename(columns={region: "value"})

        super().__init__(df, resolution=DEFAULT_RESOLUTION)
