import datetime as dt

import pandas as pd
import pytest

from energy_cost.index import DataFrameIndex, LoadProfileIndex


def test_load_profile_index_raises_on_mismatched_source_resolutions() -> None:
    load_profile = DataFrameIndex(
        pd.DataFrame(columns=["timestamp", "value"]),
        resolution=dt.timedelta(minutes=15),
    )
    data_profile = DataFrameIndex(
        pd.DataFrame(columns=["timestamp", "value"]),
        resolution=dt.timedelta(minutes=30),
    )

    with pytest.raises(ValueError, match="Load profile and data profile must have the same resolution"):
        LoadProfileIndex(
            load_profile=load_profile,
            data_profile=data_profile,
            resolution=dt.timedelta(hours=1),
        )


def test_load_profile_index_returns_weighted_average_for_target_period() -> None:
    source_resolution = dt.timedelta(minutes=15)
    start = dt.datetime(2026, 1, 1, 0, 0, tzinfo=dt.UTC)
    end = dt.datetime(2026, 1, 1, 0, 30, tzinfo=dt.UTC)

    timestamps = pd.date_range(start=start, periods=2, freq="15min")
    load_profile = DataFrameIndex(
        pd.DataFrame({"timestamp": timestamps, "value": [1.0, 3.0]}),
        resolution=source_resolution,
    )
    data_profile = DataFrameIndex(
        pd.DataFrame({"timestamp": timestamps, "value": [10.0, 20.0]}),
        resolution=source_resolution,
    )

    index = LoadProfileIndex(
        load_profile=load_profile,
        data_profile=data_profile,
        resolution=dt.timedelta(minutes=30),
    )

    result = index._get_values(start, end, dt.UTC)

    # Weighted average = (1*10 + 3*20) / (1 + 3) = 17.5
    assert len(result) == 1
    assert result["timestamp"].iloc[0] == pd.Timestamp(start)
    assert float(result["value"].iloc[0]) == pytest.approx(17.5)
