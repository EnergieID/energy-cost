from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest
import pytz

from energy_cost.index import DataFrameIndex, Index


class DummyIndex(Index):
    def __init__(self):
        super().__init__(resolution=dt.timedelta(minutes=15))

    def _get_values(self, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "timestamp": pd.date_range(start=start, end=end, freq="15min", inclusive="left"),
                "value": 1.0,
            }
        )


def test_register_and_from_name_returns_same_instance() -> None:
    index = DummyIndex()

    Index.register("dummy", index)

    assert Index.from_name("dummy") is index


def test_from_name_raises_for_unknown_index() -> None:
    with pytest.raises(ValueError, match="Unsupported index: unknown"):
        Index.from_name("unknown")


def test_index_returns_nan_for_out_of_range_values() -> None:
    index = DataFrameIndex(
        pd.DataFrame(
            {"timestamp": pd.date_range("2020-01-01", periods=4, freq="15min"), "value": [1.0, 2.0, 3.0, 4.0]}
        ),
        resolution=dt.timedelta(minutes=15),
    )

    Index.register("dummy", index)

    df = index.get_values(
        start=dt.datetime(2020, 1, 1, 0, 30),
        end=dt.datetime(2020, 1, 1, 7, 0),
        resolution=dt.timedelta(minutes=15),
    )

    assert df["value"].tolist()[:2] == [3.0, 4.0]
    assert all(pd.isna(df["value"].tolist()[2:]))


def test_index_returns_all_nan_if_no_data_in_range() -> None:
    index = DataFrameIndex(
        pd.DataFrame(
            {"timestamp": pd.date_range("2020-01-01", periods=4, freq="15min"), "value": [1.0, 2.0, 3.0, 4.0]}
        ),
        resolution=dt.timedelta(minutes=15),
    )

    Index.register("dummy", index)

    df = index.get_values(
        start=dt.datetime(2021, 1, 1, 1, 0),
        end=dt.datetime(2021, 1, 1, 2, 0),
        resolution=dt.timedelta(minutes=15),
    )

    assert all(pd.isna(df["value"].tolist()))


def test_index_ger_values_return_in_timezone_of_input() -> None:
    index = DataFrameIndex(
        pd.DataFrame(
            {"timestamp": pd.date_range("2020-01-01", periods=4, freq="15min"), "value": [1.0, 2.0, 3.0, 4.0]}
        ),
        resolution=dt.timedelta(minutes=15),
    )

    df = index.get_values(
        start=dt.datetime(2020, 1, 1, 0, 30, tzinfo=dt.UTC),
        end=dt.datetime(2020, 1, 1, 7, 0, tzinfo=dt.UTC),
        resolution=dt.timedelta(minutes=15),
    )

    assert df["timestamp"].dtype == "datetime64[us, UTC]"


def test_index_ger_values_return_in_timezone_of_input_for_dumy_index() -> None:
    class DumIndex(Index):
        def __init__(self):
            super().__init__(resolution=dt.timedelta(minutes=15))

        def _get_values(self, start: dt.datetime, end: dt.datetime) -> pd.DataFrame:
            # explicitly use naive timestamps to test that they are localized to the timezone of the input timestamps
            start = dt.datetime(2020, 1, 1, 0, 30)
            end = dt.datetime(2020, 1, 1, 7, 0)
            return pd.DataFrame(
                {
                    "timestamp": pd.date_range(start=start, end=end, freq="15min", inclusive="left"),
                    "value": 1.0,
                }
            )

    index = DumIndex()
    df = index.get_values(
        start=dt.datetime(2020, 1, 1, 0, 30, tzinfo=pytz.timezone("Europe/Amsterdam")),
        end=dt.datetime(2020, 1, 1, 7, 0, tzinfo=pytz.timezone("Europe/Amsterdam")),
        resolution=dt.timedelta(minutes=15),
    )

    assert df["timestamp"].dtype == "datetime64[us, Europe/Amsterdam]"
