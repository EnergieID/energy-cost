from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from energy_cost.index import Index


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
