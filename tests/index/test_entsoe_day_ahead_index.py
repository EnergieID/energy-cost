from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

import energy_cost.index.entsoe_day_ahead_index as entsoe_module
from energy_cost.index.entsoe_day_ahead_index import EntsoeDayAheadIndex


class StubEntsoePandasClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.calls: list[tuple[str, pd.Timestamp, pd.Timestamp]] = []

    def query_day_ahead_prices(self, country_code: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
        self.calls.append((country_code, start, end))
        return pd.Series(
            [100.0, 120.0],
            index=pd.date_range("2025-01-01", periods=2, freq="1h"),
        )


def test_init_uses_mocked_entsoe_client(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(entsoe_module, "EntsoePandasClient", StubEntsoePandasClient)

    index = EntsoeDayAheadIndex(country_code="BE", api_key="fake-key")

    assert isinstance(index.client, StubEntsoePandasClient)
    assert index.client.api_key == "fake-key"
    assert index.resolution == dt.timedelta(minutes=15)


@pytest.mark.parametrize("resolution", [dt.timedelta(hours=2), dt.timedelta(minutes=7)])
def test_get_values_raises_for_invalid_resolution(monkeypatch: pytest.MonkeyPatch, resolution: dt.timedelta) -> None:
    monkeypatch.setattr(entsoe_module, "EntsoePandasClient", StubEntsoePandasClient)
    index = EntsoeDayAheadIndex(country_code="BE", api_key="fake-key", resolution=dt.timedelta(hours=1))

    with pytest.raises(ValueError):
        index.get_values(
            start=dt.datetime(2025, 1, 1, 0, 0),
            end=dt.datetime(2025, 1, 1, 2, 0),
            resolution=resolution,
        )


def test_get_values_resamples_and_queries_with_expected_arguments(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(entsoe_module, "EntsoePandasClient", StubEntsoePandasClient)
    index = EntsoeDayAheadIndex(country_code="BE", api_key="fake-key", resolution=dt.timedelta(hours=1))

    start = dt.datetime(2025, 1, 1, 0, 0)
    end = dt.datetime(2025, 1, 1, 2, 0)

    out = index.get_values(start=start, end=end, resolution=dt.timedelta(minutes=15))

    assert isinstance(index.client, StubEntsoePandasClient)
    assert list(out.columns) == ["timestamp", "value"]
    assert out["timestamp"].tolist() == list(pd.date_range("2025-01-01", periods=8, freq="15min", tz=dt.UTC))
    assert out["value"].tolist() == [100.0, 100.0, 100.0, 100.0, 120.0, 120.0, 120.0, 120.0]


def test_get_values_supports_finer_divisor_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(entsoe_module, "EntsoePandasClient", StubEntsoePandasClient)
    index = EntsoeDayAheadIndex(country_code="BE", api_key="fake-key", resolution=dt.timedelta(hours=1))

    out = index.get_values(
        start=dt.datetime(2025, 1, 1, 0, 0),
        end=dt.datetime(2025, 1, 1, 2, 0),
        resolution=dt.timedelta(minutes=5),
    )

    assert out.iloc[0]["timestamp"] == pd.Timestamp("2025-01-01 00:00:00", tz=dt.UTC)
    assert out.iloc[0]["value"] == 100.0
    assert out.iloc[-1]["timestamp"] == pd.Timestamp("2025-01-01 01:55:00", tz=dt.UTC)
    assert out.iloc[-1]["value"] == 120.0
