import datetime as dt
from os import environ

from isodate import Duration

from energy_cost.data.be.synergrid_load_profile_index import SynergridLoadProfileIndex
from energy_cost.index.cached_index import CachedIndex
from energy_cost.index.entsoe_day_ahead_index import EntsoeDayAheadIndex
from energy_cost.index.load_profile_index import LoadProfileIndex


def test_synergrid_load_profile_index_get_values() -> None:
    index = SynergridLoadProfileIndex(profile="RLP0N", resolution=dt.timedelta(minutes=15))
    start = dt.datetime(2025, 2, 1)
    end = dt.datetime(2025, 2, 2)
    timezone = dt.timezone(dt.timedelta(hours=1))  # CET

    df = index.get_values(start, end, dt.timedelta(minutes=15), timezone)

    assert not df.empty
    assert "timestamp" in df.columns
    assert "value" in df.columns


def test_cached_synegrid_load_profile_index_get_values() -> None:
    index = CachedIndex(
        SynergridLoadProfileIndex(profile="RLP0N", resolution=dt.timedelta(minutes=15)),
        "foo",
    )
    start = dt.datetime(2025, 2, 1)
    end = dt.datetime(2025, 2, 2)
    timezone = dt.timezone(dt.timedelta(hours=1))  # CET

    df = index.get_values(start, end, dt.timedelta(minutes=15), timezone)

    assert not df.empty
    assert "timestamp" in df.columns
    assert "value" in df.columns


def test_load_profile_index() -> None:
    load_profile_index = CachedIndex(
        SynergridLoadProfileIndex(profile="RLP0N", resolution=dt.timedelta(minutes=15)),
        "foo",
    )
    data_profile_index = CachedIndex(
        EntsoeDayAheadIndex("BE", environ.get("ENTSOE_API_KEY", ""), resolution=dt.timedelta(minutes=15)),
        "bar",
    )
    index = LoadProfileIndex(load_profile_index, data_profile_index, resolution=Duration(months=1))

    start = dt.datetime(2025, 2, 1)
    end = dt.datetime(2025, 7, 1)
    timezone = dt.timezone(dt.timedelta(hours=1))  # CET

    df = index.get_values(start, end, Duration(months=1), timezone)

    assert not df.empty
    assert "timestamp" in df.columns
    assert "value" in df.columns
