import datetime as dt
from os import environ

from isodate import Duration
from pytest import skip

from energy_cost.data.be.electricity.indexes import BelpexRLP0N
from energy_cost.data.be.synergrid_load_profile_index import SynergridLoadProfileIndex
from energy_cost.index.cached_index import CachedIndex
from energy_cost.index.entsoe_day_ahead_index import EntsoeDayAheadIndex
from energy_cost.index.load_profile_index import LoadProfileIndex


def test_synergrid_load_profile_index_get_values() -> None:
    index = SynergridLoadProfileIndex(profile="RLP0N")
    start = dt.datetime(2025, 2, 1)
    end = dt.datetime(2025, 2, 2)
    timezone = dt.timezone(dt.timedelta(hours=1))  # CET

    df = index.get_values(start, end, dt.timedelta(minutes=15), timezone)

    assert not df.empty
    assert "timestamp" in df.columns
    assert "value" in df.columns


def test_synergrid_load_profile_index_region_column_selection() -> None:
    start = dt.datetime(2025, 6, 1)
    end = dt.datetime(2025, 6, 2)
    timezone = dt.timezone(dt.timedelta(hours=1))

    flanders = SynergridLoadProfileIndex(profile="RLP0N", region="flanders")
    wallonia = SynergridLoadProfileIndex(profile="RLP0N", region="wallonia")

    flanders_df = flanders.get_values(start, end, dt.timedelta(minutes=15), timezone)
    wallonia_df = wallonia.get_values(start, end, dt.timedelta(minutes=15), timezone)

    assert not flanders_df.empty
    assert not wallonia_df.empty
    assert not flanders_df["value"].equals(wallonia_df["value"])


def test_synergrid_spp_profile_get_values() -> None:
    index = SynergridLoadProfileIndex(profile="SPP", region="belgium")
    start = dt.datetime(2025, 6, 1)
    end = dt.datetime(2025, 6, 2)
    timezone = dt.timezone(dt.timedelta(hours=1))

    df = index.get_values(start, end, dt.timedelta(minutes=15), timezone)

    assert not df.empty
    assert "timestamp" in df.columns
    assert "value" in df.columns


def test_cached_synegrid_load_profile_index_get_values() -> None:
    index = CachedIndex(
        SynergridLoadProfileIndex(profile="RLP0N"),
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
    if not environ.get("ENTSOE_API_KEY"):
        skip("ENTSOE_API_KEY not set")

    load_profile_index = CachedIndex(
        SynergridLoadProfileIndex(profile="RLP0N"),
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


def test_belpex_rlp0n_index() -> None:
    if not environ.get("ENTSOE_API_KEY"):
        skip("ENTSOE_API_KEY not set")

    index = BelpexRLP0N(entsoe_api_key=environ.get("ENTSOE_API_KEY", ""))

    start = dt.datetime(2025, 1, 1)
    end = dt.datetime(2025, 9, 1)
    timezone = dt.timezone(dt.timedelta(hours=1))  # CET

    df = index.get_values(start, end, Duration(months=1), timezone)

    assert not df.empty
    assert "timestamp" in df.columns
    assert "value" in df.columns
    expected = [70.78, 85.15, 67.52, 63.28, 75.27, 93.75, 131.43, 115.37]
    expected.reverse()
    actual = df["value"].round(2).tolist()
    assert actual == expected
