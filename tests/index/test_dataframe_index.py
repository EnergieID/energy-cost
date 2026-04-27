from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from energy_cost.index import CSVIndex, DataFrameIndex, YAMLIndex


def test_dataframe_index_requires_timestamp_and_value_columns() -> None:
    with pytest.raises(ValueError, match="DataFrame must contain 'timestamp' and 'value' columns"):
        DataFrameIndex(pd.DataFrame({"timestamp": ["2025-01-01T00:00:00Z"]}))


def test_dataframe_index_sorts_and_parses_timestamps() -> None:
    index = DataFrameIndex(
        pd.DataFrame(
            {
                "timestamp": ["2025-01-01T01:00:00Z", "2025-01-01T00:00:00Z"],
                "value": [2.0, 1.0],
            }
        )
    )

    assert index.df["timestamp"].tolist() == [
        pd.Timestamp("2025-01-01T00:00:00Z"),
        pd.Timestamp("2025-01-01T01:00:00Z"),
    ]


def test_get_values_forward_fills_from_last_known_value() -> None:
    index = DataFrameIndex(
        pd.DataFrame(
            {
                "timestamp": ["2025-01-01T00:00:00Z", "2025-01-01T02:00:00Z"],
                "value": [100.0, 200.0],
            }
        )
    )

    out = index.get_values(
        start=dt.datetime(2025, 1, 1, 1, 0, tzinfo=dt.UTC),
        end=dt.datetime(2025, 1, 1, 4, 0, tzinfo=dt.UTC),
        resolution=dt.timedelta(hours=1),
    )

    assert out["timestamp"].tolist() == [
        pd.Timestamp("2025-01-01T01:00:00Z"),
        pd.Timestamp("2025-01-01T02:00:00Z"),
        pd.Timestamp("2025-01-01T03:00:00Z"),
    ]
    assert out["value"].tolist() == [100.0, 200.0, 200.0]


def test_get_values_before_first_datapoint_returns_nan() -> None:
    index = DataFrameIndex(
        pd.DataFrame(
            {
                "timestamp": ["2025-01-01T00:00:00Z"],
                "value": [100.0],
            }
        ),
        resolution=dt.timedelta(hours=1),
    )

    out = index.get_values(
        start=dt.datetime(2024, 12, 31, 23, 0, tzinfo=dt.UTC),
        end=dt.datetime(2025, 1, 1, 1, 0, tzinfo=dt.UTC),
        resolution=dt.timedelta(hours=1),
    )

    assert pd.isna(out.iloc[0]["value"])
    assert out.iloc[1]["value"] == 100.0


def test_csv_and_yaml_index_load_values(tmp_path) -> None:
    csv_path = tmp_path / "index.csv"
    csv_path.write_text("timestamp,value\n2025-01-01T00:00:00Z,10\n", encoding="utf-8")

    yaml_path = tmp_path / "index.yml"
    yaml_path.write_text("- timestamp: 2025-01-01T00:00:00Z\n  value: 12\n", encoding="utf-8")

    csv_index = CSVIndex(str(csv_path), resolution=dt.timedelta(hours=1))
    yaml_index = YAMLIndex(str(yaml_path), resolution=dt.timedelta(hours=1))

    start = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
    end = dt.datetime(2025, 1, 1, 1, tzinfo=dt.UTC)
    resolution = dt.timedelta(hours=1)

    assert csv_index.get_values(start, end, resolution)["value"].tolist() == [10]
    assert yaml_index.get_values(start, end, resolution)["value"].tolist() == [12]


def test_yaml_index_with_mixed_utc_offsets(tmp_path) -> None:
    """YAML timestamps that carry different UTC offsets (e.g. +01:00 and +02:00 across a
    DST boundary) must be normalised to UTC so that get_values returns the correct
    values in the correct order.

    The two entries represent consecutive hours straddling the spring-forward moment:
      2024-03-31T01:45:00+01:00  →  2024-03-31T00:45:00Z
      2024-03-31T03:00:00+02:00  →  2024-03-31T01:00:00Z  (first slot after the gap)
    """
    yaml_path = tmp_path / "mixed.yml"
    yaml_path.write_text(
        "- timestamp: '2024-03-31T01:45:00+01:00'\n"
        "  value: 42.0\n"
        "- timestamp: '2024-03-31T03:00:00+02:00'\n"
        "  value: 99.0\n",
        encoding="utf-8",
    )

    index = YAMLIndex(str(yaml_path), resolution=dt.timedelta(minutes=15))

    # Both timestamps must be stored as UTC and in chronological order.
    assert index.df["timestamp"].tolist() == [
        pd.Timestamp("2024-03-31T00:45:00Z"),
        pd.Timestamp("2024-03-31T01:00:00Z"),
    ]

    # get_values across the whole range must return both entries correctly.
    out = index.get_values(
        start=dt.datetime(2024, 3, 31, 0, 45, tzinfo=dt.UTC),
        end=dt.datetime(2024, 3, 31, 1, 15, tzinfo=dt.UTC),
        resolution=dt.timedelta(minutes=15),
    )

    assert out["timestamp"].tolist() == [
        pd.Timestamp("2024-03-31T00:45:00Z"),
        pd.Timestamp("2024-03-31T01:00:00Z"),
    ]
    assert out["value"].tolist() == [42.0, 99.0]
