from __future__ import annotations

import datetime as dt
from pathlib import Path
from textwrap import dedent

import pandas as pd
import pytest

from energy_cost.index import Index
from energy_cost.tariff import MeterType, Tariff


class FakeDataFrameIndex(Index):
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def get_values(self, start: dt.datetime, end: dt.datetime, resolution: dt.timedelta) -> pd.DataFrame:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        mask = (self.df["timestamp"] >= start_ts) & (self.df["timestamp"] < end_ts)
        return self.df.loc[mask, ["timestamp", "value"]].reset_index(drop=True)


@pytest.fixture
def fake_indexes() -> None:
    timestamps = pd.date_range("2026-03-08 00:00:00+01:00", periods=8, freq="15min")

    Index.register(
        "Belpex15min",
        FakeDataFrameIndex(pd.DataFrame({"timestamp": timestamps, "value": [10, 20, 30, 40, 50, 60, 70, 80]})),
    )
    Index.register(
        "BelpexRLPO",
        FakeDataFrameIndex(pd.DataFrame({"timestamp": timestamps, "value": [1, 2, 3, 4, 5, 6, 7, 8]})),
    )
    Index.register(
        "SolarAdj",
        FakeDataFrameIndex(pd.DataFrame({"timestamp": timestamps, "value": [5, 10, 15, 20, 25, 30, 35, 40]})),
    )


def _write_yaml(tmp_path: Path, file_name: str, content: str) -> Path:
    path = tmp_path / file_name
    path.write_text(dedent(content).strip() + "\n", encoding="utf-8")
    return path


def test_single_meter_single_timed_price_formula_get_cost(tmp_path: Path, fake_indexes: None) -> None:
    path = _write_yaml(
        tmp_path,
        "single_meter.yml",
        """
        supplier: Demo
        product: Single
        by_meter_type:
          single_rate:
            - start: 2026-03-08T00:00:00+01:00
              formula:
                constant_cost: 1.0
                variable_costs:
                  - index: Belpex15min
                    scalar: 0.1
        """,
    )

    tariff = Tariff.from_yaml(path)
    out = tariff.get_cost(
        meter_type=MeterType.SINGLE_RATE,
        start=dt.datetime.fromisoformat("2026-03-08T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2026-03-08T01:00:00+01:00"),
        resolution=dt.timedelta(minutes=15),
    )

    assert out["value"].tolist() == [2.0, 3.0, 4.0, 5.0]


def test_many_meters_get_cost_for_each_meter_type(tmp_path: Path, fake_indexes: None) -> None:
    path = _write_yaml(
        tmp_path,
        "many_meters.yml",
        """
        supplier: Demo
        product: ManyMeters
        by_meter_type:
          single_rate:
            - start: 2026-03-08T00:00:00+01:00
              formula:
                constant_cost: 1.0
                variable_costs:
                  - index: Belpex15min
                    scalar: 0.1
          injection:
            - start: 2026-03-08T00:00:00+01:00
              formula:
                constant_cost: -0.5
                variable_costs:
                  - index: SolarAdj
                    scalar: 0.2
        """,
    )

    tariff = Tariff.from_yaml(path)

    single_rate_out = tariff.get_cost(
        meter_type=MeterType.SINGLE_RATE,
        start=dt.datetime.fromisoformat("2026-03-08T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2026-03-08T01:00:00+01:00"),
        resolution=dt.timedelta(minutes=15),
    )
    injection_out = tariff.get_cost(
        meter_type=MeterType.INJECTION,
        start=dt.datetime.fromisoformat("2026-03-08T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2026-03-08T01:00:00+01:00"),
        resolution=dt.timedelta(minutes=15),
    )

    assert single_rate_out["value"].tolist() == [2.0, 3.0, 4.0, 5.0]
    assert injection_out["value"].tolist() == [0.5, 1.5, 2.5, 3.5]


def test_formulas_changing_over_time_get_cost(tmp_path: Path, fake_indexes: None) -> None:
    path = _write_yaml(
        tmp_path,
        "formulas_over_time.yml",
        """
        supplier: Demo
        product: TimeChange
        by_meter_type:
          single_rate:
            - start: 2026-03-08T00:00:00+01:00
              formula:
                constant_cost: 1.0
                variable_costs:
                  - index: Belpex15min
                    scalar: 0.1
            - start: 2026-03-08T00:30:00+01:00
              formula:
                constant_cost: 2.0
                variable_costs:
                  - index: Belpex15min
                    scalar: 0.05
        """,
    )

    tariff = Tariff.from_yaml(path)
    out = tariff.get_cost(
        meter_type=MeterType.SINGLE_RATE,
        start=dt.datetime.fromisoformat("2026-03-08T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2026-03-08T01:00:00+01:00"),
        resolution=dt.timedelta(minutes=15),
    )

    assert out["value"].tolist() == [2.0, 3.0, 3.5, 4.0]


def test_formula_with_many_indexes_get_cost(tmp_path: Path, fake_indexes: None) -> None:
    path = _write_yaml(
        tmp_path,
        "many_indexes.yml",
        """
        supplier: Demo
        product: ManyIndexes
        by_meter_type:
          single_rate:
            - start: 2026-03-08T00:00:00+01:00
              formula:
                constant_cost: 0.5
                variable_costs:
                  - index: Belpex15min
                    scalar: 0.1
                  - index: BelpexRLPO
                    scalar: 0.2
                  - index: SolarAdj
                    scalar: -0.05
        """,
    )

    tariff = Tariff.from_yaml(path)
    out = tariff.get_cost(
        meter_type=MeterType.SINGLE_RATE,
        start=dt.datetime.fromisoformat("2026-03-08T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2026-03-08T01:00:00+01:00"),
        resolution=dt.timedelta(minutes=15),
    )

    assert out["value"].tolist() == pytest.approx([1.45, 2.4, 3.35, 4.3])
