from __future__ import annotations

import datetime as dt
from pathlib import Path
from textwrap import dedent

import pandas as pd
import pytest

from energy_cost.index import DataFrameIndex, Index
from energy_cost.tariff import MeterType, PowerDirection, Tariff


@pytest.fixture
def fake_indexes() -> None:
    timestamps = pd.date_range("2026-03-08 00:00:00+01:00", periods=8, freq="15min")

    Index.register(
        "Belpex15min",
        DataFrameIndex(pd.DataFrame({"timestamp": timestamps, "value": [10, 20, 30, 40, 50, 60, 70, 80]})),
    )
    Index.register(
        "BelpexRLPO",
        DataFrameIndex(pd.DataFrame({"timestamp": timestamps, "value": [1, 2, 3, 4, 5, 6, 7, 8]})),
    )
    Index.register(
        "SolarAdj",
        DataFrameIndex(pd.DataFrame({"timestamp": timestamps, "value": [5, 10, 15, 20, 25, 30, 35, 40]})),
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
            consumption:
              energy:
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
        direction=PowerDirection.CONSUMPTION,
        start=dt.datetime.fromisoformat("2026-03-08T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2026-03-08T01:00:00+01:00"),
        resolution=dt.timedelta(minutes=15),
    )

    assert out["energy"].tolist() == [2.0, 3.0, 4.0, 5.0]
    assert out["total"].tolist() == [2.0, 3.0, 4.0, 5.0]


def test_single_meter_type_get_cost_for_each_direction(tmp_path: Path, fake_indexes: None) -> None:
    path = _write_yaml(
        tmp_path,
        "many_meters.yml",
        """
        supplier: Demo
        product: ManyMeters
        by_meter_type:
          single_rate:
            consumption:
              energy:
                - start: 2026-03-08T00:00:00+01:00
                  formula:
                    constant_cost: 1.0
                    variable_costs:
                      - index: Belpex15min
                        scalar: 0.1
            injection:
              energy:
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
        direction=PowerDirection.CONSUMPTION,
        start=dt.datetime.fromisoformat("2026-03-08T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2026-03-08T01:00:00+01:00"),
        resolution=dt.timedelta(minutes=15),
    )
    injection_out = tariff.get_cost(
        meter_type=MeterType.SINGLE_RATE,
        direction=PowerDirection.INJECTION,
        start=dt.datetime.fromisoformat("2026-03-08T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2026-03-08T01:00:00+01:00"),
        resolution=dt.timedelta(minutes=15),
    )

    assert single_rate_out["energy"].tolist() == [2.0, 3.0, 4.0, 5.0]
    assert single_rate_out["total"].tolist() == [2.0, 3.0, 4.0, 5.0]
    assert injection_out["energy"].tolist() == [0.5, 1.5, 2.5, 3.5]
    assert injection_out["total"].tolist() == [0.5, 1.5, 2.5, 3.5]


def test_many_meter_types_get_cost_for_each_meter_type(tmp_path: Path, fake_indexes: None) -> None:
    path = _write_yaml(
        tmp_path,
        "many_meter_types.yml",
        """
        supplier: Demo
        product: ManyMeterTypes
        by_meter_type:
          single_rate:
            consumption:
              energy:
                - start: 2026-03-08T00:00:00+01:00
                  formula:
                    constant_cost: 1.0
                    variable_costs:
                      - index: Belpex15min
                        scalar: 0.1
          tou_peak:
            consumption:
              energy:
                - start: 2026-03-08T00:00:00+01:00
                  formula:
                    constant_cost: 0.5
                    variable_costs:
                      - index: Belpex15min
                        scalar: 0.2
        """,
    )

    tariff = Tariff.from_yaml(path)

    single_rate_out = tariff.get_cost(
        meter_type=MeterType.SINGLE_RATE,
        direction=PowerDirection.CONSUMPTION,
        start=dt.datetime.fromisoformat("2026-03-08T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2026-03-08T01:00:00+01:00"),
        resolution=dt.timedelta(minutes=15),
    )
    tou_peak_out = tariff.get_cost(
        meter_type=MeterType.TOU_PEAK,
        direction=PowerDirection.CONSUMPTION,
        start=dt.datetime.fromisoformat("2026-03-08T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2026-03-08T01:00:00+01:00"),
        resolution=dt.timedelta(minutes=15),
    )

    assert single_rate_out["energy"].tolist() == [2.0, 3.0, 4.0, 5.0]
    assert single_rate_out["total"].tolist() == [2.0, 3.0, 4.0, 5.0]
    assert tou_peak_out["energy"].tolist() == [2.5, 4.5, 6.5, 8.5]
    assert tou_peak_out["total"].tolist() == [2.5, 4.5, 6.5, 8.5]


def test_formulas_changing_over_time_get_cost(tmp_path: Path, fake_indexes: None) -> None:
    path = _write_yaml(
        tmp_path,
        "formulas_over_time.yml",
        """
        supplier: Demo
        product: TimeChange
        by_meter_type:
          single_rate:
            consumption:
              energy:
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
        direction=PowerDirection.CONSUMPTION,
        start=dt.datetime.fromisoformat("2026-03-08T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2026-03-08T01:00:00+01:00"),
        resolution=dt.timedelta(minutes=15),
    )

    assert out["energy"].tolist() == [2.0, 3.0, 3.5, 4.0]
    assert out["total"].tolist() == [2.0, 3.0, 3.5, 4.0]


def test_formula_with_many_indexes_get_cost(tmp_path: Path, fake_indexes: None) -> None:
    path = _write_yaml(
        tmp_path,
        "many_indexes.yml",
        """
        supplier: Demo
        product: ManyIndexes
        by_meter_type:
          single_rate:
            consumption:
              energy:
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
        direction=PowerDirection.CONSUMPTION,
        start=dt.datetime.fromisoformat("2026-03-08T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2026-03-08T01:00:00+01:00"),
        resolution=dt.timedelta(minutes=15),
    )

    assert out["energy"].tolist() == pytest.approx([1.45, 2.4, 3.35, 4.3])
    assert out["total"].tolist() == pytest.approx([1.45, 2.4, 3.35, 4.3])


def test_defaults_injection_shared_across_meter_types(tmp_path: Path, fake_indexes: None) -> None:
    path = _write_yaml(
        tmp_path,
        "defaults_injection.yml",
        """
        supplier: Demo
        product: DefaultsInjection
        defaults:
          injection:
            energy:
              - start: 2026-03-08T00:00:00+01:00
                formula:
                  constant_cost: -0.5
                  variable_costs:
                    - index: SolarAdj
                      scalar: 0.2
          consumption:
            chp_certificates:
              - start: 2026-03-08T00:00:00+01:00
                formula:
                  constant_cost: 2.0
            renewable_certificates:
              - start: 2026-03-08T00:00:00+01:00
                formula:
                  constant_cost: 3.0
        by_meter_type:
          single_rate:
            consumption:
              energy:
                - start: 2026-03-08T00:00:00+01:00
                  formula:
                    constant_cost: 1.0
                    variable_costs:
                      - index: Belpex15min
                        scalar: 0.1
          tou_peak:
            consumption:
              energy:
                - start: 2026-03-08T00:00:00+01:00
                  formula:
                    constant_cost: 0.5
                    variable_costs:
                      - index: Belpex15min
                        scalar: 0.2
        """,
    )

    tariff = Tariff.from_yaml(path)

    # Both meter types share the same injection formula from defaults
    single_injection = tariff.get_cost(
        meter_type=MeterType.SINGLE_RATE,
        direction=PowerDirection.INJECTION,
        start=dt.datetime.fromisoformat("2026-03-08T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2026-03-08T01:00:00+01:00"),
        resolution=dt.timedelta(minutes=15),
    )
    tou_injection = tariff.get_cost(
        meter_type=MeterType.TOU_PEAK,
        direction=PowerDirection.INJECTION,
        start=dt.datetime.fromisoformat("2026-03-08T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2026-03-08T01:00:00+01:00"),
        resolution=dt.timedelta(minutes=15),
    )

    assert single_injection["energy"].tolist() == [0.5, 1.5, 2.5, 3.5]
    assert single_injection["total"].tolist() == [0.5, 1.5, 2.5, 3.5]
    assert tou_injection["energy"].tolist() == [0.5, 1.5, 2.5, 3.5]
    assert tou_injection["total"].tolist() == [0.5, 1.5, 2.5, 3.5]

    # Consumption merges defaults (wkk, green) with meter-specific energy
    single_consumption = tariff.get_cost(
        start=dt.datetime.fromisoformat("2026-03-08T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2026-03-08T01:00:00+01:00"),
    )

    assert list(single_consumption.columns) == [
        "timestamp",
        "chp_certificates",
        "renewable_certificates",
        "energy",
        "total",
    ]
    assert single_consumption["energy"].tolist() == [2.0, 3.0, 4.0, 5.0]
    assert single_consumption["chp_certificates"].tolist() == [2.0, 2.0, 2.0, 2.0]
    assert single_consumption["renewable_certificates"].tolist() == [3.0, 3.0, 3.0, 3.0]
    assert single_consumption["total"].tolist() == [7.0, 8.0, 9.0, 10.0]
