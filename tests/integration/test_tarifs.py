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


def test_single_segment_single_rate_consumption(tmp_path: Path, fake_indexes: None) -> None:
    """Single versioned segment: single_rate consumption with a constant + variable cost."""
    path = _write_yaml(
        tmp_path,
        "single.yml",
        """
        - start: 2026-03-08T00:00:00+01:00
          consumption:
            single_rate:
              energy:
                constant_cost: 1.0
                variable_costs:
                  - index: Belpex15min
                    scalar: 0.1
        """,
    )

    tariff = Tariff.from_yaml(path)
    out = tariff.get_energy_cost(
        meter_type=MeterType.SINGLE_RATE,
        direction=PowerDirection.CONSUMPTION,
        start=dt.datetime.fromisoformat("2026-03-08T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2026-03-08T01:00:00+01:00"),
        resolution=dt.timedelta(minutes=15),
    )

    assert out is not None
    assert out["energy"].tolist() == [2.0, 3.0, 4.0, 5.0]
    assert out["total"].tolist() == [2.0, 3.0, 4.0, 5.0]


def test_injection_and_consumption_separate_formulas(tmp_path: Path, fake_indexes: None) -> None:
    """Injection and consumption use separate formulas via ``all`` key."""
    path = _write_yaml(
        tmp_path,
        "directions.yml",
        """
        - start: 2026-03-08T00:00:00+01:00
          consumption:
            all:
              energy:
                constant_cost: 1.0
                variable_costs:
                  - index: Belpex15min
                    scalar: 0.1
          injection:
            all:
              energy:
                constant_cost: -0.5
                variable_costs:
                  - index: SolarAdj
                    scalar: 0.2
        """,
    )

    tariff = Tariff.from_yaml(path)

    consumption_out = tariff.get_energy_cost(
        meter_type=MeterType.SINGLE_RATE,
        direction=PowerDirection.CONSUMPTION,
        start=dt.datetime.fromisoformat("2026-03-08T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2026-03-08T01:00:00+01:00"),
        resolution=dt.timedelta(minutes=15),
    )
    injection_out = tariff.get_energy_cost(
        meter_type=MeterType.SINGLE_RATE,
        direction=PowerDirection.INJECTION,
        start=dt.datetime.fromisoformat("2026-03-08T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2026-03-08T01:00:00+01:00"),
        resolution=dt.timedelta(minutes=15),
    )

    assert consumption_out is not None
    assert injection_out is not None
    assert consumption_out["energy"].tolist() == [2.0, 3.0, 4.0, 5.0]
    assert consumption_out["total"].tolist() == [2.0, 3.0, 4.0, 5.0]
    assert injection_out["energy"].tolist() == [0.5, 1.5, 2.5, 3.5]
    assert injection_out["total"].tolist() == [0.5, 1.5, 2.5, 3.5]


def test_multiple_meter_types_with_different_formulas(tmp_path: Path, fake_indexes: None) -> None:
    """Meter-type-specific keys resolve to different formulas for each meter type."""
    path = _write_yaml(
        tmp_path,
        "meter_types.yml",
        """
        - start: 2026-03-08T00:00:00+01:00
          consumption:
            single_rate:
              energy:
                constant_cost: 1.0
                variable_costs:
                  - index: Belpex15min
                    scalar: 0.1
            tou_peak:
              energy:
                constant_cost: 0.5
                variable_costs:
                  - index: Belpex15min
                    scalar: 0.2
        """,
    )

    tariff = Tariff.from_yaml(path)

    single_rate_out = tariff.get_energy_cost(
        meter_type=MeterType.SINGLE_RATE,
        direction=PowerDirection.CONSUMPTION,
        start=dt.datetime.fromisoformat("2026-03-08T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2026-03-08T01:00:00+01:00"),
        resolution=dt.timedelta(minutes=15),
    )
    tou_peak_out = tariff.get_energy_cost(
        meter_type=MeterType.TOU_PEAK,
        direction=PowerDirection.CONSUMPTION,
        start=dt.datetime.fromisoformat("2026-03-08T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2026-03-08T01:00:00+01:00"),
        resolution=dt.timedelta(minutes=15),
    )

    assert single_rate_out is not None
    assert tou_peak_out is not None
    assert single_rate_out["energy"].tolist() == [2.0, 3.0, 4.0, 5.0]
    assert single_rate_out["total"].tolist() == [2.0, 3.0, 4.0, 5.0]
    assert tou_peak_out["energy"].tolist() == [2.5, 4.5, 6.5, 8.5]
    assert tou_peak_out["total"].tolist() == [2.5, 4.5, 6.5, 8.5]


def test_versioned_segments_switch_at_boundary(tmp_path: Path, fake_indexes: None) -> None:
    """The correct segment formula is used on each side of the segment boundary."""
    path = _write_yaml(
        tmp_path,
        "versioned.yml",
        """
        - start: 2026-03-08T00:00:00+01:00
          consumption:
            all:
              energy:
                constant_cost: 1.0
                variable_costs:
                  - index: Belpex15min
                    scalar: 0.1
        - start: 2026-03-08T00:30:00+01:00
          consumption:
            all:
              energy:
                constant_cost: 2.0
                variable_costs:
                  - index: Belpex15min
                    scalar: 0.05
        """,
    )

    tariff = Tariff.from_yaml(path)
    out = tariff.get_energy_cost(
        meter_type=MeterType.SINGLE_RATE,
        direction=PowerDirection.CONSUMPTION,
        start=dt.datetime.fromisoformat("2026-03-08T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2026-03-08T01:00:00+01:00"),
        resolution=dt.timedelta(minutes=15),
    )

    assert out is not None
    assert out["energy"].tolist() == [2.0, 3.0, 3.5, 4.0]
    assert out["total"].tolist() == [2.0, 3.0, 3.5, 4.0]


def test_multiple_variable_cost_indexes(tmp_path: Path, fake_indexes: None) -> None:
    path = _write_yaml(
        tmp_path,
        "many_indexes.yml",
        """
        - start: 2026-03-08T00:00:00+01:00
          consumption:
            all:
              energy:
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
    out = tariff.get_energy_cost(
        start=dt.datetime.fromisoformat("2026-03-08T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2026-03-08T01:00:00+01:00"),
        resolution=dt.timedelta(minutes=15),
    )

    assert out is not None
    assert out["energy"].tolist() == pytest.approx([1.45, 2.4, 3.35, 4.3])
    assert out["total"].tolist() == pytest.approx([1.45, 2.4, 3.35, 4.3])


def test_all_key_shared_costs_with_meter_specific_energy(tmp_path: Path, fake_indexes: None) -> None:
    """``all`` provides shared cost types; meter-type-specific keys add/override energy per meter."""
    path = _write_yaml(
        tmp_path,
        "all_and_specific.yml",
        """
        - start: 2026-03-08T00:00:00+01:00
          injection:
            all:
              energy:
                constant_cost: -0.5
                variable_costs:
                  - index: SolarAdj
                    scalar: 0.2
          consumption:
            all:
              chp_certificates:
                constant_cost: 2.0
              renewable_certificates:
                constant_cost: 3.0
            single_rate:
              energy:
                constant_cost: 1.0
                variable_costs:
                  - index: Belpex15min
                    scalar: 0.1
            tou_peak:
              energy:
                constant_cost: 0.5
                variable_costs:
                  - index: Belpex15min
                    scalar: 0.2
        """,
    )

    tariff = Tariff.from_yaml(path)

    # Injection is shared across meter types via ``all``
    single_injection = tariff.get_energy_cost(
        meter_type=MeterType.SINGLE_RATE,
        direction=PowerDirection.INJECTION,
        start=dt.datetime.fromisoformat("2026-03-08T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2026-03-08T01:00:00+01:00"),
        resolution=dt.timedelta(minutes=15),
    )
    tou_injection = tariff.get_energy_cost(
        meter_type=MeterType.TOU_PEAK,
        direction=PowerDirection.INJECTION,
        start=dt.datetime.fromisoformat("2026-03-08T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2026-03-08T01:00:00+01:00"),
        resolution=dt.timedelta(minutes=15),
    )

    assert single_injection is not None
    assert tou_injection is not None
    assert single_injection["energy"].tolist() == [0.5, 1.5, 2.5, 3.5]
    assert single_injection["total"].tolist() == [0.5, 1.5, 2.5, 3.5]
    assert tou_injection["energy"].tolist() == [0.5, 1.5, 2.5, 3.5]
    assert tou_injection["total"].tolist() == [0.5, 1.5, 2.5, 3.5]

    # Consumption: ``all`` (chp + renewable) merged with single_rate energy
    single_consumption = tariff.get_energy_cost(
        meter_type=MeterType.SINGLE_RATE,
        direction=PowerDirection.CONSUMPTION,
        start=dt.datetime.fromisoformat("2026-03-08T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2026-03-08T01:00:00+01:00"),
        resolution=dt.timedelta(minutes=15),
    )

    assert single_consumption is not None
    assert set(single_consumption.columns) == {
        "timestamp",
        "chp_certificates",
        "renewable_certificates",
        "energy",
        "total",
    }
    assert single_consumption["energy"].tolist() == [2.0, 3.0, 4.0, 5.0]
    assert single_consumption["chp_certificates"].tolist() == [2.0, 2.0, 2.0, 2.0]
    assert single_consumption["renewable_certificates"].tolist() == [3.0, 3.0, 3.0, 3.0]
    assert single_consumption["total"].tolist() == [7.0, 8.0, 9.0, 10.0]


def test_periodic_costs_are_prorated_correctly(tmp_path: Path) -> None:
    """Periodic costs are prorated for the queried interval."""
    path = _write_yaml(
        tmp_path,
        "periodic.yml",
        """
        - start: 2026-03-08T00:00:00+01:00
          consumption:
            all:
              energy:
                constant_cost: 1.0
          periodic:
            admin:
              period: daily
              constant_cost: 24.0
            billing:
              period: daily
              constant_cost: 12.0
        """,
    )

    tariff = Tariff.from_yaml(path)
    costs = tariff.get_periodic_cost(
        start=dt.datetime.fromisoformat("2026-03-08T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2026-03-08T01:00:00+01:00"),
    )

    # 1 hour = 1/24 of a day; admin: 24 * (1/24) = 1.0; billing: 12 * (1/24) = 0.5
    assert costs == pytest.approx({"admin": 1.0, "billing": 0.5})
