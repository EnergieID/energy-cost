from __future__ import annotations

import datetime as dt
from pathlib import Path
from textwrap import dedent

import pandas as pd
import pytest

from energy_cost.index import DataFrameIndex, Index
from energy_cost.meter import CostGroup, Meter, TimeseriesFrame
from energy_cost.tariff import Tariff


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


def test_single_segment_consumption(tmp_path: Path, fake_indexes: None) -> None:
    """Single versioned segment: consumption with a constant + variable cost."""
    path = _write_yaml(
        tmp_path,
        "single.yml",
        """
        - start: 2026-03-08T00:00:00+01:00
          consumption:
            energy:
              constant_cost: 1.0
              variable_costs:
                - index: Belpex15min
                  scalar: 0.1
        """,
    )

    tariff = Tariff.from_yaml(path)
    out = tariff.get_values(
        start=dt.datetime.fromisoformat("2026-03-08T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2026-03-08T01:00:00+01:00"),
        output_resolution=dt.timedelta(minutes=15),
    )

    assert out is not None
    assert out["energy"].tolist() == [2.0, 3.0, 4.0, 5.0]
    assert out["total"].tolist() == [2.0, 3.0, 4.0, 5.0]


def test_injection_and_consumption_separate_formulas(tmp_path: Path, fake_indexes: None) -> None:
    """Injection and consumption use separate formulas."""
    path = _write_yaml(
        tmp_path,
        "directions.yml",
        """
        - start: 2026-03-08T00:00:00+01:00
          consumption:
            energy:
              constant_cost: 1.0
              variable_costs:
                - index: Belpex15min
                  scalar: 0.1
          injection:
            energy:
              constant_cost: -0.5
              variable_costs:
                - index: SolarAdj
                  scalar: 0.2
        """,
    )

    tariff = Tariff.from_yaml(path)

    consumption_out = tariff.get_values(
        start=dt.datetime.fromisoformat("2026-03-08T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2026-03-08T01:00:00+01:00"),
        output_resolution=dt.timedelta(minutes=15),
        cost_group=CostGroup.CONSUMPTION,
    )
    injection_out = tariff.get_values(
        start=dt.datetime.fromisoformat("2026-03-08T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2026-03-08T01:00:00+01:00"),
        output_resolution=dt.timedelta(minutes=15),
        cost_group=CostGroup.INJECTION,
    )

    assert consumption_out is not None
    assert injection_out is not None
    assert consumption_out["energy"].tolist() == [2.0, 3.0, 4.0, 5.0]
    assert consumption_out["total"].tolist() == [2.0, 3.0, 4.0, 5.0]
    assert injection_out["energy"].tolist() == [0.5, 1.5, 2.5, 3.5]
    assert injection_out["total"].tolist() == [0.5, 1.5, 2.5, 3.5]


def test_versioned_segments_switch_at_boundary(tmp_path: Path, fake_indexes: None) -> None:
    """The correct segment formula is used on each side of the segment boundary."""
    path = _write_yaml(
        tmp_path,
        "versioned.yml",
        """
        - start: 2026-03-08T00:00:00+01:00
          consumption:
            energy:
              constant_cost: 1.0
              variable_costs:
                - index: Belpex15min
                  scalar: 0.1
        - start: 2026-03-08T00:30:00+01:00
          consumption:
            energy:
              constant_cost: 2.0
              variable_costs:
                - index: Belpex15min
                  scalar: 0.05
        """,
    )

    tariff = Tariff.from_yaml(path)
    out = tariff.get_values(
        start=dt.datetime.fromisoformat("2026-03-08T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2026-03-08T01:00:00+01:00"),
        output_resolution=dt.timedelta(minutes=15),
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
    out = tariff.get_values(
        start=dt.datetime.fromisoformat("2026-03-08T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2026-03-08T01:00:00+01:00"),
        output_resolution=dt.timedelta(minutes=15),
    )

    assert out is not None
    assert out["energy"].tolist() == pytest.approx([1.45, 2.4, 3.35, 4.3])
    assert out["total"].tolist() == pytest.approx([1.45, 2.4, 3.35, 4.3])


def test_multiple_named_cost_types(tmp_path: Path, fake_indexes: None) -> None:
    """Named cost types (energy, certificates, etc.) each appear as their own column."""
    path = _write_yaml(
        tmp_path,
        "cost_types.yml",
        """
        - start: 2026-03-08T00:00:00+01:00
          consumption:
            chp_certificates:
              constant_cost: 2.0
            renewable_certificates:
              constant_cost: 3.0
            energy:
              constant_cost: 1.0
              variable_costs:
                - index: Belpex15min
                  scalar: 0.1
        """,
    )

    tariff = Tariff.from_yaml(path)
    out = tariff.get_values(
        start=dt.datetime.fromisoformat("2026-03-08T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2026-03-08T01:00:00+01:00"),
        output_resolution=dt.timedelta(minutes=15),
    )

    assert out is not None
    assert set(out.columns) == {"timestamp", "chp_certificates", "renewable_certificates", "energy", "total"}
    assert out["energy"].tolist() == [2.0, 3.0, 4.0, 5.0]
    assert out["chp_certificates"].tolist() == [2.0, 2.0, 2.0, 2.0]
    assert out["renewable_certificates"].tolist() == [3.0, 3.0, 3.0, 3.0]
    assert out["total"].tolist() == [7.0, 8.0, 9.0, 10.0]


def test_fixed_costs_are_prorated_correctly(tmp_path: Path) -> None:
    """Fixed costs are prorated for the queried interval."""
    path = _write_yaml(
        tmp_path,
        "fixed.yml",
        """
        - start: 2026-03-08T00:00:00+01:00
          fixed:
            admin:
              period: P1D
              constant_cost: 24.0
            billing:
              period: P1D
              constant_cost: 12.0
        """,
    )

    tariff = Tariff.from_yaml(path)

    consumption = Meter(
        power=TimeseriesFrame(
            pd.DataFrame(
                {"timestamp": pd.date_range("2026-03-08 00:00:00+01:00", periods=4, freq="15min"), "value": [0.0] * 4}
            )
        )
    )

    result = tariff.apply(
        consumption=consumption,
        injection=None,
        start=dt.datetime.fromisoformat("2026-03-08T00:00:00+01:00"),
        end=dt.datetime.fromisoformat("2026-03-08T01:00:00+01:00"),
        output_resolution=dt.timedelta(hours=1),
    )

    # 1 hour = 1/24 of a day; admin: 24 * (1/24) = 1.0; billing: 12 * (1/24) = 0.5
    assert result is not None
    assert result[("fixed", "admin")].sum() == pytest.approx(1.0)
    assert result[("fixed", "billing")].sum() == pytest.approx(0.5)
