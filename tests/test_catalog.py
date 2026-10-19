from __future__ import annotations

import asyncio
import datetime as dt
import subprocess
import sys

import pandas as pd
import pytest
from pydantic import ValidationError

from energy_cost import Catalog, Contract, Meter, Supplier, Tariff, TimeseriesFrame, data
from energy_cost.data import ConnectionType, RegionalData
from energy_cost.data.be import flanders
from energy_cost.formula import IndexAdder, IndexFormula
from energy_cost.index import CSVIndex, DataFrameIndex, Index
from energy_cost.meter import TariffCategory

_TIMESTAMPS = pd.date_range("2025-01-01T00:00:00+01:00", "2025-03-01T00:00:00+01:00", freq="15min", inclusive="left")
_START = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
_END = dt.datetime(2025, 1, 1, 0, 30, tzinfo=dt.UTC)
_QUARTER = dt.timedelta(minutes=15)
_FLANDERS_ELECTRICITY = ("be_flanders", ConnectionType.ELECTRICITY)
_REGION = {
    "region": "be_flanders",
    "connection_type": "electricity",
    "customer_type": "residential",
    "distributor_key": "fluvius_imewo",
}


def _constant_index(value: float) -> DataFrameIndex:
    return DataFrameIndex(pd.DataFrame({"timestamp": _TIMESTAMPS, "value": value}))


def _tariff(constant_cost: float = 0.0, index: str | None = None) -> Tariff:
    consumption: dict = {"constant_cost": constant_cost}
    if index is not None:
        consumption["variable_costs"] = [{"index": index, "scalar": 1.0}]
    return Tariff.model_validate([{"start": "2025-01-01T00:00:00+01:00", "consumption": consumption}])


def _spot_values() -> list[float]:
    formula = IndexFormula(variable_costs=[IndexAdder(index="spot", scalar=1.0)])
    return formula.get_values(_START, _END, _QUARTER)["value"].tolist()


# ---------------------------------------------------------------------------
# Indexes, suppliers and regions resolve against the active catalog
# ---------------------------------------------------------------------------


def test_indexes_resolve_per_catalog() -> None:
    first = Catalog(indexes={"spot": _constant_index(10.0)})
    second = Catalog(indexes={"spot": _constant_index(20.0)})

    with first.active():
        assert _spot_values() == [10.0, 10.0]
    with second.active():
        assert _spot_values() == [20.0, 20.0]


def test_tariff_resolves_index_per_catalog() -> None:
    meter = Meter(measurements=TimeseriesFrame({"timestamp": _TIMESTAMPS[:4], "value": 1.0}))

    totals = []
    for value in (10.0, 20.0):
        with Catalog(indexes={"spot": _constant_index(value)}).active():
            out = _tariff(index="spot").apply(meter, output_resolution=_QUARTER)
            assert out is not None
            totals.append(out[("total", "total")].sum())

    assert totals == [40.0, 80.0]


def test_formula_keeps_index_of_catalog_it_was_created_in() -> None:
    Index.register("spot", _constant_index(99.0))

    with Catalog(indexes={"spot": _constant_index(10.0)}).active():
        formula = IndexFormula(variable_costs=[IndexAdder(index="spot", scalar=1.0)])

    assert formula.get_values(_START, _END, _QUARTER)["value"].tolist() == [10.0, 10.0]


def test_tariff_with_index_of_inactive_catalog_is_rejected() -> None:
    Catalog(indexes={"spot": _constant_index(1.0)})

    with pytest.raises(ValidationError, match="unknown index 'spot'"):
        _tariff(index="spot")


def test_suppliers_resolve_per_catalog() -> None:
    cheap, expensive = _tariff(10.0), _tariff(20.0)
    first = Catalog(suppliers={"acme": Supplier(products={"basic": cheap})})
    second = Catalog(suppliers={"acme": Supplier(products={"basic": expensive})})

    with first.active():
        assert Contract(supplier_key="acme", product_key="basic").supplier is cheap
    with second.active():
        assert Contract(supplier_key="acme", product_key="basic").supplier is expensive


def test_region_resolves_in_catalog_with_region_registry() -> None:
    with Catalog(regions=flanders.registry).active():
        contract = Contract.model_validate(_REGION)

    assert contract.distributor is not None


def test_region_raises_in_empty_catalog() -> None:
    with Catalog().active(), pytest.raises(KeyError, match="be_flanders"):
        Contract.model_validate(_REGION)


def test_region_resolves_through_default_catalog() -> None:
    contract = Contract.model_validate(_REGION)

    assert contract.distributor is not None


# ---------------------------------------------------------------------------
# Class methods act on the current catalog
# ---------------------------------------------------------------------------


def test_class_methods_act_on_active_catalog() -> None:
    catalog = Catalog()
    index = _constant_index(1.0)
    supplier = Supplier(products={})
    regional = flanders.electricity.data

    with catalog.active():
        Index.register("spot", index)
        Supplier.register("acme", supplier)
        RegionalData.register(_FLANDERS_ELECTRICITY, regional)

        assert Index.get("spot") is index
        assert Supplier.get("acme") is supplier
        assert RegionalData.get(_FLANDERS_ELECTRICITY) is regional
        assert list(RegionalData.items()) == [(_FLANDERS_ELECTRICITY, regional)]

    assert "spot" in catalog.indexes
    assert "acme" in catalog.suppliers
    assert "spot" not in Catalog.current().indexes
    assert "acme" not in Catalog.current().suppliers


def test_class_methods_act_on_default_catalog_outside_block() -> None:
    index = _constant_index(1.0)
    with Catalog().active():
        pass
    Index.register("spot", index)

    assert Catalog.current().indexes.get("spot") is index
    assert dict(RegionalData.items()) == data.registry


def test_clear_inside_block_leaves_default_untouched() -> None:
    Index.register("spot", _constant_index(1.0))

    with Catalog(indexes={"other": _constant_index(2.0)}).active():
        Index.clear()
        assert list(Index.items()) == []

    assert "spot" in Catalog.current().indexes


def test_no_fallback_to_default_catalog() -> None:
    Index.register("spot", _constant_index(1.0))
    Supplier.register("acme", Supplier(products={"basic": _tariff()}))

    with Catalog().active():
        with pytest.raises(KeyError, match="spot"):
            Index.get("spot")
        with pytest.raises(KeyError, match="acme"):
            Contract(supplier_key="acme", product_key="basic")
        with pytest.raises(ValidationError, match="unknown index 'spot'"):
            _tariff(index="spot")


def test_subclass_shares_registry_with_base() -> None:
    index = _constant_index(1.0)
    CSVIndex.register("spot", index)

    assert Index.get("spot") is index


# ---------------------------------------------------------------------------
# Activation scope
# ---------------------------------------------------------------------------


def test_current_returns_active_catalog() -> None:
    default = Catalog.current()
    catalog = Catalog()

    with catalog.active() as active:
        assert active is catalog
        assert Catalog.current() is catalog

    assert Catalog.current() is default


def test_nested_blocks_restore_outer_catalog() -> None:
    outer, inner = Catalog(), Catalog()

    with outer.active():
        with inner.active():
            assert Catalog.current() is inner
        assert Catalog.current() is outer

        with pytest.raises(RuntimeError), inner.active():
            raise RuntimeError
        assert Catalog.current() is outer


def test_active_catalog_carries_into_to_thread() -> None:
    catalog = Catalog()

    async def main() -> Catalog:
        with catalog.active():
            return await asyncio.to_thread(Catalog.current)

    assert asyncio.run(main()) is catalog


def test_concurrent_tasks_see_their_own_catalog() -> None:
    async def spot_values(value: float) -> list[float]:
        with Catalog(indexes={"spot": _constant_index(value)}).active():
            await asyncio.sleep(0)
            first = _spot_values()
            await asyncio.sleep(0)
            return first + _spot_values()

    async def main() -> list[list[float]]:
        return list(await asyncio.gather(spot_values(10.0), spot_values(20.0)))

    assert asyncio.run(main()) == [[10.0] * 4, [20.0] * 4]


# ---------------------------------------------------------------------------
# Bundled regions
# ---------------------------------------------------------------------------


def test_region_registries_nest() -> None:
    flanders_gas = ("be_flanders", ConnectionType.GAS)

    assert flanders.electricity.registry == {_FLANDERS_ELECTRICITY: flanders.electricity.data}
    assert flanders.registry == {**flanders.electricity.registry, **flanders.gas.registry}
    assert set(data.be.registry) == set(data.registry) == {_FLANDERS_ELECTRICITY, flanders_gas}


def test_catalog_copies_given_entries() -> None:
    catalog = Catalog(regions=flanders.electricity.registry)
    catalog.regions.update(flanders.gas.registry)

    assert list(flanders.electricity.registry) == [_FLANDERS_ELECTRICITY]
    assert len(catalog.regions) == 2


def test_importing_data_inside_block_does_not_fill_active_catalog() -> None:
    script = """
import importlib

from energy_cost import Catalog
import energy_cost.data

catalog = Catalog()
with catalog.active():
    importlib.reload(energy_cost.data)
    assert list(catalog.regions.items()) == [], catalog.regions.items()
assert len(Catalog.current().regions.items()) == 2
"""
    subprocess.run([sys.executable, "-c", script], check=True)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_contract_from_catalog_end_to_end() -> None:
    catalog = Catalog(indexes={"Belpex15min": _constant_index(50.0)}, regions=data.registry)
    consumption = Meter(measurements=TimeseriesFrame({"timestamp": _TIMESTAMPS, "value": 0.0002}))

    with catalog.active():
        Supplier.register("acme", Supplier(products={"dynamic": _tariff(10.0, index="Belpex15min")}))
        contract = Contract.model_validate({**_REGION, "supplier_key": "acme", "product_key": "dynamic"})
        costs = contract.apply(consumption)

    supplier_total = costs[(TariffCategory.SUPPLIER, "total", "total")].sum()
    assert supplier_total == pytest.approx(len(_TIMESTAMPS) * 0.0002 * 60.0)

    # The same contract through the default catalog gives the same total.
    Index.register("Belpex15min", _constant_index(50.0))
    reference = Contract.model_validate({**_REGION, "supplier": _tariff(10.0, index="Belpex15min")})
    expected = reference.apply(consumption)
    assert costs[("total", "total", "total")].sum() == pytest.approx(expected[("total", "total", "total")].sum())
