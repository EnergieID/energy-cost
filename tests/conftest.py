from __future__ import annotations

import pytest

from energy_cost.data.models import Supplier
from energy_cost.index import Index


@pytest.fixture(autouse=True)
def clear_registered_indexes() -> None:
    """Isolate tests that use the global `Index` registry."""
    Index.clear()


@pytest.fixture(autouse=True)
def clear_registered_suppliers() -> None:
    """Isolate tests that use the global `Supplier` registry."""
    Supplier.clear()
