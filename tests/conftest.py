from __future__ import annotations

import pytest

from energy_cost.index import Index


@pytest.fixture(autouse=True)
def clear_registered_indexes() -> None:
    """Isolate tests that use the global `Index.indexes` registry."""
    Index.indexes.clear()
