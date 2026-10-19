from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Self

if TYPE_CHECKING:
    from .data.models import ConnectionType, RegionalData, Supplier
    from .index.index import Index


class Catalog:
    """The indexes, suppliers and regional data that names in tariffs and contracts resolve to."""

    def __init__(
        self,
        indexes: Mapping[str, Index] | None = None,
        suppliers: Mapping[str, Supplier] | None = None,
        regions: Mapping[tuple[str, ConnectionType], RegionalData] | None = None,
    ) -> None:
        self.indexes: dict[str, Index] = dict(indexes or {})
        self.suppliers: dict[str, Supplier] = dict(suppliers or {})
        self.regions: dict[tuple[str, ConnectionType], RegionalData] = dict(regions or {})

    @contextmanager
    def active(self) -> Generator[Self]:
        """Make this catalog the current one for the block.

        Carries over into asyncio tasks and ``asyncio.to_thread``, not into plain threads or thread pools.
        """
        token = _active.set(self)
        try:
            yield self
        finally:
            _active.reset(token)

    @staticmethod
    def current() -> Catalog:
        """The active catalog, else the default one."""
        catalog = _active.get()
        if catalog is not None:
            return catalog
        _fill_default()
        return _default


_active: ContextVar[Catalog | None] = ContextVar("energy_cost_active_catalog", default=None)
_default = Catalog()
_default_filled = False


def _fill_default() -> None:
    """Add the bundled regional data to the default catalog, once."""
    global _default_filled
    if _default_filled:
        return
    from .data import registry

    _default.regions.update(registry)
    _default_filled = True
