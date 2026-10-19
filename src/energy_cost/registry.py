from collections.abc import ItemsView
from typing import Generic, TypeVar

K = TypeVar("K")
V = TypeVar("V")


class RegistryMixin(Generic[K, V]):
    """Class methods that act on the dict returned by ``_registry``."""

    @classmethod
    def _registry(cls) -> dict[K, V]:
        raise NotImplementedError

    @classmethod
    def register(cls, key: K, value: V) -> None:
        """Store *value* under *key*, replacing any existing entry."""
        cls._registry()[key] = value

    @classmethod
    def get(cls, key: K) -> V:
        """Return the value stored under *key*, raising ``KeyError`` if absent."""
        return cls._registry()[key]

    @classmethod
    def clear(cls) -> None:
        """Remove all registered entries."""
        cls._registry().clear()

    @classmethod
    def items(cls) -> ItemsView[K, V]:
        return cls._registry().items()
