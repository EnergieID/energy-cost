from collections.abc import ItemsView
from typing import ClassVar


class RegistryMixin[K, V]:
    _registry: ClassVar[dict] = {}

    def __init_subclass__(cls, **kwargs: object) -> None:
        super().__init_subclass__(**kwargs)
        cls._registry = {}

    @classmethod
    def register(cls, key: K, value: V) -> None:
        """Store *value* under *key*, replacing any existing entry."""
        cls._registry[key] = value

    @classmethod
    def get(cls, key: K) -> V:
        """Return the value stored under *key*, raising ``KeyError`` if absent."""
        return cls._registry[key]

    @classmethod
    def clear(cls) -> None:
        """Remove all registered entries."""
        cls._registry.clear()

    @classmethod
    def items(cls) -> ItemsView[K, V]:
        return cls._registry.items()
