from __future__ import annotations

import pytest

from energy_cost.registry import RegistryMixin


def test_registry_mixin_requires_registry() -> None:
    class Unbound(RegistryMixin[str, int]):
        pass

    with pytest.raises(NotImplementedError):
        Unbound.get("a")
