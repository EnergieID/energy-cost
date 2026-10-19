from . import electricity, gas

registry = {**electricity.registry, **gas.registry}

__all__ = ["electricity", "gas", "registry"]
