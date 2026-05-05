from .electricity import register as _register_electricity
from .gas import register as _register_gas


def register() -> None:
    _register_electricity()
    _register_gas()
