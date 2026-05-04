from .be.flanders import register as _register_flanders
from .models import ConnectionType, CustomerType, RegionalData, Supplier

_register_flanders()

__all__ = ["RegionalData", "CustomerType", "ConnectionType", "Supplier"]
