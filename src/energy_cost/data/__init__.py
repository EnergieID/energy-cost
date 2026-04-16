from .be.flanders import data as flanders_data
from .models import ConnectionType, CustomerType, RegionalData

regionalData: dict[str, dict[ConnectionType, RegionalData]] = {"be_flanders": flanders_data}

__all__ = ["RegionalData", "CustomerType", "ConnectionType", "regionalData"]
