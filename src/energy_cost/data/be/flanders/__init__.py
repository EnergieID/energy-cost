from energy_cost.data.models import ConnectionType

from .electricity import data as flanders_electricity_data

data = {ConnectionType.ELECTRICITY: flanders_electricity_data}
