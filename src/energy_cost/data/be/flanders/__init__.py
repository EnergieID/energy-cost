from energy_cost.data.models import ConnectionType

from .electricity import data as electricity_data
from .gas import data as gas_data

data = {
    ConnectionType.ELECTRICITY: electricity_data,
    ConnectionType.GAS: gas_data,
}
