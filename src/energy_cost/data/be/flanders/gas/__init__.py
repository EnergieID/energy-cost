from pathlib import Path

from energy_cost.data.models import CustomerType, RegionalData
from energy_cost.tariff import Tariff
from energy_cost.tax import Tax

_DIR = Path(__file__).parent
_BELGIAN_DIR = _DIR.parent.parent / "gas"
_BELGIAN_FEES_DIR = _BELGIAN_DIR / "fees"
_DISTRIBUTORS_DIR = _DIR / "distributors"

data = RegionalData(
    fees={
        customer_type: Tariff.from_yaml(_BELGIAN_FEES_DIR / f"{customer_type.value}.yml")
        for customer_type in CustomerType
    },
    distributors={path.stem: Tariff.from_yaml(path) for path in sorted(_DISTRIBUTORS_DIR.glob("*.yml"))},
    taxes=Tax.from_yaml(_BELGIAN_DIR / "taxes.yml"),
)
