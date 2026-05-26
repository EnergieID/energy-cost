from pathlib import Path
from zoneinfo import ZoneInfo

import isodate

from energy_cost.capacity import CapacityRule
from energy_cost.data.models import ConnectionType, CustomerType, RegionalData
from energy_cost.tariff import Tariff
from energy_cost.tax import Tax

_DIR = Path(__file__).parent
_BELGIAN_DIR = _DIR.parent.parent / "electricity"
_FLEMISH_FEES_DIR = _DIR / "fees"
_BELGIAN_FEES_DIR = _BELGIAN_DIR / "fees"
_DISTRIBUTORS_DIR = _DIR / "distributors"
_TIMEZONE = ZoneInfo("Europe/Brussels")

data = RegionalData(
    fees={
        customer_type: [
            Tariff.from_yaml(_FLEMISH_FEES_DIR / f"{customer_type.value}.yml"),
            Tariff.from_yaml(_BELGIAN_FEES_DIR / f"{customer_type.value}.yml"),
        ]
        for customer_type in CustomerType
    },
    distributors={path.stem: Tariff.from_yaml(path) for path in sorted(_DISTRIBUTORS_DIR.glob("*.yml"))},
    taxes=Tax.from_yaml(_BELGIAN_DIR / "taxes.yml"),
    capacity_rule=CapacityRule(
        measurement_period=isodate.parse_duration("PT15M"),
        billing_period=isodate.parse_duration("P1M"),
        window_periods=12,
    ),
    timezone=_TIMEZONE,
)


def register() -> None:
    RegionalData.register(("be_flanders", ConnectionType.ELECTRICITY), data)
