"""Connection and customer metadata models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel

from energy_cost.models._enums import CustomerType


class Region(StrEnum):
    BE_VLG = "BE-VLG"


class MeterRegister(StrEnum):
    SINGLE = "single"
    DUAL_DAY_NIGHT = "dual_day_night"


class MeterType(StrEnum):
    DIGITAL = "digital"
    ANALOG = "analog"


class VoltageLevel(StrEnum):
    LV = "LV"
    MV_1_26KV = "MV_1_26kV"
    MV_26_36KV = "MV_26_36kV"


class GasTariffClass(StrEnum):
    T1 = "T1"
    T2 = "T2"
    T3 = "T3"
    T4 = "T4"
    T5 = "T5"
    T6 = "T6"


class ElectricityConnection(BaseModel):
    """Electricity-specific connection details."""

    meter_register: MeterRegister = MeterRegister.SINGLE
    meter_type: MeterType = MeterType.DIGITAL
    voltage_level: VoltageLevel = VoltageLevel.LV
    has_separate_injection_point: bool = False


class GasConnection(BaseModel):
    """Gas-specific connection details."""

    gas_tariff_class: GasTariffClass | None = None
    telemetered: bool | None = None


class ConnectionInfo(BaseModel):
    """Connection and customer metadata."""

    region: Region
    dso: str  # e.g. "Fluvius Antwerpen"
    customer_type: CustomerType
    vat_rate: float | None = None  # inferred from customer_type + region rules if None

    electricity: ElectricityConnection = ElectricityConnection()
    gas: GasConnection = GasConnection()

    def get_vat_rate(self) -> float:
        """Return explicit VAT rate or infer from region/customer_type."""
        if self.vat_rate is not None:
            return self.vat_rate
        # Belgium standard VAT rate for energy
        return 0.21
