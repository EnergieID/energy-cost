"""Shared enums for domain-safe config and model values."""

from __future__ import annotations

from enum import StrEnum


class Carrier(StrEnum):
    ELECTRICITY = "electricity"
    GAS = "gas"


class Direction(StrEnum):
    OFFTAKE = "offtake"
    INJECTION = "injection"


class Region(StrEnum):
    BE_VLG = "BE-VLG"


class CustomerType(StrEnum):
    RESIDENTIAL = "residential"
    PROFESSIONAL = "professional"


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


class EnergyUnit(StrEnum):
    KWH = "kWh"


class PriceUnit(StrEnum):
    EUR_PER_MWH = "EUR/MWh"
    EUR_PER_KWH = "EUR/kWh"


class Market(StrEnum):
    EPEX_DA_BE_15MIN = "EPEX_DA_BE_15MIN"
    CUSTOM_MONTHLY = "CUSTOM_MONTHLY"
    CUSTOM_OTHER = "CUSTOM_OTHER"


class ContractType(StrEnum):
    DYNAMIC = "dynamic"
    VARIABLE = "variable"
    FIXED = "fixed"


class TimeOfUse(StrEnum):
    SINGLE = "single"
    DAY_NIGHT = "day_night"


class Channel(StrEnum):
    KWH = "kwh"
    KWH_DAY = "kwh_day"
    KWH_NIGHT = "kwh_night"


class PricingKind(StrEnum):
    FIXED_PER_KWH = "fixed_per_kwh"
    INDEXED_LINEAR = "indexed_linear"
    FIXED_PER_YEAR = "fixed_per_year"
    FIXED_PER_MONTH = "fixed_per_month"


class TariffComponentType(StrEnum):
    ENERGY = "energy"
    SURCHARGE_GREEN = "surcharge_green"
    SURCHARGE_WKK = "surcharge_wkk"
    FIXED_FEE = "fixed_fee"
    INJECTION_REMUNERATION = "injection_remuneration"
    OTHER = "other"


class GridComponentType(StrEnum):
    DISTRIBUTION_VARIABLE = "distribution_variable"
    DISTRIBUTION_FIXED = "distribution_fixed"
    DATABEHEER = "databeheer"
    PUBLIC_SERVICE = "public_service"
    SYSTEM_MANAGEMENT = "system_management"
    CAPACITY_TARIFF = "capacity_tariff"
    OTHER = "other"


class GridUnit(StrEnum):
    EUR_PER_KWH = "EUR/kWh"
    EUR_PER_YEAR = "EUR/year"
    EUR_PER_KW_MONTH = "EUR/kW/month"
    EUR_PER_KW_YEAR = "EUR/kW/year"
    EUR_PER_MONTH = "EUR/month"


class TaxKind(StrEnum):
    VARIABLE = "variable"
    FIXED_MONTHLY = "fixed_monthly"
    FIXED_YEARLY = "fixed_yearly"
    TIERED = "tiered"


class TaxUnit(StrEnum):
    EUR_PER_KWH = "EUR/kWh"
    EUR_PER_MONTH = "EUR/month"
    EUR_PER_YEAR = "EUR/year"
