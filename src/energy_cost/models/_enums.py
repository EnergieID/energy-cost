"""Shared domain enums used across multiple model classes."""

from __future__ import annotations

from enum import StrEnum


class Carrier(StrEnum):
    ELECTRICITY = "electricity"
    GAS = "gas"


class Direction(StrEnum):
    OFFTAKE = "offtake"
    INJECTION = "injection"


class CustomerType(StrEnum):
    RESIDENTIAL = "residential"
    PROFESSIONAL = "professional"
