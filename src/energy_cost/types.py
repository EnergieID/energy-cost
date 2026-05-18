"""Pydantic-compatible type annotations for abstract/third-party types."""

from __future__ import annotations

import datetime as dt
from typing import Annotated
from zoneinfo import ZoneInfo

from pydantic import PlainSerializer, PlainValidator, WithJsonSchema


def _validate_tzinfo(v: object) -> dt.tzinfo:
    if isinstance(v, dt.tzinfo):
        return v
    if isinstance(v, str):
        return ZoneInfo(v)
    raise ValueError(f"Expected timezone string or tzinfo, got {type(v).__name__!r}")


def _serialize_tzinfo(v: dt.tzinfo) -> str:
    return str(v.key) if hasattr(v, "key") else str(v)  # type: ignore[attr-defined]


TzInfo = Annotated[
    dt.tzinfo,
    PlainValidator(_validate_tzinfo),
    PlainSerializer(_serialize_tzinfo, return_type=str),
    WithJsonSchema({"type": "string", "examples": ["UTC", "Europe/Brussels"]}),
]
