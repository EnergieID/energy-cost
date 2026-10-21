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
    WithJsonSchema({"type": "string", "format": "timezone", "examples": ["UTC", "Europe/Brussels"]}),
]
"""An IANA timezone name, parsed into a ``tzinfo``."""


# Names that refer to entries of the active Catalog. The custom JSON schema ``format`` lets
# consumers (e.g. form builders) offer the available names instead of a free text field.

IndexName = Annotated[str, WithJsonSchema({"type": "string", "format": "index-name"})]
"""Name of an index registered in the active catalog."""

RegionKey = Annotated[str, WithJsonSchema({"type": "string", "format": "region-key"})]
"""Name of a region in the active catalog."""

DistributorKey = Annotated[str, WithJsonSchema({"type": "string", "format": "distributor-key"})]
"""Name of a distributor within a region of the active catalog."""

SupplierKey = Annotated[str, WithJsonSchema({"type": "string", "format": "supplier-key"})]
"""Name of a supplier registered in the active catalog."""

ProductKey = Annotated[str, WithJsonSchema({"type": "string", "format": "product-key"})]
"""Name of a product offered by a supplier in the active catalog."""
