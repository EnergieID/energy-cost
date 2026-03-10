"""Output model for cost calculations."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class CostResult(BaseModel):
    """Output of a cost calculation: itemized breakdown + totals."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    breakdown: Any
    totals: dict[str, float]
    assumptions: dict
