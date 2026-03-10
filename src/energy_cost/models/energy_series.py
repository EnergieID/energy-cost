"""Energy consumption or injection time series."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import narwhals as nw
from pydantic import BaseModel, ConfigDict, field_validator

from energy_cost.models._enums import Carrier, Direction


class EnergyUnit(StrEnum):
    KWH = "kWh"


class EnergySeries(BaseModel):
    """Energy consumption or injection time series.

    Accepted input is a native eager dataframe with either:
    - a DatetimeIndex and one or more energy columns, or
    - explicit ``timestamp`` column and one or more energy columns.

    Internal normalized schema is a Narwhals dataframe that always includes
    ``timestamp`` and preserves provided energy columns.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    carrier: Carrier
    direction: Direction
    timezone: str = "Europe/Brussels"
    unit: EnergyUnit = EnergyUnit.KWH
    data: nw.DataFrame

    @field_validator("data", mode="before")
    @classmethod
    def validate_data(cls, v: Any) -> nw.DataFrame:
        if v is None:
            raise ValueError("Data cannot be None")

        df = nw.from_native(v, eager_only=True)
        if "timestamp" in df.columns:
            return df

        if not hasattr(v, "index"):
            raise ValueError("EnergySeries requires a DatetimeIndex or a 'timestamp' column")

        return nw.from_dict(
            {
                "timestamp": list(v.index),
                **{column: df[column].to_list() for column in df.columns},
            },
            backend=df.implementation,
        )
