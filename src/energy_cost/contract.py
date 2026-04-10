import datetime as dt
from enum import StrEnum

import pandas as pd
from pydantic import BaseModel

from .meter import CostGroup, Meter, MeterType
from .resolution import Resolution
from .tariff import Tariff


class TariffCategory(StrEnum):
    PROVIDER = "provider"
    DISTRIBUTOR = "distributor"
    FEES = "fees"
    TAXES = "taxes"
    TOTAL = "total"


class Contract(BaseModel):
    provider: Tariff | list[Tariff] | None = None
    distributor: Tariff | list[Tariff] | None = None
    fees: Tariff | list[Tariff] | None = None
    tax_rate: float = 0.0

    def calculate_cost(
        self,
        meters: list[Meter],
        start: dt.datetime | None = None,
        end: dt.datetime | None = None,
        resolution: Resolution | None = None,
        expand_meter_types: bool = False,
    ) -> pd.DataFrame:
        """Calculate the full energy bill."""

        tariffs: dict[TariffCategory, Tariff | list[Tariff]] = {}
        for category in [TariffCategory.PROVIDER, TariffCategory.DISTRIBUTOR, TariffCategory.FEES]:
            tariff = getattr(self, category.value)
            if tariff is not None:
                tariffs[category] = tariff

        frames = []
        for category, tariff_or_list in tariffs.items():
            tariff_list = tariff_or_list if isinstance(tariff_or_list, list) else [tariff_or_list]
            category_frame: pd.DataFrame | None = None
            for tariff in tariff_list:
                optional_frame = tariff.apply(
                    meters=meters,
                    start=start,
                    end=end,
                    resolution=resolution,
                )
                if optional_frame is not None:
                    optional_frame = optional_frame.set_index("timestamp")
                    if category_frame is None:
                        category_frame = optional_frame
                    else:
                        category_frame = category_frame.add(optional_frame, fill_value=0)
            if category_frame is not None:
                category_frame.columns = pd.MultiIndex.from_tuples(
                    [(category,) + col for col in category_frame.columns]
                )
                frames.append(category_frame)

        result = pd.concat(frames, axis=1)
        _total = (TariffCategory.TOTAL, CostGroup.TOTAL, MeterType.ALL, "total")
        _taxes = (TariffCategory.TAXES, CostGroup.TOTAL, MeterType.ALL, "total")
        total_cols = [c for c in result.columns if c[-3:] == (CostGroup.TOTAL, MeterType.ALL, "total")]
        result[_taxes] = result[total_cols].sum(axis=1) * self.tax_rate
        total_cols += [_taxes]
        result[_total] = result[total_cols].sum(axis=1)

        result = result.reset_index()

        if not expand_meter_types:
            result = _collapse_meter_type(result)

        return result


def _collapse_meter_type(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse 4-level contract columns (TariffCategory, CostGroup, MeterType, cost_type)
    to 3-level (TariffCategory, CostGroup, cost_type) by summing across MeterType."""
    data = df.set_index("timestamp")
    collapsed = data.T.groupby(level=[0, 1, 3]).sum().T
    collapsed.columns = pd.MultiIndex.from_tuples(collapsed.columns)
    return collapsed.reset_index()
