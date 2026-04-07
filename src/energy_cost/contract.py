import datetime as dt

import pandas as pd
from pydantic import BaseModel

from .meter import Meter
from .resolution import Resolution
from .tariff import Tariff


class Contract(BaseModel):
    tariffs: dict[str, Tariff] = {}
    tax_rate: float = 0.0

    def calculate_cost(
        self,
        meters: list[Meter],
        start: dt.datetime | None = None,
        end: dt.datetime | None = None,
        resolution: Resolution | None = None,
    ) -> pd.DataFrame:
        """Calculate the full energy bill."""

        frames = []
        for name, tariff in self.tariffs.items():
            optional_frame = tariff.apply(
                meters=meters,
                start=start,
                end=end,
                resolution=resolution,
            )
            if optional_frame is not None:
                optional_frame = optional_frame.set_index("timestamp")
                optional_frame.columns = pd.MultiIndex.from_tuples([(name,) + col for col in optional_frame.columns])
                frames.append(optional_frame)

        result = pd.concat(frames, axis=1)
        total_cols = [c for c in result.columns if c[-2:] == ("total", "total")]
        result[("taxes", "total", "total")] = result[total_cols].sum(axis=1) * self.tax_rate
        total_cols += [("taxes", "total", "total")]
        result[("total", "total", "total")] = result[total_cols].sum(axis=1)

        return result.reset_index()
