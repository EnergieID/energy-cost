import datetime as dt

import pandas as pd
from pydantic import BaseModel

from .meter import Meter
from .resolution import Resolution
from .tariff import Tariff


class Contract(BaseModel):
    provider: Tariff = Tariff(versions=[])
    distributor: Tariff = Tariff(versions=[])
    fees: Tariff = Tariff(versions=[])
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
        for role in ["provider", "distributor", "fees"]:
            tariff = getattr(self, role)
            optional_frame = tariff.apply(
                meters=meters,
                start=start,
                end=end,
                resolution=resolution,
            )
            if optional_frame is not None:
                optional_frame = optional_frame.set_index("timestamp")
                optional_frame.columns = pd.MultiIndex.from_tuples([(role,) + col for col in optional_frame.columns])
                frames.append(optional_frame)

        result = pd.concat(frames, axis=1)

        # Taxes apply to provider + distributor totals only.
        provider_total = result[("provider", "total", "total")]
        distributor_total = result[("distributor", "total", "total")]
        result[("taxes", "total", "total")] = (provider_total + distributor_total) * self.tax_rate

        # Final total includes fees and taxes.
        total_cols = [c for c in result.columns if c[-2:] == ("total", "total")]
        result[("total", "total", "total")] = result[total_cols].sum(axis=1)

        return result.reset_index()
