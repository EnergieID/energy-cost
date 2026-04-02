import datetime as dt

import pandas as pd
from pydantic import BaseModel

from .resolution import Resolution
from .tariff import Tariff
from .tariff_version import MeterType


class Contract(BaseModel):
    provider: Tariff = Tariff(versions=[])
    distributor: Tariff = Tariff(versions=[])
    fees: Tariff = Tariff(versions=[])
    tax_rate: float = 0.0

    def calculate_cost(
        self,
        consumption: pd.DataFrame,
        injection: pd.DataFrame | None = None,
        start: dt.datetime | None = None,
        end: dt.datetime | None = None,
        resolution: Resolution | None = None,
        meter_type: MeterType = MeterType.SINGLE_RATE,
    ) -> pd.DataFrame:
        """Calculate the full energy bill.
        Parameters
        ----------
        consumption:
            DataFrame with ``timestamp`` and ``value`` columns (quantity per interval, e.g. MWh).
            May extend beyond ``start``/``end`` to supply capacity-cost history.
        injection:
            Optional injection quantities in the same format as ``consumption``.
        start:
            Billing period start (inclusive).  Defaults to the earliest timestamp in ``consumption``.
        end:
            Billing period end (exclusive).  Defaults to one data-resolution step after the last
            timestamp in ``consumption``.
        resolution:
            Output resolution.  Defaults to P1M (calendar-monthly).
        meter_type:
            Meter type used when looking up consumption/injection formulas.
        """
        kwargs: dict = dict(
            consumption=consumption,
            injection=injection,
            start=start,
            end=end,
            resolution=resolution,
            meter_type=meter_type,
        )

        frames = []
        for role in ["provider", "distributor", "fees"]:
            tariff = getattr(self, role)
            optional_frame = tariff.apply(**kwargs)
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
