"""
An energy billing is defined based on 4 parts:
- the provider's tariff
- the distributor's tariff
- government fees (also a tariff, but taxes don't apply to this tariff as they are assumed to be included in the fee)
- taxes

To then calculate the bill for a given time range, we'll need:
The users injection and consumption data for the given time range
We need to know the meter type (single rate, multi rate, etc.)

For flemish capacity cost we need one year of data before the start of the billing period
So we should support the dataframe extending beyond the requested start and end timestamps, but only use the relevant part for the cost calculation.

The response should be a dataframe with the following grouped columns:
timestamp | <provider_name>                  | <distributor_name>               | fees             | taxes       | total_cost
timestamp | consumption cost | injection cost | capacity cost | fixed cost | total | ...same... | consumption fees | injection fees | capacity fees | fixed fees | total | total taxes | total cost
timestamp | energy | renewable | energy | renewable | capacity cost | fixed cost | total | ... | total taxes | total cost

A tariff should already have a method that given injection and consumption data, and the meter type, a start and end date returns a dataframe like this:
timestamp | consumption cost   | injection cost     | capacity cost | fixed cost | total
timestamp | energy | renewable | energy | renewable | capacity cost | fixed cost | total

Which can then be reused 3 times for the provider, distributor and fees. The taxes can then be calculated based on the total cost of the provider and distributor. (taxes ignore fees as they are assumed to already include taxes)
"""

import datetime as dt

import pandas as pd
from pydantic import BaseModel

from .resolution import Resolution
from .tariff import Tariff
from .tariff_version import MeterType


class Billing(BaseModel):
    """Combines a provider tariff, distributor tariff, optional government fees and a tax rate.

    Taxes are computed on the provider + distributor totals only; fees are assumed to already
    include any applicable taxes.
    """

    provider: Tariff = Tariff(versions=[])
    distributor: Tariff = Tariff(versions=[])
    fees: Tariff = Tariff(versions=[])
    tax_rate: float = 0.0

    def calculate(
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
