import datetime as dt
from datetime import UTC
from typing import Any

import pandas as pd
from isodate import Duration
from pydantic import ConfigDict, model_validator

from .data.models import ConnectionType, CustomerType, RegionalData
from .meter import CostGroup, Meter, TariffCategory
from .resolution import Resolution, align_datetime_to_tz
from .tariff import Tariff
from .tax import Tax
from .versioning import Versioned, VersionedCollection


class Contract(Versioned):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    start: dt.datetime = dt.datetime(1970, 1, 1, tzinfo=UTC)

    # Reference keys (optional) — resolved into live objects by the validator
    region: str | None = None
    connection_type: ConnectionType | None = None
    customer_type: CustomerType | None = None
    distributor_key: str | None = None

    # Live objects — populated directly or resolved from reference keys
    supplier: Tariff | list[Tariff] | None = None
    distributor: Tariff | list[Tariff] | None = None
    fees: Tariff | list[Tariff] | None = None
    taxes: Tax | list[Tax] | None = None
    timezone: dt.tzinfo = UTC
    """All datetime operations use this timezone. Naive datetimes are treated as being
    in this timezone; tz-aware datetimes are converted to it."""

    @model_validator(mode="before")
    @classmethod
    def _resolve_references(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values

        region = values.get("region")
        connection_type = values.get("connection_type")

        if region is not None and connection_type is not None:
            regional = RegionalData.get(region, ConnectionType(connection_type))

            customer_type = values.get("customer_type")
            if values.get("fees") is None and customer_type is not None:
                values["fees"] = regional.fees[CustomerType(customer_type)]

            if values.get("taxes") is None:
                values["taxes"] = regional.taxes

            if "timezone" not in values:
                values["timezone"] = regional.timezone

            distributor_key = values.get("distributor_key")
            if values.get("distributor") is None and distributor_key is not None:
                values["distributor"] = regional.distributors[distributor_key]

        return values

    def apply(
        self,
        meters: list[Meter],
        start: dt.datetime | None = None,
        end: dt.datetime | None = None,
        resolution: Resolution | None = None,
    ) -> pd.DataFrame:
        """Calculate the full energy bill."""

        # Normalize start/end to the contract timezone up front
        if start is not None:
            start = align_datetime_to_tz(start, self.timezone)
        if end is not None:
            end = align_datetime_to_tz(end, self.timezone)
        if resolution is None:
            resolution = Duration(months=1)

        tariffs: dict[TariffCategory, Tariff | list[Tariff]] = {}
        for category in [TariffCategory.SUPPLIER, TariffCategory.DISTRIBUTOR, TariffCategory.FEES]:
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
                    timezone=self.timezone,
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

        result = pd.concat(frames, axis=1, sort=True)

        result = result.reset_index()
        _total = (TariffCategory.TOTAL, CostGroup.TOTAL, "total")

        total_cols = [c for c in result.columns if c[-2:] == (CostGroup.TOTAL, "total")]
        result[_total] = result[total_cols].sum(axis=1)

        if self.taxes is not None:
            tax_list = self.taxes if isinstance(self.taxes, list) else [self.taxes]
            tax_frame: pd.DataFrame | None = None
            for tax in tax_list:
                frame = tax.apply(result, start, end, resolution, timezone=self.timezone)
                if frame is not None and not frame.empty:
                    frame = frame.set_index("timestamp")
                    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
                    tax_frame = frame if tax_frame is None else tax_frame.add(frame, fill_value=0)
            if tax_frame is not None:
                result = result.set_index("timestamp").join(tax_frame).reset_index()

        # Recompute grand total including taxes
        total_cols = [c for c in result.columns if c[-2:] == (CostGroup.TOTAL, "total") and c != _total]
        result[_total] = result[total_cols].sum(axis=1)

        return result


class ContractHistory(VersionedCollection[Contract]):
    def apply(
        self,
        meters: list[Meter],
        start: dt.datetime | None = None,
        end: dt.datetime | None = None,
        resolution: Resolution | None = None,
    ) -> pd.DataFrame | None:
        if start is None or end is None:
            from .resolution import detect_resolution_and_range

            combined = pd.concat([m.data for m in meters], ignore_index=True)
            detected_start, detected_end, _ = detect_resolution_and_range(combined)
            if start is None:
                start = detected_start
            if end is None:
                end = detected_end

        return self.collect_version_frames(
            lambda contract, seg_start, seg_end: contract.apply(
                meters, start=seg_start, end=seg_end, resolution=resolution
            ),
            start,
            end,
        )
