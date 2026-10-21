import datetime as dt
from datetime import UTC
from pathlib import Path
from typing import Any, Self
from zoneinfo import ZoneInfo

import pandas as pd
import yaml
from isodate import Duration
from pydantic import ConfigDict, Field, ModelWrapValidatorHandler, model_validator

from energy_cost.capacity import CapacityRule
from energy_cost.model import ADVANCED

from .data.models import ConnectionType, CustomerType, RegionalData, Supplier
from .meter import Meter, TariffCategory
from .resolution import Resolution, align_datetime_to_tz
from .tariff import Tariff
from .tax import Tax
from .types import DistributorKey, ProductKey, RegionKey, SupplierKey, TzInfo
from .versioning import Versioned, VersionedCollection, sum_frames


class Contract(Versioned):
    """A connection's supplier, distributor, fees and taxes, combined into one bill."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    start: dt.datetime = dt.datetime(1970, 1, 1, tzinfo=UTC)
    """Moment from which this contract applies."""

    # Reference keys (optional) — resolved into live objects by the validator
    region: RegionKey | None = None
    """Region whose fees, taxes and distributors apply."""
    connection_type: ConnectionType | None = None
    """Energy carrier of the connection."""
    customer_type: CustomerType | None = None
    """Customer type, used to pick the region's fees."""
    distributor_key: DistributorKey | None = None
    """Distributor within the region."""
    supplier_key: SupplierKey | None = None
    """Energy supplier."""
    product_key: ProductKey | None = None
    """Product of the supplier."""

    # Live objects — populated directly or resolved from reference keys
    supplier: Tariff | list[Tariff] | None = Field(default=None, json_schema_extra=ADVANCED)
    """Supplier tariff; overrides the one selected by supplier and product."""
    distributor: Tariff | list[Tariff] | None = Field(default=None, json_schema_extra=ADVANCED)
    """Distributor tariff; overrides the one selected by region and distributor."""
    fees: Tariff | list[Tariff] | None = Field(default=None, json_schema_extra=ADVANCED)
    """Fees; override the ones selected by region and customer type."""
    taxes: Tax | list[Tax] | None = Field(default=None, json_schema_extra=ADVANCED)
    """Taxes; override the ones of the region."""
    capacity_rule: CapacityRule | None = Field(default=None, json_schema_extra=ADVANCED)
    """How peak capacity is billed; overrides the rule of the region."""
    timezone: TzInfo = Field(default=ZoneInfo("UTC"), json_schema_extra=ADVANCED)
    """Timezone for all date handling; naive datetimes are read in this timezone."""

    @model_validator(mode="wrap")
    @classmethod
    def _resolve_references(cls, values: Any, handler: ModelWrapValidatorHandler[Self]) -> Self:
        if not isinstance(values, dict):
            return handler(values)

        resolved = cls._resolve(dict(values))
        contract = handler(resolved)
        # Values looked up from the references are not set by the user, so `exclude_unset` leaves them out
        contract.__pydantic_fields_set__ -= {key for key, value in resolved.items() if values.get(key) is not value}
        return contract

    @staticmethod
    def _resolve(values: dict[str, Any]) -> dict[str, Any]:
        """Fill in the live objects selected by the reference keys."""
        # Resolve supplier from supplier registry
        supplier_key = values.get("supplier_key")
        product_key = values.get("product_key")
        if values.get("supplier") is None and supplier_key is not None and product_key is not None:
            values["supplier"] = Supplier.get(supplier_key).products[product_key]

        region = values.get("region")
        connection_type = values.get("connection_type")

        if region is not None and connection_type is not None:
            regional = RegionalData.get((region, ConnectionType(connection_type)))

            customer_type = values.get("customer_type")
            if values.get("fees") is None and customer_type is not None:
                values["fees"] = regional.fees[CustomerType(customer_type)]

            if values.get("taxes") is None:
                values["taxes"] = regional.taxes

            if values.get("timezone") is None:
                values["timezone"] = regional.timezone

            distributor_key = values.get("distributor_key")
            if values.get("distributor") is None and distributor_key is not None:
                values["distributor"] = regional.distributors[distributor_key]

            if values.get("capacity_rule") is None and regional.capacity_rule is not None:
                values["capacity_rule"] = regional.capacity_rule

        return values

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Load a single contract from a YAML file."""
        with Path(path).open(encoding="utf-8") as file:
            raw_data = yaml.safe_load(file)
        return cls.model_validate(raw_data)

    def apply(
        self,
        consumption: Meter,
        injection: Meter | None = None,
        start: dt.datetime | None = None,
        end: dt.datetime | None = None,
        output_resolution: Resolution | None = None,
        binning_anchor: dt.datetime | None = None,
    ) -> pd.DataFrame:
        """Calculate the full energy bill."""

        # Normalize start/end to the contract timezone up front
        if start is not None:
            start = align_datetime_to_tz(start, self.timezone)
        if end is not None:
            end = align_datetime_to_tz(end, self.timezone)
        binning_anchor = align_datetime_to_tz(binning_anchor, self.timezone) if binning_anchor is not None else start
        if output_resolution is None:
            output_resolution = Duration(months=1)

        consumption = consumption.align_to_timezone(self.timezone)
        if injection is not None:
            injection = injection.align_to_timezone(self.timezone)
        if self.capacity_rule is not None:
            consumption = self.capacity_rule.apply(consumption)

        tariffs: dict[TariffCategory, Tariff | list[Tariff]] = {}
        for category in [TariffCategory.SUPPLIER, TariffCategory.DISTRIBUTOR, TariffCategory.FEES]:
            tariff = getattr(self, category.value)
            if tariff is not None:
                tariffs[category] = tariff

        frames = []
        for category, tariff_or_list in tariffs.items():
            tariff_list = tariff_or_list if isinstance(tariff_or_list, list) else [tariff_or_list]
            category_frames = []
            for tariff in tariff_list:
                optional_frame = tariff.apply(
                    consumption=consumption,
                    injection=injection,
                    start=start,
                    end=end,
                    output_resolution=output_resolution,
                    timezone=self.timezone,
                    binning_anchor=binning_anchor,
                )
                if optional_frame is not None:
                    category_frames.append(optional_frame)
            category_frame = sum_frames(category_frames) if category_frames else None
            if category_frame is not None:
                category_frame = category_frame.set_index("timestamp")
                category_frame.columns = pd.MultiIndex.from_tuples(
                    [(category,) + col for col in category_frame.columns]
                )
                frames.append(category_frame)

        result = pd.concat(frames, axis=1, sort=True)

        result = result.reset_index()
        _total = ("total", "total", "total")

        total_cols = [c for c in result.columns if c[-2:] == ("total", "total")]
        result[_total] = result[total_cols].sum(axis=1, skipna=False)

        if self.taxes is not None:
            tax_list = self.taxes if isinstance(self.taxes, list) else [self.taxes]
            tax_frames = []
            for tax in tax_list:
                frame = tax.apply(result, start, end, output_resolution, timezone=self.timezone)
                if frame is not None and not frame.empty:
                    frame = frame.set_index("timestamp")
                    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
                    tax_frames.append(frame.reset_index())
            if tax_frames:
                tax_frame = sum_frames(tax_frames).set_index("timestamp")
                result = result.set_index("timestamp").join(tax_frame).reset_index()

        # Recompute grand total including taxes
        total_cols = [c for c in result.columns if c[-2:] == ("total", "total") and c != _total]
        result[_total] = result[total_cols].sum(axis=1, skipna=False)

        return result


class ContractHistory(VersionedCollection[Contract]):
    """Contracts over time, as a list of versions."""

    def apply(
        self,
        consumption: Meter,
        injection: Meter | None = None,
        start: dt.datetime | None = None,
        end: dt.datetime | None = None,
        output_resolution: Resolution | None = None,
    ) -> pd.DataFrame | None:
        if start is None:
            start = consumption.measurements.start
        if end is None:
            end = consumption.measurements.end
        if output_resolution is None:
            output_resolution = Duration(months=1)

        return self.collect_version_frames(
            lambda contract, seg_start, seg_end: contract.apply(
                consumption,
                injection,
                start=seg_start,
                end=seg_end,
                output_resolution=output_resolution,
                binning_anchor=start,
            ),
            start,
            end,
        )
