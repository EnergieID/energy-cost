"""Cost calculation engine.

This module orchestrates the end-to-end computation of itemised energy costs
for a billing period.  The first implementation covers **dynamic electricity**
(15-min resolution, Flanders).
"""

from __future__ import annotations

from datetime import datetime

import narwhals as nw

from energy_cost.models import (
    Carrier,
    ConnectionInfo,
    ContractType,
    CostResult,
    EnergySeries,
    GridComponentType,
    GridTariffSet,
    GridUnit,
    MarketPriceSeries,
    MonthlyPeaks,
    PricingKind,
    TariffComponentType,
    TariffDefinition,
    TaxKind,
    TaxRule,
)

# Standardised output column names
COL_SUPPLIER_ENERGY = "supplier_energy_eur"
COL_SUPPLIER_FIXED = "supplier_fixed_eur"
COL_GRID_VARIABLE = "grid_variable_eur"
COL_GRID_FIXED = "grid_fixed_eur"
COL_CAPACITY = "capacity_eur"
COL_TAXES = "taxes_eur"
COL_VAT = "vat_eur"
COL_TOTAL = "total_eur"

MINUTES_PER_DAY = 24 * 60


def _validate_dynamic_inputs(
    energy: EnergySeries,
    market_prices: MarketPriceSeries,
    tariff: TariffDefinition,
) -> None:
    """Raise on invalid/mismatched inputs for a dynamic electricity calculation."""
    if energy.carrier != Carrier.ELECTRICITY:
        raise ValueError(f"Expected electricity carrier, got {energy.carrier}")
    if tariff.contract_type != ContractType.DYNAMIC:
        raise ValueError(f"Expected dynamic contract type, got {tariff.contract_type}")
    if tariff.carrier != energy.carrier:
        raise ValueError(f"Tariff carrier ({tariff.carrier}) does not match energy carrier ({energy.carrier})")
    if tariff.direction != energy.direction:
        raise ValueError(f"Tariff direction ({tariff.direction}) does not match energy direction ({energy.direction})")

    energy_df = energy.data
    if "kwh" not in energy_df.columns:
        raise ValueError("Dynamic electricity requires a 'kwh' column")
    if "timestamp" not in energy_df.columns:
        raise ValueError("Dynamic electricity requires a 'timestamp' column")

    timestamps = [_to_datetime(value) for value in energy_df["timestamp"].to_list()]
    _validate_timestamps(timestamps)
    expected_step_minutes = tariff.settlement_interval_minutes
    if _step_minutes(timestamps) != expected_step_minutes:
        raise ValueError(f"Dynamic electricity requires {expected_step_minutes}-min frequency")

    price_timestamps = [_to_datetime(value) for value in market_prices.data["timestamp"].to_list()]
    if timestamps != price_timestamps:
        raise ValueError("Market price index does not align with energy data index")


def _compute_supplier_energy(
    energy_kwh: nw.Series,
    market_prices_eur_per_mwh: nw.Series,
    tariff: TariffDefinition,
) -> nw.Series:
    """Compute supplier energy cost per timestep (EUR)."""
    total = energy_kwh * 0.0
    for component in tariff.components:
        if component.component_type != TariffComponentType.ENERGY:
            continue
        pricing = component.pricing
        if pricing.kind == PricingKind.INDEXED_LINEAR:
            coef = pricing.coef or 0.0
            adder = pricing.add_cents_per_kwh or 0.0
            total = total + (((market_prices_eur_per_mwh * coef) + adder) / 100.0) * energy_kwh
        elif pricing.kind == PricingKind.FIXED_PER_KWH:
            rate = pricing.eur_per_kwh or 0.0
            total = total + (energy_kwh * rate)
    return total


def _compute_supplier_fixed(
    work_df: nw.DataFrame,
    tariff: TariffDefinition,
    settlement_minutes: int,
) -> nw.Series:
    """Prorate supplier fixed fees over timesteps (EUR per timestep)."""
    total = work_df["kwh"] * 0.0
    for component in tariff.components:
        if component.component_type != TariffComponentType.FIXED_FEE:
            continue
        pricing = component.pricing
        if pricing.kind == PricingKind.FIXED_PER_YEAR:
            year = _to_datetime(work_df["timestamp"][0]).year
            days_in_year = _days_in_year(year)
            per_timestep = (pricing.eur_per_year or 0.0) * settlement_minutes / (days_in_year * MINUTES_PER_DAY)
            total = total + per_timestep
        elif pricing.kind == PricingKind.FIXED_PER_MONTH:
            total = total + (pricing.eur_per_month or 0.0) / work_df["n_in_month"]
    return total


def _compute_grid_variable(
    energy_kwh: nw.Series,
    grid_tariffs: GridTariffSet,
) -> nw.Series:
    """Compute variable grid charges per timestep (EUR)."""
    total = energy_kwh * 0.0
    for comp in grid_tariffs.components:
        if comp.unit == GridUnit.EUR_PER_KWH:
            total = total + (energy_kwh * comp.value)
    return total


def _compute_grid_fixed(
    work_df: nw.DataFrame,
    grid_tariffs: GridTariffSet,
    connection: ConnectionInfo,
    settlement_minutes: int,
) -> nw.Series:
    """Prorate fixed grid charges over timesteps (EUR per timestep)."""
    total = work_df["kwh"] * 0.0
    for comp in grid_tariffs.components:
        if not _condition_applies(comp.conditions, connection):
            continue

        if comp.unit == GridUnit.EUR_PER_YEAR:
            year = _to_datetime(work_df["timestamp"][0]).year
            days_in_year = _days_in_year(year)
            per_timestep = comp.value * settlement_minutes / (days_in_year * MINUTES_PER_DAY)
            total = total + per_timestep
        elif comp.unit == GridUnit.EUR_PER_MONTH:
            total = total + comp.value / work_df["n_in_month"]
    return total


def _compute_capacity(
    work_df: nw.DataFrame,
    monthly_peaks: MonthlyPeaks,
    grid_tariffs: GridTariffSet,
) -> nw.Series:
    """Allocate monthly capacity charges over timesteps (EUR per timestep)."""
    cap_rate = 0.0
    for comp in grid_tariffs.components:
        if comp.component_type == GridComponentType.CAPACITY_TARIFF and comp.unit == GridUnit.EUR_PER_KW_MONTH:
            cap_rate = comp.value
            break

    total = work_df["kwh"] * 0.0
    if cap_rate == 0.0:
        return total

    peaks_df = monthly_peaks.data
    month_peaks = work_df.select("month").join(peaks_df, on="month", how="left")["peak_kw"]
    if bool(month_peaks.is_null().any()):
        raise KeyError("No monthly peak found for one or more months")

    total = (month_peaks * cap_rate) / work_df["n_in_month"]

    return total


def _compute_taxes(
    work_df: nw.DataFrame,
    tax_rules: list[TaxRule],
    settlement_minutes: int,
) -> nw.Series:
    """Compute tax/levy charges per timestep (EUR).

    For tiered taxes in this POC, we use the lowest applicable tier rate
    (suitable for typical residential consumption < 20 MWh/year).
    Full tiered calculation based on cumulative annual volume is planned for v1.
    """
    energy_kwh = work_df["kwh"]
    total = energy_kwh * 0.0
    for rule in tax_rules:
        if rule.kind == TaxKind.VARIABLE:
            rate = rule.value or 0.0
            total = total + (energy_kwh * rate)
        elif rule.kind == TaxKind.TIERED:
            if rule.tiers:
                rate = rule.tiers[0].eur_per_kwh
                total = total + (energy_kwh * rate)
        elif rule.kind == TaxKind.FIXED_YEARLY:
            year = _to_datetime(work_df["timestamp"][0]).year
            days_in_year = _days_in_year(year)
            per_timestep = (rule.value or 0.0) * settlement_minutes / (days_in_year * MINUTES_PER_DAY)
            total = total + per_timestep
        elif rule.kind == TaxKind.FIXED_MONTHLY:
            total = total + (rule.value or 0.0) / work_df["n_in_month"]
    return total


def _compute_vat(
    breakdown: dict[str, nw.Series],
    vat_rate: float,
    tariff: TariffDefinition,
    grid_tariffs: GridTariffSet,
    tax_rules: list[TaxRule],
) -> nw.Series:
    """Compute VAT on all VAT-applicable components."""
    vat_base = breakdown[COL_SUPPLIER_ENERGY] * 0.0

    if any(c.vat_applicable for c in tariff.components if c.component_type == TariffComponentType.ENERGY):
        vat_base = vat_base + breakdown[COL_SUPPLIER_ENERGY]

    if any(c.vat_applicable for c in tariff.components if c.component_type == TariffComponentType.FIXED_FEE):
        vat_base = vat_base + breakdown[COL_SUPPLIER_FIXED]

    if any(c.vat_applicable for c in grid_tariffs.components if c.unit == GridUnit.EUR_PER_KWH):
        vat_base = vat_base + breakdown[COL_GRID_VARIABLE]

    if any(c.vat_applicable for c in grid_tariffs.components if c.unit in (GridUnit.EUR_PER_YEAR, GridUnit.EUR_PER_MONTH)):
        vat_base = vat_base + breakdown[COL_GRID_FIXED]

    if any(c.vat_applicable for c in grid_tariffs.components if c.component_type == GridComponentType.CAPACITY_TARIFF):
        vat_base = vat_base + breakdown[COL_CAPACITY]

    if any(r.vat_applicable for r in tax_rules):
        vat_base = vat_base + breakdown[COL_TAXES]

    return vat_base * vat_rate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        return value
    if hasattr(value, "to_pydatetime"):
        return value.to_pydatetime()  # type: ignore[no-any-return]
    year = getattr(value, "year", None)
    month = getattr(value, "month", None)
    if year is not None and month is not None:
        return datetime(int(year), int(month), 1)
    return datetime.fromisoformat(str(value))


def _validate_timestamps(timestamps: list[datetime]) -> None:
    if not timestamps:
        raise ValueError("Energy data cannot be empty")
    if any(timestamps[i] >= timestamps[i + 1] for i in range(len(timestamps) - 1)):
        raise ValueError("Index must be monotonically increasing and unique")


def _step_minutes(timestamps: list[datetime]) -> int:
    if len(timestamps) < 2:
        return 0
    deltas = [
        int((timestamps[i + 1] - timestamps[i]).total_seconds() // 60)
        for i in range(len(timestamps) - 1)
    ]
    if any(delta != deltas[0] for delta in deltas[1:]):
        raise ValueError("Dynamic electricity requires a constant frequency")
    return deltas[0]


def _days_in_year(year: int) -> int:
    """Return the number of days in a given year."""
    return 366 if _is_leap_year(year) else 365


def _is_leap_year(year: int) -> bool:
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def _condition_applies(conditions: dict, connection: ConnectionInfo) -> bool:
    """Check whether a conditional component applies to the given connection."""
    if not conditions:
        return True
    for key, required_value in conditions.items():
        if key == "has_separate_injection_point" and connection.electricity.has_separate_injection_point != required_value:
            return False
    return True


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calculate_dynamic_electricity_cost(
    energy: EnergySeries,
    market_prices: MarketPriceSeries,
    tariff: TariffDefinition,
    connection: ConnectionInfo,
    monthly_peaks: MonthlyPeaks,
    grid_tariffs: GridTariffSet,
    tax_rules: list[TaxRule],
) -> CostResult:
    """Calculate itemised cost for dynamic electricity (15-min resolution).

    Parameters
    ----------
    energy:
        15-min electricity consumption/injection series with a ``kwh`` column.
    market_prices:
        15-min day-ahead price series aligned to the energy index.
    tariff:
        Supplier-agnostic dynamic tariff product definition.
    connection:
        Customer connection metadata (region, DSO, customer type, …).
    monthly_peaks:
        Monthly peak demand in kW (one value per calendar month in the period).
    grid_tariffs:
        Regulated grid tariff set for the connection's DSO.
    tax_rules:
        Applicable tax/levy rules for the connection's region.

    Returns
    -------
    CostResult
        Itemised breakdown DataFrame, totals dict, and assumptions.
    """
    _validate_dynamic_inputs(energy, market_prices, tariff)

    energy_df = energy.data
    backend = energy_df.implementation
    settlement_minutes = tariff.settlement_interval_minutes

    timestamps = [_to_datetime(value) for value in energy_df["timestamp"].to_list()]
    prices_eur_per_mwh = market_prices.data["price"].cast(nw.Float64) * market_prices.factor_to_eur_per_mwh()
    vat_rate = connection.get_vat_rate()
    month_keys = [f"{ts.year:04d}-{ts.month:02d}" for ts in timestamps]

    work_df = nw.from_dict(
        {
            "timestamp": timestamps,
            "month": month_keys,
        },
        backend=backend,
    ).with_columns(
        energy_df["kwh"].cast(nw.Float64).alias("kwh"),
        prices_eur_per_mwh.alias("price_eur_per_mwh"),
    )
    month_counts = work_df.group_by("month").agg(nw.len().alias("n_in_month"))
    work_df = work_df.join(month_counts, on="month", how="left")

    breakdown_dict: dict[str, nw.Series] = {
        COL_SUPPLIER_ENERGY: _compute_supplier_energy(work_df["kwh"], work_df["price_eur_per_mwh"], tariff),
        COL_SUPPLIER_FIXED: _compute_supplier_fixed(work_df, tariff, settlement_minutes),
        COL_GRID_VARIABLE: _compute_grid_variable(work_df["kwh"], grid_tariffs),
        COL_GRID_FIXED: _compute_grid_fixed(work_df, grid_tariffs, connection, settlement_minutes),
        COL_CAPACITY: _compute_capacity(work_df, monthly_peaks, grid_tariffs),
        COL_TAXES: _compute_taxes(work_df, tax_rules, settlement_minutes),
    }
    breakdown_dict[COL_VAT] = _compute_vat(breakdown_dict, vat_rate, tariff, grid_tariffs, tax_rules)
    breakdown_dict[COL_TOTAL] = (
        breakdown_dict[COL_SUPPLIER_ENERGY]
        + breakdown_dict[COL_SUPPLIER_FIXED]
        + breakdown_dict[COL_GRID_VARIABLE]
        + breakdown_dict[COL_GRID_FIXED]
        + breakdown_dict[COL_CAPACITY]
        + breakdown_dict[COL_TAXES]
        + breakdown_dict[COL_VAT]
    )

    out = work_df.select(
        "timestamp",
        breakdown_dict[COL_SUPPLIER_ENERGY].alias(COL_SUPPLIER_ENERGY),
        breakdown_dict[COL_SUPPLIER_FIXED].alias(COL_SUPPLIER_FIXED),
        breakdown_dict[COL_GRID_VARIABLE].alias(COL_GRID_VARIABLE),
        breakdown_dict[COL_GRID_FIXED].alias(COL_GRID_FIXED),
        breakdown_dict[COL_CAPACITY].alias(COL_CAPACITY),
        breakdown_dict[COL_TAXES].alias(COL_TAXES),
        breakdown_dict[COL_VAT].alias(COL_VAT),
        breakdown_dict[COL_TOTAL].alias(COL_TOTAL),
    )
    native_breakdown = out.to_native()

    totals = {column: float(values.sum()) for column, values in breakdown_dict.items()}

    assumptions = {
        "vat_rate": vat_rate,
        "contract_type": tariff.contract_type.value,
        "tariff_name": tariff.name,
        "dso": connection.dso,
        "region": connection.region.value,
        "excise_tier_note": "POC uses flat lowest-tier rate; full tiered calculation planned for v1",
    }

    return CostResult(breakdown=native_breakdown, totals=totals, assumptions=assumptions)
