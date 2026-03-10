"""Dynamic Electricity Tariff — Flanders POC.

This marimo notebook demonstrates the end-to-end calculation of dynamic
electricity costs for a residential consumer in Flanders (Fluvius Antwerpen).
"""

import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    mo.md(
        """
        # ⚡ Dynamic Electricity Tariff — Flanders POC

        This notebook demonstrates the **energy-cost** library calculating an
        itemised cost breakdown for a dynamic electricity tariff in Flanders.

        **Scenario:**
        - Residential consumer, Fluvius Antwerpen, LV digital meter
        - Dynamic offtake tariff: `c€/kWh = 0.108 × Belpex15′ + 1.625`
        - Supplier fixed fee: 70.75 EUR/year
        - Grid tariffs & taxes: Fluvius Antwerpen 2026 regulated rates
        - VAT: 21% (residential)
        """
    )
    return (mo,)


@app.cell
def _():
    from datetime import date

    import numpy as np
    import pandas as pd

    from energy_cost.core.engine import calculate_dynamic_electricity_cost
    from energy_cost.models import (
        Carrier,
        Channel,
        ConnectionInfo,
        ContractType,
        CustomerType,
        Direction,
        ElectricityConnection,
        EnergySeries,
        GridTariffSet,
        Market,
        MarketPriceSeries,
        MeterRegister,
        MeterType,
        MonthlyPeaks,
        PricingKind,
        PricingRule,
        Region,
        TariffComponent,
        TariffComponentType,
        TariffDefinition,
        TaxRule,
        VoltageLevel,
    )

    return (
        Carrier,
        Channel,
        ConnectionInfo,
        ContractType,
        CustomerType,
        Direction,
        ElectricityConnection,
        EnergySeries,
        GridTariffSet,
        Market,
        MarketPriceSeries,
        MeterRegister,
        MeterType,
        MonthlyPeaks,
        PricingKind,
        PricingRule,
        Region,
        TariffComponent,
        TariffComponentType,
        TariffDefinition,
        TaxRule,
        VoltageLevel,
        calculate_dynamic_electricity_cost,
        date,
        np,
        pd,
    )


@app.cell
def _(mo):
    mo.md("""
    ## 1. Synthetic input data

    We generate a realistic week of 15-minute consumption data and
    day-ahead market prices for January 2026.
    """)
    return


@app.cell
def _(np, pd):
    # --- Synthetic consumption: realistic residential profile ---
    np.random.seed(2026)
    index = pd.date_range("2026-01-12", periods=7 * 96, freq="15min", tz="Europe/Brussels")

    # Base load ~0.3 kW with morning/evening peaks
    hours = np.asarray(index.hour + index.minute / 60.0, dtype=float)
    base_load = 0.075  # kWh per 15 min ≈ 0.3 kW
    morning_peak = 0.15 * np.exp(-((hours - 7.5) ** 2) / 2)
    evening_peak = 0.25 * np.exp(-((hours - 19.0) ** 2) / 3)
    noise = np.random.normal(0, 0.02, size=len(index)).clip(-0.05, 0.05)
    consumption_kwh = (base_load + morning_peak + evening_peak + noise).clip(0.01)

    energy_df = pd.DataFrame({"kwh": consumption_kwh}, index=index)
    return consumption_kwh, energy_df, index


@app.cell
def _(index, np, pd):
    # --- Synthetic day-ahead prices: ~80 EUR/MWh with daily pattern ---
    hours_p = np.asarray(index.hour + index.minute / 60.0, dtype=float)
    base_price = 80.0
    daily_pattern = 30 * np.sin((hours_p - 6) * np.pi / 12).clip(0) + 10 * np.random.normal(0, 1, len(index))
    prices_eur_per_mwh = (base_price + daily_pattern).clip(10, 300)

    price_series = pd.Series(prices_eur_per_mwh, index=index)
    return (price_series,)


@app.cell
def _(energy_df, mo, pd, price_series):
    _fig_data = pd.DataFrame(
        {"Consumption (kWh)": energy_df["kwh"], "Price (EUR/MWh)": price_series}
    )
    mo.md(
        f"""
        ### Input data preview

        **Consumption** — {len(energy_df)} quarter-hourly readings, total {energy_df['kwh'].sum():.1f} kWh

        **Day-ahead prices** — mean {price_series.mean():.1f} EUR/MWh,
        range [{price_series.min():.1f}, {price_series.max():.1f}] EUR/MWh
        """
    )
    return


@app.cell
def _(energy_df, mo, price_series):
    import altair as _alt

    _energy_df = energy_df.reset_index()
    _price_df = price_series.reset_index().rename(columns={"index": "time", 0: "price"})

    _chart_e_spec = (
        _alt.Chart(_energy_df, title="Consumption (kWh per 15 min)")
        .mark_line()
        .encode(x="index:T", y="kwh:Q")
        .properties(width="container", height=200)
    )
    _chart_p_spec = (
        _alt.Chart(_price_df, title="Day-ahead price (EUR/MWh)")
        .mark_line()
        .encode(x="time:T", y="price:Q")
        .properties(width="container", height=200)
    )

    _chart_e = mo.ui.altair_chart(
        _chart_e_spec
    )
    _chart_p = mo.ui.altair_chart(
        _chart_p_spec
    )
    mo.vstack([_chart_e, _chart_p])
    return


@app.cell
def _(mo):
    mo.md("""
    ## 2. Define tariff & connection

    We set up an EBEM-style dynamic tariff and a residential connection at
    Fluvius Antwerpen.
    """)
    return


@app.cell
def _(
    Carrier,
    Channel,
    ConnectionInfo,
    ContractType,
    CustomerType,
    Direction,
    ElectricityConnection,
    MeterRegister,
    MeterType,
    PricingKind,
    PricingRule,
    Region,
    TariffComponent,
    TariffComponentType,
    TariffDefinition,
    VoltageLevel,
    date,
):
    tariff = TariffDefinition(
        name="EBEM Dynamic 2026",
        carrier=Carrier.ELECTRICITY,
        direction=Direction.OFFTAKE,
        contract_type=ContractType.DYNAMIC,
        components=[
            TariffComponent(
                component_type=TariffComponentType.ENERGY,
                channel=Channel.KWH,
                pricing=PricingRule(
                    kind=PricingKind.INDEXED_LINEAR,
                    index_name="Belpex15'",
                    coef=0.108,
                    add_cents_per_kwh=1.625,
                ),
                vat_applicable=True,
            ),
            TariffComponent(
                component_type=TariffComponentType.FIXED_FEE,
                pricing=PricingRule(kind=PricingKind.FIXED_PER_YEAR, eur_per_year=70.75),
                vat_applicable=True,
            ),
        ],
        valid_from=date(2026, 1, 1),
        valid_to=date(2026, 12, 31),
    )

    connection = ConnectionInfo(
        region=Region.BE_VLG,
        dso="Fluvius Antwerpen",
        customer_type=CustomerType.RESIDENTIAL,
        electricity=ElectricityConnection(
            meter_register=MeterRegister.SINGLE,
            meter_type=MeterType.DIGITAL,
            voltage_level=VoltageLevel.LV,
        ),
    )
    return connection, tariff


@app.cell
def _(MonthlyPeaks, consumption_kwh, index, pd):
    # Compute monthly peak from the consumption data (kW = kWh / 0.25h)
    power_kw = consumption_kwh / 0.25
    peak_per_month = pd.Series(power_kw, index=index).resample("MS").max()
    monthly_peaks = MonthlyPeaks(data=peak_per_month)
    return (monthly_peaks,)


@app.cell
def _(mo):
    mo.md("""
    ## 3. Calculate costs

    We resolve the regulated grid tariffs and tax rules, then run the
    engine to produce an itemised breakdown.
    """)
    return


@app.cell
def _(
    Carrier,
    Direction,
    EnergySeries,
    GridTariffSet,
    Market,
    MarketPriceSeries,
    TaxRule,
    calculate_dynamic_electricity_cost,
    connection,
    energy_df,
    monthly_peaks,
    price_series,
    tariff,
):
    energy = EnergySeries(carrier=Carrier.ELECTRICITY, direction=Direction.OFFTAKE, data=energy_df)
    prices = MarketPriceSeries(market=Market.EPEX_DA_BE_15MIN, data=price_series)

    grid = GridTariffSet.resolve("energy_cost.regions.be_flanders", connection, Carrier.ELECTRICITY, Direction.OFFTAKE)
    taxes = TaxRule.resolve("energy_cost.regions.be_flanders", connection, Carrier.ELECTRICITY, Direction.OFFTAKE)

    result = calculate_dynamic_electricity_cost(
        energy=energy,
        market_prices=prices,
        tariff=tariff,
        connection=connection,
        monthly_peaks=monthly_peaks,
        grid_tariffs=grid,
        tax_rules=taxes,
    )
    return (result,)


@app.cell
def _(mo, pd, result):
    _totals = pd.Series(result.totals).round(4)
    _nice_names = {
        "supplier_energy_eur": "⚡ Supplier energy",
        "supplier_fixed_eur": "📋 Supplier fixed fee",
        "grid_variable_eur": "🔌 Grid variable",
        "grid_fixed_eur": "🔌 Grid fixed",
        "capacity_eur": "📊 Capacity tariff",
        "taxes_eur": "🏛️ Taxes & levies",
        "vat_eur": "💰 VAT (21%)",
        "total_eur": "**💶 TOTAL**",
    }
    summary = pd.DataFrame(
        {"Component": [_nice_names.get(k, k) for k in _totals.index], "EUR": _totals.values}
    )
    mo.md("### Cost summary (billing period totals)")
    return (summary,)


@app.cell
def _(mo, summary):
    mo.ui.table(summary, selection=None)
    return


@app.cell
def _(mo):
    mo.md("""
    ### Breakdown over time

    The chart below shows cost components stacked per quarter-hour,
    letting you see how costs shift with market prices.
    """)
    return


@app.cell
def _(mo, result):
    _cols = [
        "supplier_energy_eur",
        "supplier_fixed_eur",
        "grid_variable_eur",
        "grid_fixed_eur",
        "capacity_eur",
        "taxes_eur",
        "vat_eur",
    ]

    _melt = (
        result.breakdown[_cols]
        .reset_index()
        .melt(id_vars=result.breakdown.index.name or "index", value_vars=_cols, var_name="component", value_name="EUR")
    )

    import altair as alt

    _chart = (
        alt.Chart(_melt, title="Cost breakdown per 15 min")
        .mark_area()
        .encode(
            x=alt.X(f"{result.breakdown.index.name or 'index'}:T", title="Time"),
            y=alt.Y("EUR:Q", title="EUR per 15 min", stack=True),
            color=alt.Color("component:N", title="Component"),
        )
        .properties(width="container", height=350)
    )
    mo.ui.altair_chart(_chart)
    return


@app.cell
def _(mo, result):
    mo.md(f"""
    ### Assumptions

    ```
    {chr(10).join(f'{k}: {v}' for k, v in result.assumptions.items())}
    ```
    """)
    return


if __name__ == "__main__":
    app.run()
