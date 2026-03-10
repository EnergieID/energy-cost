"""Tests for dynamic electricity cost calculation (Flanders POC)."""

from __future__ import annotations

import pandas as pd
import pytest

from energy_cost.core.engine import (
    COL_CAPACITY,
    COL_GRID_FIXED,
    COL_GRID_VARIABLE,
    COL_SUPPLIER_ENERGY,
    COL_SUPPLIER_FIXED,
    COL_TAXES,
    COL_TOTAL,
    COL_VAT,
    calculate_dynamic_electricity_cost,
)
from energy_cost.models import (
    Carrier,
    ConnectionInfo,
    Direction,
    EnergySeries,
    GridTariffSet,
    Market,
    MarketPriceSeries,
    MonthlyPeaks,
    TariffDefinition,
    TaxRule,
)

ALL_COLUMNS = [
    COL_SUPPLIER_ENERGY,
    COL_SUPPLIER_FIXED,
    COL_GRID_VARIABLE,
    COL_GRID_FIXED,
    COL_CAPACITY,
    COL_TAXES,
    COL_VAT,
    COL_TOTAL,
]


class TestDynamicElectricityCost:
    """End-to-end tests for calculate_dynamic_electricity_cost."""

    def _run(
        self,
        energy: EnergySeries,
        prices: MarketPriceSeries,
        tariff: TariffDefinition,
        connection: ConnectionInfo,
        peaks: MonthlyPeaks,
    ):
        grid = GridTariffSet.resolve("energy_cost.regions.be_flanders", connection, Carrier.ELECTRICITY, Direction.OFFTAKE)
        taxes = TaxRule.resolve("energy_cost.regions.be_flanders", connection, Carrier.ELECTRICITY, Direction.OFFTAKE)
        return calculate_dynamic_electricity_cost(
            energy=energy,
            market_prices=prices,
            tariff=tariff,
            connection=connection,
            monthly_peaks=peaks,
            grid_tariffs=grid,
            tax_rules=taxes,
        )

    def test_breakdown_has_all_columns(
        self, constant_energy, constant_prices, dynamic_offtake_tariff, residential_connection, monthly_peaks_january
    ):
        result = self._run(constant_energy, constant_prices, dynamic_offtake_tariff, residential_connection, monthly_peaks_january)
        for col in ALL_COLUMNS:
            assert col in result.breakdown.columns, f"Missing column: {col}"

    def test_breakdown_length_matches_input(
        self, constant_energy, constant_prices, dynamic_offtake_tariff, residential_connection, monthly_peaks_january
    ):
        result = self._run(constant_energy, constant_prices, dynamic_offtake_tariff, residential_connection, monthly_peaks_january)
        assert len(result.breakdown) == len(constant_energy.data)

    def test_total_is_sum_of_components(
        self, constant_energy, constant_prices, dynamic_offtake_tariff, residential_connection, monthly_peaks_january
    ):
        result = self._run(constant_energy, constant_prices, dynamic_offtake_tariff, residential_connection, monthly_peaks_january)
        component_cols = [c for c in ALL_COLUMNS if c != COL_TOTAL]
        expected_total = result.breakdown[component_cols].sum(axis=1)
        pd.testing.assert_series_equal(result.breakdown[COL_TOTAL], expected_total, check_names=False)

    def test_totals_dict_matches_breakdown_sums(
        self, constant_energy, constant_prices, dynamic_offtake_tariff, residential_connection, monthly_peaks_january
    ):
        result = self._run(constant_energy, constant_prices, dynamic_offtake_tariff, residential_connection, monthly_peaks_january)
        for col in ALL_COLUMNS:
            assert result.totals[col] == pytest.approx(result.breakdown[col].sum(), rel=1e-9)

    def test_supplier_energy_calculation(
        self, constant_energy, constant_prices, dynamic_offtake_tariff, residential_connection, monthly_peaks_january
    ):
        """Verify supplier energy with known values.

        Formula: c€/kWh = 0.108 × 100 + 1.625 = 12.425 c€/kWh = 0.12425 EUR/kWh
        Per timestep: 0.12425 × 0.25 = 0.0310625 EUR
        Daily total: 0.0310625 × 96 = 2.982 EUR
        """
        result = self._run(constant_energy, constant_prices, dynamic_offtake_tariff, residential_connection, monthly_peaks_january)

        expected_per_ts = (0.108 * 100.0 + 1.625) / 100.0 * 0.25
        assert result.breakdown[COL_SUPPLIER_ENERGY].iloc[0] == pytest.approx(expected_per_ts)
        assert result.totals[COL_SUPPLIER_ENERGY] == pytest.approx(expected_per_ts * 96, rel=1e-9)

    def test_grid_variable_calculation(
        self, constant_energy, constant_prices, dynamic_offtake_tariff, residential_connection, monthly_peaks_january
    ):
        """Verify grid variable cost.

        Netgebruik: 0.0234492 EUR/kWh
        ODV: 0.0254845 EUR/kWh
        Total rate: 0.0489337 EUR/kWh
        Per timestep: 0.0489337 × 0.25 = 0.012233425 EUR
        """
        result = self._run(constant_energy, constant_prices, dynamic_offtake_tariff, residential_connection, monthly_peaks_january)

        expected_rate = 0.0234492 + 0.0254845
        expected_per_ts = expected_rate * 0.25
        assert result.breakdown[COL_GRID_VARIABLE].iloc[0] == pytest.approx(expected_per_ts, rel=1e-6)

    def test_grid_fixed_proration(
        self, constant_energy, constant_prices, dynamic_offtake_tariff, residential_connection, monthly_peaks_january
    ):
        """Verify grid fixed proration (databeheer 17.85 EUR/year).

        Per timestep = 17.85 × 15 / (365 × 1440) = 17.85 / 35040 ≈ 0.000509247 EUR
        """
        result = self._run(constant_energy, constant_prices, dynamic_offtake_tariff, residential_connection, monthly_peaks_january)

        expected_per_ts = 17.85 * 15 / (365 * 24 * 60)
        assert result.breakdown[COL_GRID_FIXED].iloc[0] == pytest.approx(expected_per_ts, rel=1e-6)

    def test_capacity_allocation(
        self, constant_energy, constant_prices, dynamic_offtake_tariff, residential_connection, monthly_peaks_january
    ):
        """Verify capacity charge allocation.

        Rate: 4.1169714 EUR/kW/month, peak: 2.5 kW
        Monthly cost: 4.1169714 × 2.5 = 10.2924285 EUR
        January has 31×96 = 2976 timesteps
        Per timestep (for a full month): 10.2924285 / 2976 ≈ 0.003458 EUR
        But we only have 96 timesteps (1 day), so per timestep = 10.2924285 / 96 ≈ 0.107213 EUR
        """
        result = self._run(constant_energy, constant_prices, dynamic_offtake_tariff, residential_connection, monthly_peaks_january)

        monthly_cap_cost = 4.1169714 * 2.5
        # The function groups by month; our 96 timesteps are all in January
        expected_per_ts = monthly_cap_cost / 96
        assert result.breakdown[COL_CAPACITY].iloc[0] == pytest.approx(expected_per_ts, rel=1e-6)

    def test_taxes_calculation(
        self, constant_energy, constant_prices, dynamic_offtake_tariff, residential_connection, monthly_peaks_january
    ):
        """Verify tax calculation.

        Excise (tier 1): 0.050329 EUR/kWh
        Energy fund: 0.0020417 EUR/kWh
        Total tax rate: 0.0523707 EUR/kWh
        Per timestep: 0.0523707 × 0.25 = 0.013092675 EUR
        """
        result = self._run(constant_energy, constant_prices, dynamic_offtake_tariff, residential_connection, monthly_peaks_january)

        expected_tax_rate = 0.050329 + 0.0020417
        expected_per_ts = expected_tax_rate * 0.25
        assert result.breakdown[COL_TAXES].iloc[0] == pytest.approx(expected_per_ts, rel=1e-6)

    def test_vat_is_21_percent(
        self, constant_energy, constant_prices, dynamic_offtake_tariff, residential_connection, monthly_peaks_january
    ):
        """Verify VAT = 21% of all pre-VAT components."""
        result = self._run(constant_energy, constant_prices, dynamic_offtake_tariff, residential_connection, monthly_peaks_january)

        pre_vat_cols = [c for c in ALL_COLUMNS if c not in (COL_VAT, COL_TOTAL)]
        pre_vat_total = result.breakdown[pre_vat_cols].sum(axis=1)
        expected_vat = pre_vat_total * 0.21
        pd.testing.assert_series_equal(result.breakdown[COL_VAT], expected_vat, check_names=False, rtol=1e-9)

    def test_no_nan_in_breakdown(
        self, constant_energy, constant_prices, dynamic_offtake_tariff, residential_connection, monthly_peaks_january
    ):
        result = self._run(constant_energy, constant_prices, dynamic_offtake_tariff, residential_connection, monthly_peaks_january)
        assert not result.breakdown.isna().any().any(), "Breakdown contains NaN values"

    def test_all_costs_positive(
        self, constant_energy, constant_prices, dynamic_offtake_tariff, residential_connection, monthly_peaks_january
    ):
        result = self._run(constant_energy, constant_prices, dynamic_offtake_tariff, residential_connection, monthly_peaks_january)
        assert (result.breakdown[COL_TOTAL] > 0).all(), "Total cost should be positive for offtake"

    def test_assumptions_populated(
        self, constant_energy, constant_prices, dynamic_offtake_tariff, residential_connection, monthly_peaks_january
    ):
        result = self._run(constant_energy, constant_prices, dynamic_offtake_tariff, residential_connection, monthly_peaks_january)
        assert result.assumptions["vat_rate"] == 0.21
        assert result.assumptions["contract_type"] == "dynamic"
        assert result.assumptions["dso"] == "Fluvius Antwerpen"


class TestInputValidation:
    """Tests for input validation in the engine."""

    def test_rejects_gas_carrier(
        self, constant_prices, dynamic_offtake_tariff, residential_connection, monthly_peaks_january, one_day_index
    ):
        gas_energy = EnergySeries(
            carrier=Carrier.GAS,
            direction=Direction.OFFTAKE,
            data=pd.DataFrame({"timestamp": one_day_index, "kwh": 0.25}),
        )
        grid = GridTariffSet.resolve("energy_cost.regions.be_flanders", residential_connection, Carrier.ELECTRICITY, Direction.OFFTAKE)
        taxes = TaxRule.resolve("energy_cost.regions.be_flanders", residential_connection, Carrier.ELECTRICITY, Direction.OFFTAKE)
        with pytest.raises(ValueError, match="Expected electricity"):
            calculate_dynamic_electricity_cost(
                gas_energy, constant_prices, dynamic_offtake_tariff, residential_connection,
                monthly_peaks_january, grid, taxes,
            )

    def test_rejects_missing_kwh_column(
        self, constant_prices, dynamic_offtake_tariff, residential_connection, monthly_peaks_january, one_day_index
    ):
        bad_energy = EnergySeries(
            carrier=Carrier.ELECTRICITY,
            direction=Direction.OFFTAKE,
            data=pd.DataFrame({"timestamp": one_day_index, "kwh_day": 0.15, "kwh_night": 0.10}),
        )
        grid = GridTariffSet.resolve("energy_cost.regions.be_flanders", residential_connection, Carrier.ELECTRICITY, Direction.OFFTAKE)
        taxes = TaxRule.resolve("energy_cost.regions.be_flanders", residential_connection, Carrier.ELECTRICITY, Direction.OFFTAKE)
        with pytest.raises(ValueError, match="'kwh' column"):
            calculate_dynamic_electricity_cost(
                bad_energy, constant_prices, dynamic_offtake_tariff, residential_connection,
                monthly_peaks_january, grid, taxes,
            )

    def test_rejects_misaligned_prices(
        self, constant_energy, dynamic_offtake_tariff, residential_connection, monthly_peaks_january
    ):
        wrong_index = pd.date_range("2026-01-16", periods=96, freq="15min", tz="Europe/Brussels")
        bad_prices = MarketPriceSeries(
            market=Market.EPEX_DA_BE_15MIN,
            data=pd.DataFrame({"timestamp": wrong_index, "price": 100.0}),
        )
        grid = GridTariffSet.resolve("energy_cost.regions.be_flanders", residential_connection, Carrier.ELECTRICITY, Direction.OFFTAKE)
        taxes = TaxRule.resolve("energy_cost.regions.be_flanders", residential_connection, Carrier.ELECTRICITY, Direction.OFFTAKE)
        with pytest.raises(ValueError, match="does not align"):
            calculate_dynamic_electricity_cost(
                constant_energy, bad_prices, dynamic_offtake_tariff, residential_connection,
                monthly_peaks_january, grid, taxes,
            )


class TestVariablePrices:
    """Tests with non-constant market prices to verify per-timestep calculation."""

    def test_varying_prices(self, dynamic_offtake_tariff, residential_connection, monthly_peaks_january, one_day_index):
        """Each timestep should reflect its own market price."""
        import numpy as np

        np.random.seed(42)
        prices_array = np.random.uniform(20, 200, size=96)
        kwh_array = np.random.uniform(0.0, 0.5, size=96)

        energy = EnergySeries(
            carrier=Carrier.ELECTRICITY,
            direction=Direction.OFFTAKE,
            data=pd.DataFrame({"timestamp": one_day_index, "kwh": kwh_array}),
        )
        prices = MarketPriceSeries(
            market=Market.EPEX_DA_BE_15MIN,
            data=pd.DataFrame({"timestamp": one_day_index, "price": prices_array}),
        )

        grid = GridTariffSet.resolve("energy_cost.regions.be_flanders", residential_connection, Carrier.ELECTRICITY, Direction.OFFTAKE)
        taxes = TaxRule.resolve("energy_cost.regions.be_flanders", residential_connection, Carrier.ELECTRICITY, Direction.OFFTAKE)
        result = calculate_dynamic_electricity_cost(
            energy, prices, dynamic_offtake_tariff, residential_connection,
            monthly_peaks_january, grid, taxes,
        )

        # Verify a specific timestep
        i = 10
        expected_energy_price = (0.108 * prices_array[i] + 1.625) / 100.0
        expected_supplier = expected_energy_price * kwh_array[i]
        assert result.breakdown[COL_SUPPLIER_ENERGY].iloc[i] == pytest.approx(expected_supplier, rel=1e-9)

        # Total still consistent
        component_cols = [c for c in ALL_COLUMNS if c != COL_TOTAL]
        expected_total = result.breakdown[component_cols].sum(axis=1)
        pd.testing.assert_series_equal(result.breakdown[COL_TOTAL], expected_total, check_names=False)
