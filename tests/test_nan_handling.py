"""Tests for consistent NaN handling.

These tests verify the intended NaN semantics:
- Legitimate NaN: index gaps or formula producing NaN → stays NaN in output
- Structural zero: a cost column absent in a tariff version but present in others → zero
- Totals with NaN: any NaN component makes the total NaN (skipna=False)
- Aggregation with NaN: any NaN sub-period in a bin makes the bin NaN
"""

from __future__ import annotations

import datetime as dt

import isodate
import pandas as pd
import pytest

from energy_cost.contract import Contract
from energy_cost.formula import IndexFormula
from energy_cost.formula.index import IndexAdder
from energy_cost.index import DataFrameIndex, Index
from energy_cost.meter import CostGroup, Meter, TimeseriesFrame
from energy_cost.tariff import Tariff
from energy_cost.tariff_version import TariffVersion
from energy_cost.versioning import sum_frames

# ---------------------------------------------------------------------------
# NaN propagation: meter data shorter than requested range
# ---------------------------------------------------------------------------


class TestNaNFromMissingMeterData:
    """When start/end is explicitly provided and extends beyond the meter data,
    the output should include rows for the full billing window. Periods without
    meter data should be NaN."""

    def test_tariff_produces_nan_for_month_without_meter_data(self) -> None:
        """A tariff applied over [Jan, Mar) with meter data only in January
        should produce NaN for February."""
        tariff = Tariff(
            [
                TariffVersion(
                    start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
                    consumption={"energy": IndexFormula(constant_cost=10.0)},
                )
            ]
        )
        # Meter data only in January
        timestamps = pd.date_range("2025-01-01", "2025-02-01", freq="h", inclusive="left", tz=dt.UTC)
        meter = Meter(measurements=TimeseriesFrame(pd.DataFrame({"timestamp": timestamps, "value": 1.0})))

        result = tariff.apply(
            meter,
            start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
            end=dt.datetime(2025, 3, 1, tzinfo=dt.UTC),
            output_resolution=isodate.Duration(months=1),
        )

        assert result is not None
        assert len(result) == 2
        # January: has data → real value
        jan = result[result["timestamp"] == pd.Timestamp("2025-01-01", tz="UTC")]
        assert pd.notna(jan[(CostGroup.CONSUMPTION, "energy")].iloc[0])
        # February: no meter data → NaN
        feb = result[result["timestamp"] == pd.Timestamp("2025-02-01", tz="UTC")]
        assert pd.isna(feb[(CostGroup.CONSUMPTION, "energy")].iloc[0])

    def test_nan_in_energy_propagates_to_total(self) -> None:
        """If energy is NaN for a period, the total must also be NaN."""
        tariff = Tariff(
            [
                TariffVersion(
                    start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
                    consumption={"energy": IndexFormula(constant_cost=10.0)},
                )
            ]
        )
        timestamps = pd.date_range("2025-01-01", "2025-02-01", freq="h", inclusive="left", tz=dt.UTC)
        meter = Meter(measurements=TimeseriesFrame(pd.DataFrame({"timestamp": timestamps, "value": 1.0})))

        result = tariff.apply(
            meter,
            start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
            end=dt.datetime(2025, 3, 1, tzinfo=dt.UTC),
            output_resolution=isodate.Duration(months=1),
        )

        assert result is not None
        feb = result[result["timestamp"] == pd.Timestamp("2025-02-01", tz="UTC")]
        assert pd.isna(feb[(CostGroup.CONSUMPTION, "total")].iloc[0])
        assert pd.isna(feb[("total", "total")].iloc[0])

    def test_contract_nan_for_month_without_data(self) -> None:
        """Contract grand total is NaN for months without meter data."""
        contract = Contract(
            supplier=Tariff(
                [
                    TariffVersion(
                        start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
                        consumption={"energy": IndexFormula(constant_cost=10.0)},
                    )
                ]
            ),
            timezone=dt.UTC,
        )
        timestamps = pd.date_range("2025-01-01", "2025-02-01", freq="h", inclusive="left", tz=dt.UTC)
        meter = Meter(measurements=TimeseriesFrame(pd.DataFrame({"timestamp": timestamps, "value": 1.0})))

        result = contract.apply(
            meter,
            start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
            end=dt.datetime(2025, 3, 1, tzinfo=dt.UTC),
            output_resolution=isodate.Duration(months=1),
        )

        assert len(result) == 2
        feb = result[result["timestamp"] == pd.Timestamp("2025-02-01", tz="UTC")]
        assert pd.isna(feb[("total", "total", "total")].iloc[0])


# ---------------------------------------------------------------------------
# NaN spec: explicit start/end vs meter coverage
# ---------------------------------------------------------------------------


class TestNaNBinSpecs:
    """Tests for the core NaN spec: bins produce NaN when the requested [start, end)
    range extends beyond the available data (meter or index)."""

    def test_meter_shorter_than_requested_range_produces_nan(self) -> None:
        """Meter 2025-01-01 to 2025-01-15, monthly output [Jan, Feb) → NaN on Jan row."""
        tariff = Tariff(
            [
                TariffVersion(
                    start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
                    consumption={"energy": IndexFormula(constant_cost=10.0)},
                )
            ]
        )
        # Meter only covers first 14 days (not the full month)
        timestamps = pd.date_range("2025-01-01", "2025-01-15", freq="h", inclusive="left", tz=dt.UTC)
        meter = Meter(measurements=TimeseriesFrame(pd.DataFrame({"timestamp": timestamps, "value": 1.0})))

        result = tariff.apply(
            meter,
            start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
            end=dt.datetime(2025, 2, 1, tzinfo=dt.UTC),
            output_resolution=isodate.Duration(months=1),
        )

        assert result is not None
        assert len(result) == 1
        # The monthly bin is incomplete (meter doesn't cover Jan 15-31) → NaN
        assert pd.isna(result[(CostGroup.CONSUMPTION, "energy")].iloc[0])

    def test_meter_covers_full_requested_range_no_nan(self) -> None:
        """Meter 2025-01-01 to 2025-02-15, monthly output [Jan, Feb) → no NaN."""
        tariff = Tariff(
            [
                TariffVersion(
                    start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
                    consumption={"energy": IndexFormula(constant_cost=10.0)},
                )
            ]
        )
        # Meter covers more than the requested range
        timestamps = pd.date_range("2025-01-01", "2025-02-15", freq="h", inclusive="left", tz=dt.UTC)
        meter = Meter(measurements=TimeseriesFrame(pd.DataFrame({"timestamp": timestamps, "value": 1.0})))

        result = tariff.apply(
            meter,
            start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
            end=dt.datetime(2025, 2, 1, tzinfo=dt.UTC),
            output_resolution=isodate.Duration(months=1),
        )

        assert result is not None
        assert len(result) == 1
        assert pd.notna(result[(CostGroup.CONSUMPTION, "energy")].iloc[0])

    def test_meter_matches_requested_range_but_not_full_bin_produces_nan(self) -> None:
        """Meter 2025-01-01 to 2025-01-15, monthly output [Jan 1, Jan 15) → NaN.
        The output bin is snapped to a full month, so the meter doesn't cover it."""
        tariff = Tariff(
            [
                TariffVersion(
                    start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
                    consumption={"energy": IndexFormula(constant_cost=10.0)},
                )
            ]
        )
        timestamps = pd.date_range("2025-01-01", "2025-01-15", freq="h", inclusive="left", tz=dt.UTC)
        meter = Meter(measurements=TimeseriesFrame(pd.DataFrame({"timestamp": timestamps, "value": 1.0})))

        result = tariff.apply(
            meter,
            start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
            end=dt.datetime(2025, 1, 15, tzinfo=dt.UTC),
            output_resolution=isodate.Duration(months=1),
        )

        assert result is not None
        # Bin is a full month but data only covers half → NaN
        assert pd.isna(result[(CostGroup.CONSUMPTION, "energy")].iloc[0])

    def test_meter_covers_only_first_month_of_year_request(self) -> None:
        """Meter 2025-01-01 to 2025-02-15, monthly output [Jan, Jan next year) →
        no NaN on first row, NaN on all other rows."""
        tariff = Tariff(
            [
                TariffVersion(
                    start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
                    consumption={"energy": IndexFormula(constant_cost=10.0)},
                )
            ]
        )
        timestamps = pd.date_range("2025-01-01", "2025-02-15", freq="h", inclusive="left", tz=dt.UTC)
        meter = Meter(measurements=TimeseriesFrame(pd.DataFrame({"timestamp": timestamps, "value": 1.0})))

        result = tariff.apply(
            meter,
            start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
            end=dt.datetime(2026, 1, 1, tzinfo=dt.UTC),
            output_resolution=isodate.Duration(months=1),
        )

        assert result is not None
        assert len(result) == 12
        # January: full coverage → no NaN
        assert pd.notna(result[(CostGroup.CONSUMPTION, "energy")].iloc[0])
        # All other months: no data → NaN
        for i in range(1, 12):
            assert pd.isna(result[(CostGroup.CONSUMPTION, "energy")].iloc[i]), f"Row {i} should be NaN"

    def test_index_shorter_than_requested_range_produces_nan(self) -> None:
        """Same as meter test but the gap is in the index, not the meter.
        Index covers only first 14 days, meter covers the full month."""
        index_ts = pd.date_range("2025-01-01", "2025-01-15", freq="h", inclusive="left", tz=dt.UTC)
        index_df = pd.DataFrame({"timestamp": index_ts, "value": 5.0})
        Index.register("test_spec_short_index", DataFrameIndex(index_df, forward_fill=False))

        formula = IndexFormula(variable_costs=[IndexAdder(index="test_spec_short_index", scalar=1.0)])
        tariff = Tariff([TariffVersion(start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC), consumption={"energy": formula})])

        # Meter covers full month
        meter_ts = pd.date_range("2025-01-01", "2025-02-01", freq="h", inclusive="left", tz=dt.UTC)
        meter = Meter(measurements=TimeseriesFrame(pd.DataFrame({"timestamp": meter_ts, "value": 1.0})))

        result = tariff.apply(
            meter,
            start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
            end=dt.datetime(2025, 2, 1, tzinfo=dt.UTC),
            output_resolution=isodate.Duration(months=1),
        )

        assert result is not None
        assert len(result) == 1
        # Index gap after Jan 14 → NaN for the monthly bin
        assert pd.isna(result[(CostGroup.CONSUMPTION, "energy")].iloc[0])

    def test_index_covers_full_range_no_nan(self) -> None:
        """Index and meter both cover the full requested range → no NaN."""
        index_ts = pd.date_range("2025-01-01", "2025-02-01", freq="h", inclusive="left", tz=dt.UTC)
        index_df = pd.DataFrame({"timestamp": index_ts, "value": 5.0})
        Index.register("test_spec_full_index", DataFrameIndex(index_df, forward_fill=False))

        formula = IndexFormula(variable_costs=[IndexAdder(index="test_spec_full_index", scalar=1.0)])
        tariff = Tariff([TariffVersion(start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC), consumption={"energy": formula})])

        meter_ts = pd.date_range("2025-01-01", "2025-02-01", freq="h", inclusive="left", tz=dt.UTC)
        meter = Meter(measurements=TimeseriesFrame(pd.DataFrame({"timestamp": meter_ts, "value": 1.0})))

        result = tariff.apply(
            meter,
            start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
            end=dt.datetime(2025, 2, 1, tzinfo=dt.UTC),
            output_resolution=isodate.Duration(months=1),
        )

        assert result is not None
        assert len(result) == 1
        assert pd.notna(result[(CostGroup.CONSUMPTION, "energy")].iloc[0])
        # 744 hours × 1 MWh × 5 €/MWh = 3720
        assert result[(CostGroup.CONSUMPTION, "energy")].iloc[0] == pytest.approx(3720.0)


# ---------------------------------------------------------------------------
# NaN propagation: index with missing values
# ---------------------------------------------------------------------------


class TestNaNPropagatesFromIndexGaps:
    """When an index has missing values for some timestamps, those should
    propagate as NaN through the formula and into the output."""

    def test_index_gap_produces_nan_in_daily_output(self) -> None:
        """An index missing values for part of the day produces NaN for the daily bin."""
        # Index only covers the first 12 hours of the day
        index_ts = pd.date_range("2025-01-01", periods=12, freq="h", tz=dt.UTC)
        index_df = pd.DataFrame({"timestamp": index_ts, "value": 5.0})
        Index.register("test_gap_index", DataFrameIndex(index_df, forward_fill=False))

        formula = IndexFormula(variable_costs=[IndexAdder(index="test_gap_index", scalar=1.0)])
        tariff = Tariff([TariffVersion(start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC), consumption={"energy": formula})])

        # Meter covers full day
        meter_ts = pd.date_range("2025-01-01", periods=24, freq="h", tz=dt.UTC)
        meter = Meter(measurements=TimeseriesFrame(pd.DataFrame({"timestamp": meter_ts, "value": 1.0})))

        result = tariff.apply(
            meter,
            start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
            end=dt.datetime(2025, 1, 2, tzinfo=dt.UTC),
            output_resolution=dt.timedelta(days=1),
        )

        assert result is not None
        # The daily bin has NaN sub-periods (hours 12-23 have no index value) → result is NaN
        assert pd.isna(result[(CostGroup.CONSUMPTION, "energy")].iloc[0])

    def test_index_gap_produces_nan_total(self) -> None:
        """NaN in energy from an index gap propagates to the total."""
        index_ts = pd.date_range("2025-01-01", periods=12, freq="h", tz=dt.UTC)
        index_df = pd.DataFrame({"timestamp": index_ts, "value": 5.0})
        Index.register("test_gap_index_2", DataFrameIndex(index_df, forward_fill=False))

        formula = IndexFormula(variable_costs=[IndexAdder(index="test_gap_index_2", scalar=1.0)])
        tariff = Tariff([TariffVersion(start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC), consumption={"energy": formula})])

        meter_ts = pd.date_range("2025-01-01", periods=24, freq="h", tz=dt.UTC)
        meter = Meter(measurements=TimeseriesFrame(pd.DataFrame({"timestamp": meter_ts, "value": 1.0})))

        result = tariff.apply(
            meter,
            start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
            end=dt.datetime(2025, 1, 2, tzinfo=dt.UTC),
            output_resolution=dt.timedelta(days=1),
        )

        assert result is not None
        assert pd.isna(result[(CostGroup.CONSUMPTION, "total")].iloc[0])
        assert pd.isna(result[("total", "total")].iloc[0])

    def test_complete_index_produces_real_value(self) -> None:
        """An index covering the full range produces a real (non-NaN) value."""
        index_ts = pd.date_range("2025-01-01", periods=24, freq="h", tz=dt.UTC)
        index_df = pd.DataFrame({"timestamp": index_ts, "value": 5.0})
        Index.register("test_full_index", DataFrameIndex(index_df, forward_fill=False))

        formula = IndexFormula(variable_costs=[IndexAdder(index="test_full_index", scalar=1.0)])
        tariff = Tariff([TariffVersion(start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC), consumption={"energy": formula})])

        meter_ts = pd.date_range("2025-01-01", periods=24, freq="h", tz=dt.UTC)
        meter = Meter(measurements=TimeseriesFrame(pd.DataFrame({"timestamp": meter_ts, "value": 1.0})))

        result = tariff.apply(
            meter,
            start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
            end=dt.datetime(2025, 1, 2, tzinfo=dt.UTC),
            output_resolution=dt.timedelta(days=1),
        )

        assert result is not None
        # 24 hours × 1 MWh × 5 €/MWh = 120
        assert result[(CostGroup.CONSUMPTION, "energy")].iloc[0] == pytest.approx(120.0)

    def test_contract_nan_propagates_to_grand_total(self) -> None:
        """Contract grand total is NaN when a component has an index gap."""
        index_ts = pd.date_range("2025-01-01", periods=12, freq="h", tz=dt.UTC)
        index_df = pd.DataFrame({"timestamp": index_ts, "value": 5.0})
        Index.register("test_gap_index_3", DataFrameIndex(index_df, forward_fill=False))

        formula = IndexFormula(variable_costs=[IndexAdder(index="test_gap_index_3", scalar=1.0)])
        contract = Contract(
            supplier=Tariff(
                [TariffVersion(start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC), consumption={"energy": formula})]
            ),
            timezone=dt.UTC,
        )

        meter_ts = pd.date_range("2025-01-01", periods=24, freq="h", tz=dt.UTC)
        meter = Meter(measurements=TimeseriesFrame(pd.DataFrame({"timestamp": meter_ts, "value": 1.0})))

        result = contract.apply(
            meter,
            start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
            end=dt.datetime(2025, 1, 2, tzinfo=dt.UTC),
            output_resolution=dt.timedelta(days=1),
        )

        assert pd.isna(result[("total", "total", "total")].iloc[0])


# ---------------------------------------------------------------------------
# Structural zeros: version boundary with disappearing cost
# ---------------------------------------------------------------------------


class TestStructuralZerosAtVersionBoundary:
    """When a cost column exists in one tariff version but not another,
    the missing column should be treated as zero (not NaN)."""

    def test_cost_removed_in_new_version_becomes_zero(self) -> None:
        """A cost that exists in v1 but not v2 is zero in v2's period."""
        tariff = Tariff(
            [
                TariffVersion(
                    start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
                    consumption={
                        "energy": IndexFormula(constant_cost=10.0),
                        "surcharge": IndexFormula(constant_cost=5.0),
                    },
                ),
                TariffVersion(
                    start=dt.datetime(2025, 2, 1, tzinfo=dt.UTC),
                    consumption={
                        "energy": IndexFormula(constant_cost=12.0),
                        # surcharge removed in v2
                    },
                ),
            ]
        )
        # Full data for both months
        timestamps = pd.date_range("2025-01-01", "2025-03-01", freq="h", inclusive="left", tz=dt.UTC)
        meter = Meter(measurements=TimeseriesFrame(pd.DataFrame({"timestamp": timestamps, "value": 1.0})))

        result = tariff.apply(
            meter,
            start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
            end=dt.datetime(2025, 3, 1, tzinfo=dt.UTC),
            output_resolution=isodate.Duration(months=1),
        )

        assert result is not None
        # January: surcharge exists → real value
        jan = result[result["timestamp"] == pd.Timestamp("2025-01-01", tz="UTC")]
        assert jan[(CostGroup.CONSUMPTION, "surcharge")].iloc[0] == pytest.approx(5.0 * 744)  # 744 hours in Jan 2025

        # February: surcharge absent in v2 → structural zero
        feb = result[result["timestamp"] == pd.Timestamp("2025-02-01", tz="UTC")]
        assert feb[(CostGroup.CONSUMPTION, "surcharge")].iloc[0] == 0.0
        assert pd.notna(feb[(CostGroup.CONSUMPTION, "surcharge")].iloc[0])

    def test_cost_added_in_new_version_is_zero_in_old(self) -> None:
        """A cost that only exists in v2 is zero in v1's period."""
        tariff = Tariff(
            [
                TariffVersion(
                    start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
                    consumption={"energy": IndexFormula(constant_cost=10.0)},
                ),
                TariffVersion(
                    start=dt.datetime(2025, 2, 1, tzinfo=dt.UTC),
                    consumption={
                        "energy": IndexFormula(constant_cost=10.0),
                        "green_tax": IndexFormula(constant_cost=2.0),
                    },
                ),
            ]
        )
        timestamps = pd.date_range("2025-01-01", "2025-03-01", freq="h", inclusive="left", tz=dt.UTC)
        meter = Meter(measurements=TimeseriesFrame(pd.DataFrame({"timestamp": timestamps, "value": 1.0})))

        result = tariff.apply(
            meter,
            start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
            end=dt.datetime(2025, 3, 1, tzinfo=dt.UTC),
            output_resolution=isodate.Duration(months=1),
        )

        assert result is not None
        # January: green_tax absent in v1 → structural zero
        jan = result[result["timestamp"] == pd.Timestamp("2025-01-01", tz="UTC")]
        assert jan[(CostGroup.CONSUMPTION, "green_tax")].iloc[0] == 0.0
        assert pd.notna(jan[(CostGroup.CONSUMPTION, "green_tax")].iloc[0])

        # February: green_tax exists → real value
        feb = result[result["timestamp"] == pd.Timestamp("2025-02-01", tz="UTC")]
        assert feb[(CostGroup.CONSUMPTION, "green_tax")].iloc[0] == pytest.approx(2.0 * 672)  # 672 hours in Feb 2025

    def test_totals_are_correct_across_version_boundary(self) -> None:
        """Totals reflect real values + structural zeros correctly (not NaN)."""
        tariff = Tariff(
            [
                TariffVersion(
                    start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
                    consumption={
                        "energy": IndexFormula(constant_cost=10.0),
                        "surcharge": IndexFormula(constant_cost=5.0),
                    },
                ),
                TariffVersion(
                    start=dt.datetime(2025, 2, 1, tzinfo=dt.UTC),
                    consumption={"energy": IndexFormula(constant_cost=12.0)},
                ),
            ]
        )
        timestamps = pd.date_range("2025-01-01", "2025-03-01", freq="h", inclusive="left", tz=dt.UTC)
        meter = Meter(measurements=TimeseriesFrame(pd.DataFrame({"timestamp": timestamps, "value": 1.0})))

        result = tariff.apply(
            meter,
            start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
            end=dt.datetime(2025, 3, 1, tzinfo=dt.UTC),
            output_resolution=isodate.Duration(months=1),
        )

        # February total = energy only (surcharge is structural zero)
        assert result is not None
        feb = result[result["timestamp"] == pd.Timestamp("2025-02-01", tz="UTC")]
        expected_energy = 12.0 * 672
        assert feb[(CostGroup.CONSUMPTION, "total")].iloc[0] == pytest.approx(expected_energy)
        assert feb[("total", "total")].iloc[0] == pytest.approx(expected_energy)
        # No NaN in any column
        assert feb.drop(columns="timestamp").notna().all().all()


# ---------------------------------------------------------------------------
# Aggregation: NaN within a bin
# ---------------------------------------------------------------------------


class TestAggregationNaN:
    """When aggregating sub-periods into a coarser bin, any NaN sub-period
    makes the entire bin NaN."""

    def test_partial_data_in_weekly_bin_produces_nan(self) -> None:
        """A weekly bin where meter only covers part of the period produces NaN.
        The formula generates the full [start, end) grid; timestamps without
        meter data become NaN; skipna=False makes the bin NaN."""
        tariff = Tariff(
            [
                TariffVersion(
                    start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
                    consumption={"energy": IndexFormula(constant_cost=10.0)},
                )
            ]
        )
        # Meter data only for first 3 days of the week (Mon-Wed)
        timestamps = pd.date_range("2025-01-06", periods=3 * 24, freq="h", tz=dt.UTC)  # Mon Jan 6
        meter = Meter(measurements=TimeseriesFrame(pd.DataFrame({"timestamp": timestamps, "value": 1.0})))

        result = tariff.apply(
            meter,
            start=dt.datetime(2025, 1, 6, tzinfo=dt.UTC),
            end=dt.datetime(2025, 1, 13, tzinfo=dt.UTC),
            output_resolution=dt.timedelta(days=7),
        )

        assert result is not None
        # Meter doesn't cover the full weekly bin → NaN
        assert pd.isna(result[(CostGroup.CONSUMPTION, "energy")].iloc[0])

    def test_full_data_in_bin_produces_real_value(self) -> None:
        """A weekly bin with complete data produces a real value."""
        tariff = Tariff(
            [
                TariffVersion(
                    start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
                    consumption={"energy": IndexFormula(constant_cost=10.0)},
                )
            ]
        )
        # Full week of data
        timestamps = pd.date_range("2025-01-06", "2025-01-13", freq="h", inclusive="left", tz=dt.UTC)
        meter = Meter(measurements=TimeseriesFrame(pd.DataFrame({"timestamp": timestamps, "value": 1.0})))

        result = tariff.apply(
            meter,
            start=dt.datetime(2025, 1, 6, tzinfo=dt.UTC),
            end=dt.datetime(2025, 1, 13, tzinfo=dt.UTC),
            output_resolution=dt.timedelta(days=7),
        )

        assert result is not None
        assert result[(CostGroup.CONSUMPTION, "energy")].iloc[0] == pytest.approx(10.0 * 168)  # 168 hours

    def test_empty_bin_produces_nan(self) -> None:
        """An output bin with NO meter data at all produces NaN."""
        tariff = Tariff(
            [
                TariffVersion(
                    start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC),
                    consumption={"energy": IndexFormula(constant_cost=10.0)},
                )
            ]
        )
        # Meter data for week 1 only
        timestamps = pd.date_range("2025-01-06", "2025-01-13", freq="h", inclusive="left", tz=dt.UTC)
        meter = Meter(measurements=TimeseriesFrame(pd.DataFrame({"timestamp": timestamps, "value": 1.0})))

        result = tariff.apply(
            meter,
            start=dt.datetime(2025, 1, 6, tzinfo=dt.UTC),
            end=dt.datetime(2025, 1, 20, tzinfo=dt.UTC),  # 2 weeks requested
            output_resolution=dt.timedelta(days=7),
        )

        assert result is not None
        assert len(result) == 2
        # Week 1 has data → real value
        assert result[(CostGroup.CONSUMPTION, "energy")].iloc[0] == pytest.approx(10.0 * 168)
        # Week 2 has NO data → NaN
        assert pd.isna(result[(CostGroup.CONSUMPTION, "energy")].iloc[1])


# ---------------------------------------------------------------------------
# sum_frames helper
# ---------------------------------------------------------------------------


class TestSumFrames:
    """Unit tests for the sum_frames helper function."""

    def test_structural_zero_for_absent_columns(self) -> None:
        """Columns missing in one frame get zero, not NaN."""
        f1 = pd.DataFrame({"timestamp": [1, 2], "a": [1.0, 2.0], "b": [3.0, 4.0]})
        f2 = pd.DataFrame({"timestamp": [1, 2], "a": [10.0, 20.0]})  # no column 'b'

        result = sum_frames([f1, f2])

        assert result.set_index("timestamp")["a"].tolist() == [11.0, 22.0]
        # 'b' in f1 + structural zero from f2 = same as f1
        assert result.set_index("timestamp")["b"].tolist() == [3.0, 4.0]

    def test_nan_propagates_in_existing_columns(self) -> None:
        """NaN in a column that exists in the frame is preserved through the sum."""
        f1 = pd.DataFrame({"timestamp": [1, 2], "a": [1.0, float("nan")]})
        f2 = pd.DataFrame({"timestamp": [1, 2], "a": [10.0, 20.0]})

        result = sum_frames([f1, f2])

        assert result.set_index("timestamp")["a"].iloc[0] == 11.0
        assert pd.isna(result.set_index("timestamp")["a"].iloc[1])

    def test_structural_zero_for_absent_rows(self) -> None:
        """Rows missing in one frame get zero (structural), not NaN."""
        f1 = pd.DataFrame({"timestamp": [1, 2, 3], "a": [1.0, 2.0, 3.0]})
        f2 = pd.DataFrame({"timestamp": [2, 3, 4], "a": [20.0, 30.0, 40.0]})

        result = sum_frames([f1, f2]).sort_values("timestamp").reset_index(drop=True)

        expected = {1: 1.0, 2: 22.0, 3: 33.0, 4: 40.0}
        for _, row in result.iterrows():
            assert row["a"] == expected[row["timestamp"]]
