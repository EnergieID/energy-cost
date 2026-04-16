from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from energy_cost.meter import CostGroup, TariffCategory
from energy_cost.tax import ColumnPattern, Tax, TaxRule, TaxVersion, _matches_pattern, _specificity

# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------


class TestMatchesPattern:
    def test_exact_match(self) -> None:
        assert _matches_pattern(
            (TariffCategory.FEES, CostGroup.CONSUMPTION, "excise"),
            (TariffCategory.FEES, CostGroup.CONSUMPTION, "excise"),
        )

    def test_no_match(self) -> None:
        assert not _matches_pattern(
            (TariffCategory.FEES, CostGroup.CONSUMPTION, "excise"), (TariffCategory.FEES, CostGroup.CONSUMPTION, "vat")
        )

    def test_wildcard_first(self) -> None:
        assert _matches_pattern(
            ("*", CostGroup.CONSUMPTION, "excise"), (TariffCategory.SUPPLIER, CostGroup.CONSUMPTION, "excise")
        )

    def test_wildcard_middle(self) -> None:
        assert _matches_pattern(
            (TariffCategory.FEES, "*", "excise"), (TariffCategory.FEES, CostGroup.CAPACITY, "excise")
        )

    def test_wildcard_last(self) -> None:
        assert _matches_pattern(
            (TariffCategory.FEES, CostGroup.CONSUMPTION, "*"), (TariffCategory.FEES, CostGroup.CONSUMPTION, "anything")
        )

    def test_all_wildcards(self) -> None:
        assert _matches_pattern(("*", "*", "*"), (TariffCategory.FEES, CostGroup.CONSUMPTION, "excise"))

    def test_mixed_wildcards(self) -> None:
        assert _matches_pattern(
            ("*", CostGroup.CAPACITY, "*"), (TariffCategory.DISTRIBUTOR, CostGroup.CAPACITY, "peak_demand")
        )
        assert not _matches_pattern(
            ("*", CostGroup.CAPACITY, "*"), (TariffCategory.DISTRIBUTOR, CostGroup.CONSUMPTION, "energy")
        )


# ---------------------------------------------------------------------------
# Specificity – rightmost positions weighted higher
# ---------------------------------------------------------------------------


class TestSpecificity:
    def test_all_wildcards(self) -> None:
        assert _specificity(("*", "*", "*")) == 0

    def test_all_literal(self) -> None:
        # 2^0 + 2^1 + 2^2 = 7
        assert _specificity((TariffCategory.FEES, CostGroup.CONSUMPTION, "excise")) == 7

    def test_rightmost_more_significant(self) -> None:
        # (*, *, d) = 2^2 = 4; (a, *, *) = 2^0 = 1
        assert _specificity(("*", "*", "excise")) > _specificity((TariffCategory.FEES, "*", "*"))

    def test_ordering_matches_user_spec(self) -> None:
        """Verify the full ordering matches the user's binary-number spec."""
        patterns: list[ColumnPattern] = [
            (TariffCategory.FEES, CostGroup.CONSUMPTION, "excise"),  # 7
            ("*", CostGroup.CONSUMPTION, "excise"),  # 6
            (TariffCategory.FEES, "*", "excise"),  # 5
            ("*", "*", "excise"),  # 4
            (TariffCategory.FEES, CostGroup.CONSUMPTION, "*"),  # 3
            ("*", CostGroup.CONSUMPTION, "*"),  # 2
            (TariffCategory.FEES, "*", "*"),  # 1
            ("*", "*", "*"),  # 0
        ]
        specs = [_specificity(p) for p in patterns]
        assert specs == sorted(specs, reverse=True)


# ---------------------------------------------------------------------------
# TaxVersion.apply – subtraction-based algorithm
# ---------------------------------------------------------------------------


class TestTaxVersionApply:
    def test_default_rate_uses_grand_total(self) -> None:
        """With no rules, default rate is applied via the (*,*,*) total."""
        v = TaxVersion(start=dt.datetime(2025, 1, 1), default=0.10)
        timestamps = pd.date_range("2025-01-01", periods=2, freq="MS", tz="UTC")
        df = pd.DataFrame(
            {
                ("supplier", "consumption", "energy"): [100.0, 200.0],
                ("supplier", "consumption", "total"): [100.0, 200.0],
                ("supplier", "total", "total"): [100.0, 200.0],
                ("total", "total", "total"): [100.0, 200.0],
            }
        )
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        df.insert(0, "timestamp", timestamps)
        result = v.apply(df)
        # Uses ("total","total","total") × 0.10
        assert result[("taxes", "total", "total")].tolist() == pytest.approx([10.0, 20.0])

    def test_specific_rule_then_default_remainder(self) -> None:
        """A specific rule taxes its part, then the default taxes the rest via totals."""
        v = TaxVersion(
            start=dt.datetime(2025, 1, 1),
            default=0.06,
            rates=[TaxRule(rate=0.21, columns=[("*", CostGroup.CAPACITY, "*")])],
        )
        df = pd.DataFrame(
            {
                ("supplier", "consumption", "energy"): [100.0],
                ("supplier", "consumption", "total"): [100.0],
                ("supplier", "capacity", "total"): [50.0],
                ("supplier", "total", "total"): [150.0],
                ("total", "total", "total"): [150.0],
            }
        )
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        df.insert(0, "timestamp", pd.date_range("2025-01-01", periods=1, freq="MS", tz="UTC"))
        result = v.apply(df)
        # capacity: 50 * 0.21 = 10.5
        # remainder: (150 - 50) * 0.06 = 6.0
        assert result[("taxes", "total", "total")].iloc[0] == pytest.approx(16.5)

    def test_zero_rate_subtracts_from_totals(self) -> None:
        """A zero-rate rule removes its amount from what the default sees."""
        v = TaxVersion(
            start=dt.datetime(2025, 1, 1),
            default=0.10,
            rates=[TaxRule(rate=0.0, columns=[("*", CostGroup.INJECTION, "*")])],
        )
        df = pd.DataFrame(
            {
                ("supplier", "consumption", "energy"): [100.0],
                ("supplier", "consumption", "total"): [100.0],
                ("supplier", "injection", "energy"): [30.0],
                ("supplier", "injection", "total"): [30.0],
                ("supplier", "total", "total"): [130.0],
                ("total", "total", "total"): [130.0],
            }
        )
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        df.insert(0, "timestamp", pd.date_range("2025-01-01", periods=1, freq="MS", tz="UTC"))
        result = v.apply(df)
        # injection: 30 * 0.0 = 0; remainder: (130 - 30) * 0.10 = 10
        assert result[("taxes", "total", "total")].iloc[0] == pytest.approx(10.0)

    def test_multiple_categories(self) -> None:
        """Tax works correctly across supplier + distributor categories."""
        v = TaxVersion(
            start=dt.datetime(2025, 1, 1),
            default=0.06,
            rates=[TaxRule(rate=0.21, columns=[("*", CostGroup.CAPACITY, "*")])],
        )
        df = pd.DataFrame(
            {
                ("supplier", "consumption", "energy"): [100.0],
                ("supplier", "consumption", "total"): [100.0],
                ("supplier", "total", "total"): [100.0],
                ("distributor", "capacity", "total"): [50.0],
                ("distributor", "consumption", "energy"): [40.0],
                ("distributor", "consumption", "total"): [40.0],
                ("distributor", "total", "total"): [90.0],
                ("total", "total", "total"): [190.0],
            }
        )
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        df.insert(0, "timestamp", pd.date_range("2025-01-01", periods=1, freq="MS", tz="UTC"))
        result = v.apply(df)
        # capacity: 50 * 0.21 = 10.5
        # remainder: (190 - 50) * 0.06 = 8.4
        assert result[("taxes", "total", "total")].iloc[0] == pytest.approx(18.9)

    def test_named_and_unnamed_costs_no_double_count(self) -> None:
        """When a category has both named and unnamed costs (total > sum of named),
        the unnamed portion is correctly taxed via the subtraction algorithm."""
        v = TaxVersion(
            start=dt.datetime(2025, 1, 1),
            default=0.06,
            rates=[TaxRule(rate=0.21, columns=[(TariffCategory.FEES, CostGroup.CONSUMPTION, "excise")])],
        )
        # fees/consumption has excise=20 + unnamed=30 → total=50
        df = pd.DataFrame(
            {
                ("fees", "consumption", "excise"): [20.0],
                ("fees", "consumption", "total"): [50.0],
                ("fees", "total", "total"): [50.0],
                ("total", "total", "total"): [50.0],
            }
        )
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        df.insert(0, "timestamp", pd.date_range("2025-01-01", periods=1, freq="MS", tz="UTC"))
        result = v.apply(df)
        # excise: 20 * 0.21 = 4.2
        # remainder: (50 - 20) * 0.06 = 1.8
        assert result[("taxes", "total", "total")].iloc[0] == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# Tax – YAML round-trip
# ---------------------------------------------------------------------------


class TestTaxYaml:
    def test_from_yaml(self, tmp_path) -> None:
        yaml_content = """
- start: 2025-01-01T00:00:00+01:00
  default: 0.06
  rates:
    - rate: 0.21
      columns:
        - ["*", "capacity", "*"]
    - rate: 0.0
      columns:
        - ["*", "injection", "*"]
- start: 2026-01-01T00:00:00+01:00
  default: 0.09
"""
        path = tmp_path / "tax.yml"
        path.write_text(yaml_content, encoding="utf-8")

        tax = Tax.from_yaml(path)

        assert len(tax.versions) == 2
        assert tax.versions[0].default == 0.06
        assert len(tax.versions[0].rates) == 2
        assert tax.versions[0].rates[0].rate == 0.21
        assert tax.versions[0].rates[0].columns == [("*", "capacity", "*")]
        assert tax.versions[1].default == 0.09
        assert tax.versions[1].rates == []


# ---------------------------------------------------------------------------
# Tax.apply – version switching
# ---------------------------------------------------------------------------


class TestTaxApply:
    def test_single_version(self) -> None:
        tax = Tax(versions=[TaxVersion(start=dt.datetime(2025, 1, 1), default=0.10)])
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2025-03-01", periods=2, freq="MS"),
                ("total", "total", "total"): [100.0, 200.0],
            }
        )
        df.columns = pd.Index(["timestamp"]).append(pd.MultiIndex.from_tuples([("total", "total", "total")]))
        result = tax.apply(df)
        assert result is not None
        assert result[("taxes", "total", "total")].tolist() == pytest.approx([10.0, 20.0])

    def test_version_switch_at_boundary(self) -> None:
        tax = Tax(
            versions=[
                TaxVersion(start=dt.datetime(2025, 1, 1, tzinfo=dt.UTC), default=0.06),
                TaxVersion(start=dt.datetime(2026, 1, 1, tzinfo=dt.UTC), default=0.10),
            ]
        )
        df = pd.DataFrame(
            {
                "timestamp": [
                    pd.Timestamp("2025-06-01", tz="UTC"),
                    pd.Timestamp("2026-06-01", tz="UTC"),
                ],
                ("total", "total", "total"): [100.0, 100.0],
            }
        )
        df.columns = pd.Index(["timestamp"]).append(pd.MultiIndex.from_tuples([("total", "total", "total")]))
        result = tax.apply(df, timezone=dt.UTC)
        assert result is not None
        assert result[("taxes", "total", "total")].iloc[0] == pytest.approx(6.0)
        assert result[("taxes", "total", "total")].iloc[1] == pytest.approx(10.0)
