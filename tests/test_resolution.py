import datetime
import datetime as dt
from typing import cast

import isodate
import pandas as pd
import pytest

from energy_cost.resolution import (
    _find_common_divisor,
    align_datetime_to_tz,
    align_timestamps_to_tz,
    detect_resolution_and_range,
    is_divisor,
    parse_resolution,
    redistribute_to_resolution,
    snap_billing_period,
    to_pandas_freq,
)


def test_to_pandas_freq_correctly_handles_monthly_durations():
    assert to_pandas_freq(isodate.parse_duration("P1M")) == "1MS"
    assert to_pandas_freq(isodate.parse_duration("P3M")) == "3MS"


def test_to_pandas_freq_correctly_handles_yearly_durations():
    assert to_pandas_freq(isodate.parse_duration("P1Y")) == "1YS"
    assert to_pandas_freq(isodate.parse_duration("P2Y")) == "2YS"


def test_complex_durations_are_not_supported():
    with pytest.raises(ValueError):
        to_pandas_freq(isodate.parse_duration("P1Y2M"))
    with pytest.raises(ValueError):
        to_pandas_freq(isodate.parse_duration("P2MT1H"))


def test_to_pandas_freq_correctly_handles_simple_timedeltas():
    assert to_pandas_freq(isodate.parse_duration("PT45S")) == "45s"
    assert to_pandas_freq(isodate.parse_duration("PT15M")) == "15min"
    assert to_pandas_freq(isodate.parse_duration("PT1H")) == "1h"
    assert to_pandas_freq(isodate.parse_duration("P1D")) == "1D"


def test_to_pandas_freq_correctly_handles_complex_timedeltas():
    assert to_pandas_freq(isodate.parse_duration("P1DT1H")) == "25h"
    assert to_pandas_freq(isodate.parse_duration("P1DT1H30M")) == "1530min"
    assert to_pandas_freq(isodate.parse_duration("PT1M30S")) == "90s"


def test_is_divisor_is_always_true_when_calendar_periods_are_divided_by_timedelta_that_is_a_divisor_of_1_day():
    assert is_divisor(isodate.parse_duration("P1M"), isodate.parse_duration("PT1H"))
    assert is_divisor(isodate.parse_duration("P1Y"), isodate.parse_duration("PT15M"))
    assert is_divisor(isodate.parse_duration("P1M"), isodate.parse_duration("P1D"))


def test_is_divisor_is_false_when_calendar_periods_are_divided_by_timedelta_that_is_not_a_divisor_of_1_day():
    assert not is_divisor(isodate.parse_duration("P1M"), isodate.parse_duration("PT7H"))
    assert not is_divisor(isodate.parse_duration("P1Y"), isodate.parse_duration("PT23M"))
    assert not is_divisor(isodate.parse_duration("P1M"), isodate.parse_duration("PT25H"))


def test_is_divisor_is_always_false_when_fixed_timedeltas_are_divided_by_calendar_periods():
    assert not is_divisor(isodate.parse_duration("PT1H"), isodate.parse_duration("P1M"))
    assert not is_divisor(isodate.parse_duration("PT10000H"), isodate.parse_duration("P1M"))
    assert not is_divisor(isodate.parse_duration("PT15M"), isodate.parse_duration("P1Y"))


def test_is_divisor_works_as_expected_for_timedelta_divisors():
    assert is_divisor(isodate.parse_duration("PT1H"), isodate.parse_duration("PT15M"))
    assert is_divisor(isodate.parse_duration("PT1H"), isodate.parse_duration("PT30M"))
    assert not is_divisor(isodate.parse_duration("PT1H"), isodate.parse_duration("PT7M"))
    assert is_divisor(isodate.parse_duration("P1D"), isodate.parse_duration("PT4H"))
    assert is_divisor(isodate.parse_duration("PT15M"), isodate.parse_duration("PT3M"))
    assert not is_divisor(isodate.parse_duration("PT15M"), isodate.parse_duration("PT4M"))


def test_is_divisor_works_as_expected_for_calendar_divisors():
    assert is_divisor(isodate.parse_duration("P1Y"), isodate.parse_duration("P1M"))
    assert is_divisor(isodate.parse_duration("P1Y"), isodate.parse_duration("P2M"))
    assert not is_divisor(isodate.parse_duration("P1Y"), isodate.parse_duration("P7M"))
    assert is_divisor(isodate.parse_duration("P2Y"), isodate.parse_duration("P6M"))
    assert not is_divisor(isodate.parse_duration("P2Y"), isodate.parse_duration("P7M"))
    assert is_divisor(isodate.parse_duration("P7Y"), isodate.parse_duration("P7M"))
    assert is_divisor(isodate.parse_duration("P3M"), isodate.parse_duration("P1M"))
    assert not is_divisor(isodate.parse_duration("P3M"), isodate.parse_duration("P2M"))


def test_detect_resolution_infers_monthly_and_yearly_resolutions_correctly():
    from energy_cost.resolution import detect_resolution

    timestamps = pd.Series(pd.date_range("2020-01-01", periods=12, freq="MS"))
    assert detect_resolution(timestamps) == isodate.parse_duration("P1M")

    timestamps = pd.Series(pd.date_range("2020-01-01", periods=5, freq="YS"))
    assert detect_resolution(timestamps) == isodate.parse_duration("P1Y")


def test_detect_resolution_falls_back_to_timedelta_for_non_calendar_resolutions():
    from energy_cost.resolution import detect_resolution

    timestamps = pd.Series(pd.date_range("2020-01-01", periods=4, freq="15min"))
    assert detect_resolution(timestamps) == isodate.parse_duration("PT15M")


def test_detect_resolution_raises_when_to_few_timestamps():
    from energy_cost.resolution import detect_resolution

    timestamps = pd.Series(pd.date_range("2020-01-01", periods=1, freq="15min"))
    with pytest.raises(ValueError):
        detect_resolution(timestamps)


def test_parse_resolution_correctly_parses_iso_strings():

    assert parse_resolution("PT15M") == datetime.timedelta(minutes=15)
    assert parse_resolution("P1M") == isodate.Duration(months=1)
    assert parse_resolution("P1Y") == isodate.Duration(years=1)


def test_detect_resolution_and_range_raises_when_no_timestamps():
    with pytest.raises(ValueError):
        detect_resolution_and_range(pd.DataFrame(columns=["timestamp", "value"]))


# ---------------------------------------------------------------------------
# align_datetime_to_tz
# ---------------------------------------------------------------------------


def test_align_datetime_to_tz_with_none_tz_returns_naive_unchanged() -> None:
    d = dt.datetime(2025, 3, 1, 12, 0)
    assert align_datetime_to_tz(d, None) == d


def test_align_datetime_to_tz_with_none_tz_strips_timezone_from_aware_datetime() -> None:
    tz = dt.timezone(dt.timedelta(hours=1))
    d = dt.datetime(2025, 3, 1, 12, 0, tzinfo=tz)
    result = align_datetime_to_tz(d, None)
    assert result == dt.datetime(2025, 3, 1, 12, 0)
    assert result.tzinfo is None


def test_align_datetime_to_tz_localizes_naive_to_given_tz() -> None:
    tz = dt.timezone(dt.timedelta(hours=1))
    d = dt.datetime(2025, 3, 1, 12, 0)
    result = align_datetime_to_tz(d, tz)
    assert result.tzinfo is not None
    assert result.utcoffset() == dt.timedelta(hours=1)


def test_align_datetime_to_tz_converts_aware_to_target_tz() -> None:
    utc = dt.UTC
    cet = dt.timezone(dt.timedelta(hours=1))
    d = dt.datetime(2025, 1, 1, 0, 0, tzinfo=utc)  # midnight UTC
    result = align_datetime_to_tz(d, cet)
    # UTC midnight is 01:00 CET
    assert result.replace(tzinfo=None) == dt.datetime(2025, 1, 1, 1, 0)
    assert result.utcoffset() == dt.timedelta(hours=1)


# ---------------------------------------------------------------------------
# align_timestamps_to_tz — mixed UTC offsets
# ---------------------------------------------------------------------------


def test_align_timestamps_to_tz_handles_mixed_utc_offsets():
    """Timestamps with different UTC offsets (e.g. across a DST boundary) produce an
    object-dtype column in pandas. align_timestamps_to_tz must handle this without
    raising AttributeError."""
    from zoneinfo import ZoneInfo

    tz = ZoneInfo("Europe/Brussels")
    data = pd.DataFrame(
        {
            "timestamp": [
                dt.datetime.fromisoformat("2024-03-31T01:45:00+01:00"),  # CET
                dt.datetime.fromisoformat("2024-03-31T03:00:00+02:00"),  # CEST (after spring-forward)
            ],
            "value": [150.5, 75.3],
        }
    )

    # Confirm the precondition: mixed offsets produce object dtype
    assert data["timestamp"].dtype == object

    # Must not raise, and result must be datetime64 in the target timezone
    result = align_timestamps_to_tz(data, tz)

    assert result["timestamp"].dtype != object
    assert str(result["timestamp"].dt.tz) == str(tz)
    assert result["timestamp"].iloc[0] == pd.Timestamp("2024-03-31T01:45:00", tz=tz)
    assert result["timestamp"].iloc[1] == pd.Timestamp("2024-03-31T03:00:00", tz=tz)


# ---------------------------------------------------------------------------
# redistribute_to_resolution
# ---------------------------------------------------------------------------


def test_same_resolution_returns_values_unchanged() -> None:
    """When source and output resolution are both monthly, values must pass through unchanged.

    Previously this caused division by zero (→ infinity) because the equal case
    entered the distribution branch instead of short-circuiting to the aggregate path.
    """
    start = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
    end = dt.datetime(2025, 4, 1, tzinfo=dt.UTC)

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="MS", tz=dt.UTC),
            "value": [10.0, 20.0, 30.0],
        }
    )

    result = redistribute_to_resolution(df, isodate.parse_duration("P1M"), isodate.parse_duration("P1M"), start, end)

    assert len(result) == 3
    assert result["value"].tolist() == pytest.approx([10.0, 20.0, 30.0])
    assert not result["value"].isin([float("inf"), float("-inf")]).any()


def test_distribute_partial_window_divides_by_full_source_period() -> None:
    """A single daily value distributed into 5-min slots for only 15 min of the day.
    Even though only 3 slots fall inside the output window, the divisor must be
    288 — so each slot receives 288 / 288 = 1, not 288 / 3 = 96.
    """
    start = dt.datetime(2025, 1, 1, 0, 0, tzinfo=dt.UTC)
    end = dt.datetime(2025, 1, 1, 0, 15, tzinfo=dt.UTC)

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=1, freq="1D", tz=dt.UTC),
            "value": [288.0],
        }
    )

    result = redistribute_to_resolution(df, dt.timedelta(days=1), dt.timedelta(minutes=5), start, end)

    assert len(result) == 3
    assert result["value"].iloc[0] == pytest.approx(1.0)
    assert result["value"].iloc[1] == pytest.approx(1.0)
    assert result["value"].iloc[2] == pytest.approx(1.0)


def test_distribute_full_window_matches_partial_window_per_slot_value() -> None:
    """Distributing a daily value over the full day gives the same per-slot value
    as distributing it over a partial window — both divide by 288."""
    start = dt.datetime(2025, 1, 1, 0, 0, tzinfo=dt.UTC)
    end_full = dt.datetime(2025, 1, 2, 0, 0, tzinfo=dt.UTC)
    end_partial = dt.datetime(2025, 1, 1, 0, 15, tzinfo=dt.UTC)

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=1, freq="1D", tz=dt.UTC),
            "value": [288.0],
        }
    )

    full = redistribute_to_resolution(df, dt.timedelta(days=1), dt.timedelta(minutes=5), start, end_full)
    partial = redistribute_to_resolution(df, dt.timedelta(days=1), dt.timedelta(minutes=5), start, end_partial)

    assert full["value"].iloc[0] == pytest.approx(partial["value"].iloc[0])
    assert len(full) == 288
    assert len(partial) == 3


def test_distribute_aligned_window_divides_by_slots_in_source_period() -> None:
    """15-min source value distributed into 5-min slots: divides by 3 (slots per 15-min period)."""
    start = dt.datetime(2025, 1, 1, 0, 0, tzinfo=dt.UTC)
    end = dt.datetime(2025, 1, 1, 1, 0, tzinfo=dt.UTC)

    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=4, freq="15min", tz=dt.UTC),
            "value": [30.0, 30.0, 30.0, 30.0],
        }
    )

    result = redistribute_to_resolution(df, dt.timedelta(minutes=15), dt.timedelta(minutes=5), start, end)

    assert len(result) == 12
    assert result["value"].iloc[0] == pytest.approx(10.0)  # 30 / 3
    assert result["value"].sum() == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# _find_common_divisor
# ---------------------------------------------------------------------------


def test_find_common_divisor_returns_gcd_for_two_timedeltas() -> None:
    assert _find_common_divisor(dt.timedelta(hours=2), dt.timedelta(hours=3)) == dt.timedelta(hours=1)
    assert _find_common_divisor(dt.timedelta(hours=1), dt.timedelta(minutes=15)) == dt.timedelta(minutes=15)
    assert _find_common_divisor(dt.timedelta(days=1), dt.timedelta(hours=1)) == dt.timedelta(hours=1)
    assert _find_common_divisor(dt.timedelta(hours=1), dt.timedelta(minutes=1)) == dt.timedelta(minutes=1)


def test_find_common_divisor_returns_gcd_for_two_calendar_durations() -> None:
    assert _find_common_divisor(isodate.parse_duration("P1Y"), isodate.parse_duration("P1M")) == isodate.Duration(
        months=1
    )
    assert _find_common_divisor(isodate.parse_duration("P3M"), isodate.parse_duration("P2M")) == isodate.Duration(
        months=1
    )
    assert _find_common_divisor(isodate.parse_duration("P6M"), isodate.parse_duration("P4M")) == isodate.Duration(
        months=2
    )


def test_find_common_divisor_returns_timedelta_for_mixed_types_when_timedelta_divides_calendar() -> None:
    assert _find_common_divisor(isodate.parse_duration("P1M"), dt.timedelta(days=1)) == dt.timedelta(days=1)
    assert _find_common_divisor(isodate.parse_duration("P1M"), dt.timedelta(hours=1)) == dt.timedelta(hours=1)
    assert _find_common_divisor(isodate.parse_duration("P1Y"), dt.timedelta(hours=1)) == dt.timedelta(hours=1)
    assert _find_common_divisor(isodate.parse_duration("P1M"), dt.timedelta(minutes=15)) == dt.timedelta(minutes=15)


def test_find_common_divisor_mixed_uses_gcd_with_one_day() -> None:
    # Mixed calendar + timedelta: common divisor is gcd(timedelta, P1D).
    # P7D and P1D share P1D as GCD, so P1M + P7D → P1D.
    assert _find_common_divisor(isodate.parse_duration("P1M"), dt.timedelta(weeks=1)) == dt.timedelta(days=1)
    # P8H and P1D share P8H as GCD, so P1M + P8H → P8H.
    assert _find_common_divisor(isodate.parse_duration("P1M"), dt.timedelta(hours=8)) == dt.timedelta(hours=8)
    # P25H and P1D: gcd(90000s, 86400s) = 3600s = P1H.
    assert _find_common_divisor(dt.timedelta(hours=25), dt.timedelta(hours=24)) == dt.timedelta(hours=1)


# ---------------------------------------------------------------------------
# redistribute_to_resolution: cases A / B / C / D
# ---------------------------------------------------------------------------


def test_redistribute_to_resolution_case_a_distribute_month_to_day() -> None:
    """Case A: source (P1M) is coarser than output (P1D) → distribute."""
    start = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
    end = dt.datetime(2025, 2, 1, tzinfo=dt.UTC)
    df = pd.DataFrame({"timestamp": [pd.Timestamp("2025-01-01", tz=dt.UTC)], "value": [310.0]})
    result = redistribute_to_resolution(df, isodate.parse_duration("P1M"), dt.timedelta(days=1), start, end)
    assert len(result) == 31
    assert result["value"].sum() == pytest.approx(310.0)
    assert result["value"].iloc[0] == pytest.approx(310.0 / 31)


def test_redistribute_to_resolution_case_a_distribute_day_to_hour() -> None:
    """Case A: source (P1D) is coarser than output (P1H) → distribute."""
    start = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
    end = dt.datetime(2025, 1, 2, tzinfo=dt.UTC)
    df = pd.DataFrame({"timestamp": [pd.Timestamp("2025-01-01", tz=dt.UTC)], "value": [24.0]})
    result = redistribute_to_resolution(df, dt.timedelta(days=1), dt.timedelta(hours=1), start, end)
    assert len(result) == 24
    assert result["value"].sum() == pytest.approx(24.0)
    assert result["value"].tolist() == pytest.approx([1.0] * 24)


def test_redistribute_to_resolution_case_b_aggregate_15min_to_day() -> None:
    """Case B: source (PT15M) is finer than output (P1D) → sum."""
    start = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
    end = dt.datetime(2025, 1, 2, tzinfo=dt.UTC)
    timestamps = pd.date_range(start, end, freq="15min", inclusive="left")
    df = pd.DataFrame({"timestamp": timestamps, "value": [1.0] * len(timestamps)})
    result = redistribute_to_resolution(df, dt.timedelta(minutes=15), dt.timedelta(days=1), start, end)
    assert len(result) == 1
    assert result["value"].iloc[0] == pytest.approx(96.0)  # 4 * 24 slots


def test_redistribute_to_resolution_case_b_aggregate_hour_to_month() -> None:
    """Case B: source (P1H) is finer than output (P1M) → sum."""
    start = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
    end = dt.datetime(2025, 2, 1, tzinfo=dt.UTC)
    timestamps = pd.date_range(start, end, freq="1h", inclusive="left")
    df = pd.DataFrame({"timestamp": timestamps, "value": [1.0] * len(timestamps)})
    result = redistribute_to_resolution(df, dt.timedelta(hours=1), isodate.parse_duration("P1M"), start, end)
    assert len(result) == 1
    assert result["value"].iloc[0] == pytest.approx(31 * 24)


def test_redistribute_to_resolution_case_c_intermediate_gcd() -> None:
    """Case C: source (P2H) and output (P3H) share GCD (P1H) → distribute then aggregate."""
    start = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
    end = dt.datetime(2025, 1, 1, 6, tzinfo=dt.UTC)  # 3 x P2H slots = 2 x P3H slots
    timestamps = pd.date_range(start, end, freq="2h", inclusive="left")
    df = pd.DataFrame({"timestamp": timestamps, "value": [60.0, 60.0, 60.0]})
    result = redistribute_to_resolution(df, dt.timedelta(hours=2), dt.timedelta(hours=3), start, end)
    assert len(result) == 2
    # Each P2H value 60 distributes to 2 x P1H slots of 30 each; 3 x P1H slots sum to 90.
    assert result["value"].sum() == pytest.approx(180.0)
    assert result["value"].iloc[0] == pytest.approx(90.0)
    assert result["value"].iloc[1] == pytest.approx(90.0)


def test_redistribute_to_resolution_case_c_month_to_week_via_day() -> None:
    """Case C: P1M → P7D has no direct divisor relationship, but P1D is a common divisor.
    The function distributes P1M → P1D first, then aggregates P1D → P7D.
    January has 31 days: weeks receive 7, 7, 7, 7 and 3 daily shares.
    """
    start = dt.datetime(2025, 1, 1, tzinfo=dt.UTC)
    end = dt.datetime(2025, 2, 1, tzinfo=dt.UTC)
    df = pd.DataFrame({"timestamp": [pd.Timestamp("2025-01-01", tz=dt.UTC)], "value": [310.0]})
    result = redistribute_to_resolution(df, isodate.parse_duration("P1M"), dt.timedelta(weeks=1), start, end)
    assert len(result) == 5
    assert result["value"].sum() == pytest.approx(310.0)
    # Jan 1–7 (7 days), Jan 8–14 (7), Jan 15–21 (7), Jan 22–28 (7), Jan 29–31 (3)
    expected = [310.0 * n / 31 for n in (7, 7, 7, 7, 3)]
    assert result["value"].tolist() == pytest.approx(expected)


# ---------------------------------------------------------------------------
# snap_billing_period: anchor-based snapping
# ---------------------------------------------------------------------------


def test_snap_billing_period_7d_default_anchor_uses_billing_start() -> None:
    """Without explicit anchor, snap points derive from billing_start (normalised to midnight)."""
    billing_start = dt.datetime(2026, 2, 1, tzinfo=dt.UTC)  # Sunday
    billing_end = dt.datetime(2026, 2, 22, tzinfo=dt.UTC)
    snapped_start, snapped_end = snap_billing_period(billing_start, billing_end, "7D")
    assert snapped_start == pd.Timestamp(billing_start)
    assert snapped_end >= pd.Timestamp(billing_end)
    delta = pd.Timedelta(snapped_end - pd.Timestamp(billing_start))
    assert delta.value % pd.Timedelta("7D").value == 0


def test_snap_billing_period_7d_explicit_anchor() -> None:
    """With an explicit anchor, snap points are anchor_midnight + x*7D."""
    anchor = dt.datetime(2026, 1, 5, tzinfo=dt.UTC)  # Monday
    billing_start = dt.datetime(2026, 2, 1, tzinfo=dt.UTC)  # Sunday
    billing_end = dt.datetime(2026, 2, 22, tzinfo=dt.UTC)
    snapped_start, snapped_end = snap_billing_period(billing_start, billing_end, "7D", anchor=anchor)
    anchor_norm = cast(pd.Timestamp, pd.Timestamp(anchor)).normalize()
    # snapped_start must be <= billing_start and on the anchor grid
    assert snapped_start <= pd.Timestamp(billing_start)
    assert pd.Timedelta(snapped_start - anchor_norm).value % pd.Timedelta("7D").value == 0
    # snapped_end must be >= billing_end and on the anchor grid
    assert snapped_end >= pd.Timestamp(billing_end)
    assert pd.Timedelta(snapped_end - anchor_norm).value % pd.Timedelta("7D").value == 0


def test_snap_billing_period_7d_no_shift_when_billing_start_is_on_anchor_grid() -> None:
    """When billing_start falls exactly on the anchor grid, snapped_start == billing_start."""
    billing_start = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)  # Thursday
    billing_end = dt.datetime(2026, 1, 29, tzinfo=dt.UTC)
    snapped_start, snapped_end = snap_billing_period(billing_start, billing_end, "7D")
    assert snapped_start == pd.Timestamp(billing_start)
    assert snapped_end >= pd.Timestamp(billing_end)


def test_snap_billing_period_consistent_across_different_starts_with_same_anchor() -> None:
    """Two different billing windows with the same anchor must produce snap points on the same grid."""
    anchor = dt.datetime(2026, 1, 1, tzinfo=dt.UTC)
    s1, e1 = snap_billing_period(
        dt.datetime(2026, 1, 5, tzinfo=dt.UTC),
        dt.datetime(2026, 1, 20, tzinfo=dt.UTC),
        "7D",
        anchor=anchor,
    )
    s2, e2 = snap_billing_period(
        dt.datetime(2026, 2, 3, tzinfo=dt.UTC),
        dt.datetime(2026, 2, 18, tzinfo=dt.UTC),
        "7D",
        anchor=anchor,
    )
    anchor_norm = cast(pd.Timestamp, pd.Timestamp(anchor)).normalize()
    # Both snapped_starts must be on the same grid
    assert pd.Timedelta(s1 - anchor_norm).value % pd.Timedelta("7D").value == 0
    assert pd.Timedelta(s2 - anchor_norm).value % pd.Timedelta("7D").value == 0


def test_redistribute_to_resolution_7d_bins_start_at_billing_start_not_epoch_thursday() -> None:
    """Aggregating hourly data to P7D must produce bins aligned to billing_start, not
    to the Unix epoch's Thursday, even when billing_start falls on another weekday."""
    billing_start = dt.datetime(2026, 2, 2, tzinfo=dt.UTC)
    billing_end = dt.datetime(2026, 2, 16, tzinfo=dt.UTC)
    timestamps = pd.date_range(billing_start, billing_end, freq="1h", inclusive="left")
    df = pd.DataFrame({"timestamp": timestamps, "value": [1.0] * len(timestamps)})

    result = redistribute_to_resolution(df, dt.timedelta(hours=1), dt.timedelta(weeks=1), billing_start, billing_end)

    assert len(result) == 2
    assert result["timestamp"].iloc[0] == pd.Timestamp(billing_start)
    assert result["timestamp"].iloc[1] == pd.Timestamp(billing_start) + pd.Timedelta("7D")
    assert result["value"].sum() == pytest.approx(14 * 24)
