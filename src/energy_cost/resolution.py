import datetime as dt
from typing import Annotated, cast

import isodate
import pandas as pd
from pydantic import BeforeValidator


def align_datetime_to_tz(d: dt.datetime, tz: dt.tzinfo | None) -> dt.datetime:
    if tz is None:
        return d.replace(tzinfo=None) if d.tzinfo is not None else d
    if d.tzinfo is None:
        return cast(
            dt.datetime,
            pd.Timestamp(d).tz_localize(tz, ambiguous=False, nonexistent="shift_forward").to_pydatetime(),
        )
    return cast(dt.datetime, pd.Timestamp(d).tz_convert(tz).to_pydatetime())


def align_timestamps_to_tz(data: pd.DataFrame, tz: dt.tzinfo) -> pd.DataFrame:
    """Return a copy of *data* with the ``"timestamp"`` column converted/localized to *tz*."""
    data = data.copy()
    col = data["timestamp"]
    # Coerce object columns (mixed-offset tz-aware datetimes) to a uniform datetime64
    if col.dtype == object:
        col = pd.to_datetime(col, utc=True)
    if col.dt.tz is not None:
        data["timestamp"] = col.dt.tz_convert(tz)
    else:
        data["timestamp"] = col.dt.tz_localize(tz, ambiguous=False, nonexistent="shift_forward")
    return data


def parse_resolution(value: dt.timedelta | isodate.Duration | str) -> dt.timedelta | isodate.Duration:
    if isinstance(value, str):
        return isodate.parse_duration(value)
    return value


# A resolution is either a fixed timedelta or a calendar-aware Duration.
Resolution = Annotated[dt.timedelta | isodate.Duration, BeforeValidator(parse_resolution)]


def validate_non_mixed_duration(resolution: Resolution) -> None:
    if isinstance(resolution, isodate.Duration) and (
        (resolution.years and resolution.months)
        or (resolution.years and resolution.tdelta)
        or (resolution.months and resolution.tdelta)
    ):
        raise ValueError(f"Mixed durations with multiple components are not supported as resolutions: {resolution}")


def to_pandas_freq(resolution: Resolution) -> str:
    """Convert a resolution to a pandas frequency string."""
    validate_non_mixed_duration(resolution)
    if isinstance(resolution, isodate.Duration):
        if resolution.years:
            return f"{int(resolution.years)}YS"
        if resolution.months:
            return f"{int(resolution.months)}MS"
    # Plain timedelta — express in the largest clean unit to keep freq strings readable
    total_seconds = int(resolution.total_seconds())
    if total_seconds % 86400 == 0:
        return f"{total_seconds // 86400}D"
    if total_seconds % 3600 == 0:
        return f"{total_seconds // 3600}h"
    if total_seconds % 60 == 0:
        return f"{total_seconds // 60}min"
    return f"{total_seconds}s"


def to_pandas_offset(resolution: Resolution) -> pd.tseries.offsets.BaseOffset:
    return pd.tseries.frequencies.to_offset(to_pandas_freq(resolution))


def is_divisor(root: Resolution, divisor: Resolution) -> bool:
    validate_non_mixed_duration(root)
    validate_non_mixed_duration(divisor)
    root_is_calendar = isinstance(root, isodate.Duration) and (root.years or root.months)
    divisor_is_calendar = isinstance(divisor, isodate.Duration) and (divisor.years or divisor.months)

    if not root_is_calendar and not divisor_is_calendar:
        root_s = int(root.total_seconds())
        divisor_s = int(divisor.total_seconds())
        return divisor_s > 0 and root_s % divisor_s == 0

    if root_is_calendar and not divisor_is_calendar:
        # a timedelta is a divisor of a calendar duration if it is a divisor of 1 day
        return is_divisor(isodate.parse_duration("P1D"), divisor)

    if root_is_calendar and divisor_is_calendar:
        # Normalise both to months
        root_dur = cast(isodate.Duration, root)
        divisor_dur = cast(isodate.Duration, divisor)
        root_months = int(root_dur.years * 12 + root_dur.months)
        divisor_months = int(divisor_dur.years * 12 + divisor_dur.months)
        return divisor_months > 0 and root_months % divisor_months == 0

    # divisor is calendar, root is fixed timedelta — nonsensical subdivision
    return False


def detect_resolution(timestamps: pd.Series) -> Resolution:
    """
    Infer the resolution from a sorted timestamp series.

    Strategy
    --------
    1. Check for yearly alignment: all timestamps on Jan 1, gaps are multiples of 1 year
    2. Check for monthly alignment: all timestamps on the 1st, gaps are multiples of 1 month
    3. Fall back to timedelta mode (works correctly for fixed-period data like 15min)
    """
    if len(timestamps) < 2:
        raise ValueError("Cannot detect resolution from fewer than 2 timestamps.")

    all_month_starts = (timestamps.dt.day == 1).all() and (timestamps.dt.hour == 0).all()

    if all_month_starts:
        all_year_starts = (timestamps.dt.month == 1).all()

        if all_year_starts:
            year_gaps = timestamps.dt.year.diff().dropna()
            if (year_gaps > 0).all() and (year_gaps == year_gaps.iloc[0]).all():
                return isodate.Duration(years=int(year_gaps.iloc[0]))

        month_numbers = timestamps.dt.year * 12 + timestamps.dt.month
        month_gaps = month_numbers.diff().dropna()
        if (month_gaps > 0).all() and (month_gaps == month_gaps.iloc[0]).all():
            return isodate.Duration(months=int(month_gaps.iloc[0]))

    deltas = timestamps.diff().dropna()
    mode_delta = deltas.mode()[0]
    return pd.to_timedelta(mode_delta)


def snap_billing_period(
    billing_start: dt.datetime,
    billing_end: dt.datetime,
    output_freq: str,
) -> tuple[dt.datetime, dt.datetime]:
    """Snap billing start/end to clean output-period boundaries (floor start, ceil end).

    For example, with monthly output and data starting on 2026-04-09, billing_start
    is snapped back to 2026-04-01 and billing_end forward to the next month start.
    """
    offset = pd.tseries.frequencies.to_offset(output_freq)
    assert offset is not None

    if not isinstance(offset, pd.tseries.offsets.Tick):
        # Calendar offsets (MonthBegin, YearBegin, …) consider any day-1 timestamp
        # "on-offset" regardless of time-of-day, so rollforward/rollback preserve
        # non-midnight times. Normalize to midnight first (pd.Timestamp.normalize()
        # handles DST-aware timezones correctly), then handle the ceiling case: if
        # the original ts_end was after midnight on a day that rollforward considers
        # already on-offset, advance one extra period.
        snapped_start = cast(pd.Timestamp, pd.Timestamp(offset.rollback(billing_start))).normalize()
        end_norm = cast(pd.Timestamp, pd.Timestamp(billing_end)).normalize()
        snapped_end = offset.rollforward(end_norm)
        if end_norm < billing_end and snapped_end == end_norm:
            snapped_end = snapped_end + offset
    else:
        snapped_start = offset.rollback(billing_start)
        snapped_end = offset.rollforward(billing_end)

    return snapped_start, snapped_end


def resample_or_distribute(
    df: pd.DataFrame,
    source_resolution: Resolution,
    output_resolution: Resolution,
    start: dt.datetime,
    end: dt.datetime,
) -> pd.DataFrame:
    output_freq = to_pandas_freq(output_resolution)
    source_freq = to_pandas_freq(source_resolution)

    if source_freq == output_freq or not is_divisor(source_resolution, output_resolution):
        return df.set_index("timestamp").resample(output_freq).sum().reset_index()

    # Snap to full source periods so every coarse slot is fully populated in the target index.
    snapped_start, snapped_end = snap_billing_period(start, end, source_freq)

    # Build fine-grained target index over the full snapped range.
    target_df = pd.DataFrame(
        {
            "timestamp": pd.date_range(start=snapped_start, end=snapped_end, freq=output_freq, inclusive="left"),
        }
    )

    # Backward merge: each fine slot inherits the coarse value and its coarse timestamp.
    src = df.rename(columns={"timestamp": "__coarse_ts"})
    merged = pd.merge_asof(target_df, src, left_on="timestamp", right_on="__coarse_ts", direction="backward")

    # Each full source period is now fully populated, so count == total fine slots in that period.
    merged["__n"] = merged.groupby("__coarse_ts")["timestamp"].transform("count")

    value_cols = [c for c in merged.columns if c not in ("timestamp", "__coarse_ts", "__n")]
    merged[value_cols] = merged[value_cols].div(merged["__n"], axis=0)

    # Trim to the requested window and clean up helper columns.
    result = merged[(merged["timestamp"] >= start) & (merged["timestamp"] < end)]
    return result.drop(columns=["__coarse_ts", "__n"]).reset_index(drop=True)


def detect_resolution_and_range(
    data: pd.DataFrame,
    resolution: Resolution | None = None,
) -> tuple[dt.datetime, dt.datetime, Resolution]:
    if resolution is None:
        if len(data) < 2:
            raise ValueError("A resolution is required when applying a formula to fewer than 2 timestamps.")
        resolution = detect_resolution(data["timestamp"])
    start = data["timestamp"].min()
    end = data["timestamp"].max() + to_pandas_offset(resolution)
    return start, end, resolution
