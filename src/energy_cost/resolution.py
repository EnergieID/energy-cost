import datetime as dt
import math
from typing import Annotated, cast

import isodate
import pandas as pd
from pydantic import BeforeValidator, PlainSerializer, WithJsonSchema


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
    """Return *data* with the ``"timestamp"`` column converted/localized to *tz*."""
    col = data["timestamp"]
    # Coerce object columns (mixed-offset tz-aware datetimes) to a uniform datetime64
    if col.dtype == object:
        col = pd.to_datetime(col, utc=True)
    if col.dt.tz is not None:
        # Use str() comparison: pytz.UTC and datetime.timezone.utc both stringify to "UTC"
        if str(col.dt.tz) == str(tz):
            return data
        new_col = col.dt.tz_convert(tz)
    else:
        new_col = col.dt.tz_localize(tz, ambiguous=False, nonexistent="shift_forward")
    return data.assign(timestamp=new_col)


def parse_resolution(value: dt.timedelta | isodate.Duration | str) -> dt.timedelta | isodate.Duration:
    if isinstance(value, str):
        return isodate.parse_duration(value)
    return value


# A resolution is either a fixed timedelta or a calendar-aware Duration.
Resolution = Annotated[
    dt.timedelta | isodate.Duration,
    BeforeValidator(parse_resolution),
    PlainSerializer(isodate.duration_isoformat, return_type=str),
    WithJsonSchema({"type": "string", "examples": ["PT15M", "P1M"]}),
]


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
        root_months = int(root.years * 12 + root.months)
        divisor_months = int(divisor.years * 12 + divisor.months)
        return divisor_months > 0 and root_months % divisor_months == 0

    # divisor is calendar, root is fixed timedelta — nonsensical subdivision
    return False


def detect_resolution(timestamps: pd.Series) -> Resolution:
    """Infer the resolution from the first two timestamps."""
    if len(timestamps) < 2:
        raise ValueError("Cannot detect resolution from fewer than 2 timestamps.")

    t0, t1 = timestamps.iloc[0], timestamps.iloc[1]

    if t0.day == 1 and t1.day == 1:
        month_gap = (t1.year - t0.year) * 12 + (t1.month - t0.month)
        if month_gap > 0:
            if month_gap % 12 == 0:
                return isodate.Duration(years=month_gap // 12)
            return isodate.Duration(months=month_gap)

    return pd.to_timedelta(t1 - t0)


def snap_billing_period(
    billing_start: dt.datetime,
    billing_end: dt.datetime,
    output_freq: str,
    anchor: dt.datetime | None = None,
) -> tuple[dt.datetime, dt.datetime]:
    """Snap billing start/end outward to output-period boundaries derived from *anchor*.

    The snap points form the series ``anchor_norm + x * offset`` where *anchor_norm* is
    *anchor* normalised to midnight (and for calendar offsets rolled back to the 1st of
    the month/year).  When *anchor* is ``None`` it defaults to *billing_start*.

    Returns ``(latest snap <= billing_start, earliest snap >= billing_end)``.
    """
    if anchor is None:
        anchor = billing_start

    offset = pd.tseries.frequencies.to_offset(output_freq)
    assert offset is not None

    ts_anchor = cast(pd.Timestamp, pd.Timestamp(anchor))

    try:
        tick_nanos = offset.nanos
    except ValueError:
        tick_nanos = None

    if tick_nanos is None:
        # Calendar offsets (MonthBegin, YearBegin, …): the anchor is normalised to
        # midnight on the 1st of the month/year so the grid is always the natural
        # calendar boundary.  Different anchors therefore always produce the same grid.
        snapped_start = cast(pd.Timestamp, pd.Timestamp(offset.rollback(billing_start))).normalize()

        end_norm = cast(pd.Timestamp, pd.Timestamp(billing_end)).normalize()
        snapped_end = offset.rollforward(end_norm)
        if end_norm < billing_end and snapped_end == end_norm:
            snapped_end = snapped_end + offset
    else:
        # Fixed-duration offsets (Day, Hour, Minute, …): normalise anchor to midnight
        # and compute snap points as anchor_norm + n * offset.
        anchor_norm = ts_anchor.normalize()
        ts_start = cast(pd.Timestamp, pd.Timestamp(billing_start))
        ts_end = cast(pd.Timestamp, pd.Timestamp(billing_end))

        # Latest snap point <= billing_start  (floor division from anchor_norm)
        delta_start = (ts_start - anchor_norm).value
        n_start = delta_start // tick_nanos if delta_start >= 0 else -((-delta_start - 1) // tick_nanos + 1)
        snapped_start = anchor_norm + n_start * offset

        # Earliest snap point >= billing_end  (ceiling division from anchor_norm)
        delta_end = (ts_end - anchor_norm).value
        n_end = -(-delta_end // tick_nanos)  # ceiling division
        snapped_end = anchor_norm + n_end * offset

    return snapped_start, snapped_end


def find_common_divisor(a: Resolution, b: Resolution) -> Resolution:
    """Return a common divisor of *a* and *b*, i.e. the coarsest resolution C such that
    C divides both *a* and *b* (both are integer multiples of C).
    """
    a_is_cal = isinstance(a, isodate.Duration) and (a.years or a.months)
    b_is_cal = isinstance(b, isodate.Duration) and (b.years or b.months)

    if not a_is_cal and not b_is_cal:
        a_s = int(a.total_seconds())
        b_s = int(b.total_seconds())
        g = math.gcd(a_s, b_s)
        return dt.timedelta(seconds=g)

    if a_is_cal and b_is_cal:
        a_months = int(a.years * 12 + a.months)
        b_months = int(b.years * 12 + b.months)
        g = math.gcd(a_months, b_months)
        return isodate.Duration(months=g)

    # Mixed: one calendar, one timedelta. 1 day is a divisor of every calendar duration
    # So a common divisor between P1D and the timedelta also divides the calendar duration.
    td = b if a_is_cal else a
    return find_common_divisor(td, isodate.parse_duration("P1D"))


def _distribute(
    df: pd.DataFrame,
    source_resolution: Resolution,
    output_resolution: Resolution,
    start: dt.datetime,
    end: dt.datetime,
    binning_anchor: dt.datetime | None = None,
) -> pd.DataFrame:
    """Distribute coarse *source_resolution* values proportionally into finer *output_resolution* bins."""
    source_freq = to_pandas_freq(source_resolution)
    output_freq = to_pandas_freq(output_resolution)

    # Snap to full source periods so every coarse slot is fully populated in the target index.
    snapped_start, snapped_end = snap_billing_period(start, end, source_freq, anchor=binning_anchor)

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


def redistribute_to_resolution(
    df: pd.DataFrame,
    source_resolution: Resolution,
    output_resolution: Resolution,
    start: dt.datetime,
    end: dt.datetime,
    binning_anchor: dt.datetime | None = None,
) -> pd.DataFrame:
    """Convert *df* from *source_resolution* to *output_resolution*."""

    if source_resolution == output_resolution:
        # Same resolution — just filter to [start, end)
        filtered = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
        return filtered.reset_index(drop=True)

    gcd = find_common_divisor(source_resolution, output_resolution)
    result = df

    if gcd != source_resolution:
        result = _distribute(result, source_resolution, gcd, start, end, binning_anchor=binning_anchor)

    if gcd != output_resolution:
        result = _aggregate_to_resolution(result, output_resolution, start, end, binning_anchor=binning_anchor)

    return result


def _aggregate_to_resolution(
    df: pd.DataFrame,
    output_resolution: Resolution,
    start: dt.datetime,
    end: dt.datetime,
    binning_anchor: dt.datetime | None = None,
) -> pd.DataFrame:
    """Aggregate *df* into *output_resolution* bins anchored to *start*."""
    freq = to_pandas_freq(output_resolution)
    snapped_start, snapped_end = snap_billing_period(start, end, freq, anchor=binning_anchor)
    bin_df = pd.DataFrame({"__bin": pd.date_range(start=snapped_start, end=snapped_end, freq=freq, inclusive="left")})
    # Filter to [start, end) — rows outside are out-of-scope and should not affect bin sums
    df = df[(df["timestamp"] >= start) & (df["timestamp"] < end)]
    merged = pd.merge_asof(df, bin_df, left_on="timestamp", right_on="__bin", direction="backward")
    value_cols = [c for c in merged.columns if c not in ("timestamp", "__bin")]
    agg = merged.groupby("__bin")[value_cols].sum()
    # Mask bins that contain any NaN back to NaN (skipna=False semantics)
    has_nan = merged.set_index("__bin")[value_cols].isna().groupby(level=0).any()
    agg = agg.where(~has_nan)
    # Reindex to all expected bins — bins without any data become NaN
    agg = agg.reindex(bin_df["__bin"])
    return agg.reset_index().rename(columns={"__bin": "timestamp"})


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
