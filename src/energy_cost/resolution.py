import datetime as dt

import isodate
import pandas as pd

# A resolution is either a fixed timedelta or a calendar-aware Duration.
Resolution = dt.timedelta | isodate.Duration


def to_pandas_freq(resolution: Resolution) -> str:
    """Convert a resolution to a pandas frequency string."""
    if isinstance(resolution, isodate.Duration):
        if resolution.years and not resolution.months and not resolution.tdelta:
            return f"{int(resolution.years)}YS"
        if resolution.months and not resolution.years and not resolution.tdelta:
            return f"{int(resolution.months)}MS"
        # Mixed Duration (e.g. P1Y6M) — not supported as a regular frequency
        raise ValueError(f"Cannot convert mixed isodate.Duration to pandas freq: {resolution}")
    # Plain timedelta — express in the largest clean unit to keep freq strings readable
    total_seconds = int(resolution.total_seconds())
    if total_seconds % 3600 == 0:
        return f"{total_seconds // 3600}h"
    if total_seconds % 60 == 0:
        return f"{total_seconds // 60}min"
    return f"{total_seconds}s"


def to_pandas_offset(resolution: Resolution) -> pd.tseries.offsets.BaseOffset:
    return pd.tseries.frequencies.to_offset(to_pandas_freq(resolution))


def resolution_divides(source: Resolution, requested: Resolution) -> bool:
    """
    Return True if *requested* is a valid subdivision of *source*.

    Rules
    -----
    - timedelta / timedelta  : requested seconds must evenly divide source seconds.
    - Duration(months) / timedelta : any fixed timedelta always subdivides a
      calendar period, because calendar period boundaries land on whole seconds.
    - Duration / Duration    : the requested calendar period must evenly divide
      the source calendar period (years are normalised to months).
    - timedelta / Duration   : a fixed window cannot subdivide a calendar period
      that is larger than it — reject.
    """
    src_is_calendar = isinstance(source, isodate.Duration) and (source.years or source.months)
    req_is_calendar = isinstance(requested, isodate.Duration) and (requested.years or requested.months)

    if not src_is_calendar and not req_is_calendar:
        src_s = int(source.total_seconds())
        req_s = int(requested.total_seconds())
        return req_s > 0 and src_s % req_s == 0

    if src_is_calendar and not req_is_calendar:
        # Fixed timedelta subdivides any calendar period — always valid.
        req_s = int(requested.total_seconds())
        return req_s > 0

    if src_is_calendar and req_is_calendar:
        # Normalise both to months
        src_months = int(source.years * 12 + source.months)  # type: ignore[union-attr]
        req_months = int(requested.years * 12 + requested.months)  # type: ignore[union-attr]
        return req_months > 0 and src_months % req_months == 0

    # requested is calendar, source is fixed timedelta — nonsensical subdivision
    return False
