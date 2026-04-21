from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

import pandas as pd

from .entsoe_day_ahead_index import EntsoeDayAheadIndex
from .index import Index

_log = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path(__file__).parents[3] / ".cache" / "entsoe"

_OLD_DATA_THRESHOLD = dt.timedelta(days=7)
_REFRESH_INTERVAL = dt.timedelta(hours=1)

_COLUMNS = ["timestamp", "value", "fetch_time", "stable"]


class CachedEntsoeDayAheadIndex(Index):
    """ENTSO-E day-ahead index with transparent disk caching.

    Parameters
    ----------
    country_code:
        ENTSO-E area code, e.g. ``"BE"``.
    api_key:
        ENTSO-E transparency platform API key.
    resolution:
        Native resolution of the underlying index (default: 15 min).
    cache_dir:
        Root directory for cache files.  Defaults to ``<repo_root>/.cache/entsoe``.
    old_threshold:
        How far in the past data must be before it is considered immutable
        (default: 7 days).
    refresh_interval:
        Minimum time between re-fetches of recent/future data (default: 1 hour).
    """

    def __init__(
        self,
        country_code: str,
        api_key: str,
        resolution: dt.timedelta = dt.timedelta(minutes=15),
        cache_dir: Path | str | None = None,
        old_threshold: dt.timedelta = _OLD_DATA_THRESHOLD,
        refresh_interval: dt.timedelta = _REFRESH_INTERVAL,
    ) -> None:
        self._source = EntsoeDayAheadIndex(country_code=country_code, api_key=api_key, resolution=resolution)
        self.country_code = country_code
        self.cache_dir = Path(cache_dir) if cache_dir is not None else _DEFAULT_CACHE_DIR
        self.old_threshold = old_threshold
        self.refresh_interval = refresh_interval
        super().__init__(resolution=resolution)

    def _csv_path(self) -> Path:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"{self.country_code}.csv"

    def _load_cache(self) -> pd.DataFrame:
        """Load the cache and re-evaluate the ``stable`` flag based on current time."""
        path = self._csv_path()
        if not path.exists():
            return pd.DataFrame(columns=_COLUMNS)

        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.as_unit("us")
        df["fetch_time"] = pd.to_datetime(df["fetch_time"], utc=True).dt.as_unit("us")
        df["stable"] = df["stable"].astype(bool)

        return df

    def _save_cache(self, df: pd.DataFrame) -> None:
        path = self._csv_path()
        tmp = path.with_suffix(".tmp")
        df.to_csv(tmp, index=False)
        tmp.replace(path)

    def _compute_fetch_range(
        self,
        cache: pd.DataFrame,
        start_utc: pd.Timestamp,
        end_utc: pd.Timestamp,
        now: pd.Timestamp,
    ) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        """Return ``(fetch_start, fetch_end)`` if a network request is needed, else ``None``."""
        if cache.empty:
            return start_utc, end_utc

        starts: list[pd.Timestamp] = []
        ends: list[pd.Timestamp] = []

        # Gap before cached data
        if start_utc + self.resolution < cache["timestamp"].min():
            starts.append(start_utc)
            ends.append(cache["timestamp"].min())

        # Gap after cached data (continuity check)
        if end_utc > cache["timestamp"].max() + self.resolution:
            starts.append(cache["timestamp"].max() + self.resolution)
            ends.append(end_utc)

        # Unstable rows in the requested range with a stale fetch_time
        in_range = cache[(~cache["stable"]) & (cache["timestamp"] >= start_utc) & (cache["timestamp"] < end_utc)]
        if not in_range.empty:
            oldest_fetch = pd.Timestamp(in_range["fetch_time"].min())
            if (now - oldest_fetch) > pd.Timedelta(self.refresh_interval):
                starts.append(in_range["timestamp"].min())
                ends.append(in_range["timestamp"].max() + self.resolution)
        if not starts:
            return None

        return min(starts), max(ends)

    # ── Fetch and merge ───────────────────────────────────────────────────────

    def _fetch_and_merge(
        self,
        cache: pd.DataFrame,
        fetch_start: pd.Timestamp,
        fetch_end: pd.Timestamp,
        now: pd.Timestamp,
    ) -> pd.DataFrame:
        _log.debug("Fetching ENTSOE %s [%s, %s) ...", self.country_code, fetch_start, fetch_end)
        try:
            raw = self._source.get_values(fetch_start, fetch_end, self.resolution, dt.UTC)
        except Exception:
            _log.warning(
                "Failed to fetch ENTSOE %s [%s, %s)",
                self.country_code,
                fetch_start,
                fetch_end,
                exc_info=True,
            )
            return cache

        if raw.empty:
            return cache

        raw = raw.dropna(subset=["value"])
        if raw.empty:
            return cache

        raw["timestamp"] = pd.to_datetime(raw["timestamp"], utc=True)
        raw["fetch_time"] = now
        raw["stable"] = (raw["timestamp"] + pd.Timedelta(self.old_threshold)) < now

        if cache.empty:
            return raw

        return (
            pd.concat([cache, raw], ignore_index=True)
            .drop_duplicates(subset="timestamp", keep="last")
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

    def _get_values(self, start: pd.Timestamp, end: pd.Timestamp, timezone: dt.tzinfo) -> pd.DataFrame:
        now = pd.Timestamp.now(tz="UTC")
        start_utc = start.tz_convert("UTC") if start.tzinfo else start.tz_localize("UTC")
        end_utc = end.tz_convert("UTC") if end.tzinfo else end.tz_localize("UTC")

        cache = self._load_cache()
        fetch_range = self._compute_fetch_range(cache, start_utc, end_utc, now)

        if fetch_range is not None:
            fetch_start, fetch_end = fetch_range
            cache = self._fetch_and_merge(cache, fetch_start, fetch_end, now)
            self._save_cache(cache)

        return cache[(cache["timestamp"] >= start_utc) & (cache["timestamp"] < end_utc)][
            ["timestamp", "value"]
        ].reset_index(drop=True)
