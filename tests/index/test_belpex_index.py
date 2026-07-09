import datetime as dt
import zoneinfo
from os import environ

import pytest
from isodate import Duration
from pytest import skip

from energy_cost.data.be.electricity.indexes import BelpexRLP0N, BelpexSPP

TIMEZONE = zoneinfo.ZoneInfo("Europe/Brussels")
REAL_VALUES = [
    ("RLP0N", "flanders", 2026, 6, 1, 121.38),
    ("RLP0N", "flanders", 2025, 12, 1, 87.18),
    ("RLP0N", "flanders", 2025, 4, 1, 76.72),
    ("RLP0N", "flanders", 2025, 3, 1, 94.61),
    ("RLP0N", "flanders", 2024, 6, 1, 60.27),
    ("RLP0N", "flanders", 2023, 8, 1, 93.17),
    ("RLP0N", "flanders", 2022, 5, 1, 177.23),
    ("SPP", "flanders", 2026, 5, 1, 42.37),
    ("SPP", "flanders", 2025, 3, 1, 54.53),
    ("SPP", "flanders", 2024, 6, 1, 32.52),
    ("SPP", "flanders", 2024, 5, 1, 31.82),
    ("SPP", "flanders", 2024, 1, 1, 78.44),
    ("SPP", "flanders", 2023, 7, 1, 55.60),
    ("SPP", "flanders", 2022, 12, 1, 297.77),
    ("RLP0N", "belgium", 2026, 4, 1, 84.72),
    ("RLP0N", "belgium", 2025, 10, 1, 78.12),
    ("RLP0N", "belgium", 2024, 2, 1, 63.13),
    ("RLP0N", "belgium", 2023, 8, 1, 93.12),
]

SPP_FLANDERS_HISTORY = [
    ("2022-06-01T00:00:00+02:00", 196.78),
    ("2022-07-01T00:00:00+02:00", 282.89),
    ("2022-08-01T00:00:00+02:00", 386.8),
    ("2022-09-01T00:00:00+02:00", 312.29),
    ("2022-10-01T00:00:00+02:00", 142.69),
    ("2022-11-01T00:00:00+01:00", 175.01),
    ("2022-12-01T00:00:00+01:00", 297.77),
    ("2023-01-01T00:00:00+01:00", 145.49),
    ("2023-02-01T00:00:00+01:00", 130.99),
    ("2023-03-01T00:00:00+01:00", 93.88),
    ("2023-04-01T00:00:00+02:00", 89.88),
    ("2023-05-01T00:00:00+02:00", 61.27),
    ("2023-06-01T00:00:00+02:00", 77.57),
    ("2023-07-01T00:00:00+02:00", 55.6),
    ("2023-08-01T00:00:00+02:00", 73.56),
    ("2023-09-01T00:00:00+02:00", 73.34),
    ("2023-10-01T00:00:00+02:00", 74.85),
    ("2023-11-01T00:00:00+01:00", 86.4),
    ("2023-12-01T00:00:00+01:00", 75.37),
    ("2024-01-01T00:00:00+01:00", 78.44),
    ("2024-02-01T00:00:00+01:00", 61.33),
    ("2024-03-01T00:00:00+01:00", 47.58),
    ("2024-04-01T00:00:00+02:00", 33.29),
    ("2024-05-01T00:00:00+02:00", 31.82),
    ("2024-06-01T00:00:00+02:00", 32.52),
    ("2024-07-01T00:00:00+02:00", 32.41),
    ("2024-08-01T00:00:00+02:00", 33.37),
    ("2024-09-01T00:00:00+02:00", 43.48),
    ("2024-10-01T00:00:00+02:00", 65.55),
    ("2024-11-01T00:00:00+01:00", 104.77),
    ("2024-12-01T00:00:00+01:00", 116.59),
    ("2025-01-01T00:00:00+01:00", 111.79),
    ("2025-02-01T00:00:00+01:00", 119.79),
    ("2025-03-01T00:00:00+01:00", 54.53),
    ("2025-04-01T00:00:00+02:00", 33.99),
    ("2025-05-01T00:00:00+02:00", 15.08),
    ("2025-06-01T00:00:00+02:00", 22.78),
    ("2025-07-01T00:00:00+02:00", 57.22),
    ("2025-08-01T00:00:00+02:00", 34.76),
    ("2025-09-01T00:00:00+02:00", 36.22),
    ("2025-10-01T00:00:00+02:00", 65.99),
    ("2025-11-01T00:00:00+01:00", 79.04),
    ("2025-12-01T00:00:00+01:00", 86.91),
    ("2026-01-01T00:00:00+01:00", 114.16),
    ("2026-02-01T00:00:00+01:00", 74.51),
    ("2026-03-01T00:00:00+01:00", 53.18),
    ("2026-04-01T00:00:00+02:00", 27.95),
    ("2026-05-01T00:00:00+02:00", 42.37),
]

RLP0N_FLANDERS_HISTORY = [
    ("2022-01-01T00:00:00+01:00", 195.48),
    ("2022-02-01T00:00:00+01:00", 165.33),
    ("2022-03-01T00:00:00+01:00", 270.05),
    ("2022-04-01T00:00:00+02:00", 188.05),
    ("2022-05-01T00:00:00+02:00", 177.23),
    ("2022-06-01T00:00:00+02:00", 220.5),
    ("2022-07-01T00:00:00+02:00", 321.13),
    ("2022-08-01T00:00:00+02:00", 452.26),
    ("2022-09-01T00:00:00+02:00", 353.81),
    ("2022-10-01T00:00:00+02:00", 160.77),
    ("2022-11-01T00:00:00+01:00", 190.31),
    ("2022-12-01T00:00:00+01:00", 274.15),
    ("2023-01-01T00:00:00+01:00", 134.14),
    ("2023-02-01T00:00:00+01:00", 145.69),
    ("2023-03-01T00:00:00+01:00", 112.6),
    ("2023-04-01T00:00:00+02:00", 106.93),
    ("2023-05-01T00:00:00+02:00", 80.63),
    ("2023-06-01T00:00:00+02:00", 93.24),
    ("2023-07-01T00:00:00+02:00", 75.77),
    ("2023-08-01T00:00:00+02:00", 93.17),
    ("2023-09-01T00:00:00+02:00", 97.24),
    ("2023-10-01T00:00:00+02:00", 90.33),
    ("2023-11-01T00:00:00+01:00", 95.94),
    ("2023-12-01T00:00:00+01:00", 70.82),
    ("2024-01-01T00:00:00+01:00", 80.65),
    ("2024-02-01T00:00:00+01:00", 63.0),
    ("2024-03-01T00:00:00+01:00", 63.22),
    ("2024-04-01T00:00:00+02:00", 48.91),
    ("2024-05-01T00:00:00+02:00", 55.08),
    ("2024-06-01T00:00:00+02:00", 60.27),
    ("2024-07-01T00:00:00+02:00", 55.34),
    ("2024-08-01T00:00:00+02:00", 67.08),
    ("2024-09-01T00:00:00+02:00", 68.82),
    ("2024-10-01T00:00:00+02:00", 81.96),
    ("2024-11-01T00:00:00+01:00", 111.91),
    ("2024-12-01T00:00:00+01:00", 108.27),
    ("2025-01-01T00:00:00+01:00", 115.25),
    ("2025-02-01T00:00:00+01:00", 131.51),
    ("2025-03-01T00:00:00+01:00", 94.61),
    ("2025-04-01T00:00:00+02:00", 76.72),
    ("2025-05-01T00:00:00+02:00", 64.97),
    ("2025-06-01T00:00:00+02:00", 68.98),
    ("2025-07-01T00:00:00+02:00", 86.26),
    ("2025-08-01T00:00:00+02:00", 72.01),
    ("2025-09-01T00:00:00+02:00", 68.27),
    ("2025-10-01T00:00:00+02:00", 78.43),
    ("2025-11-01T00:00:00+01:00", 89.59),
    ("2025-12-01T00:00:00+01:00", 87.18),
    ("2026-01-01T00:00:00+01:00", 110.88),
    ("2026-02-01T00:00:00+01:00", 87.39),
    ("2026-03-01T00:00:00+01:00", 98.25),
    ("2026-04-01T00:00:00+02:00", 85.80),
    ("2026-05-01T00:00:00+02:00", 99.02),
]


def _next_month_start(ts: dt.datetime) -> dt.datetime:
    return (
        dt.datetime(ts.year + 1, 1, 1, tzinfo=TIMEZONE)
        if ts.month == 12
        else dt.datetime(ts.year, ts.month + 1, 1, tzinfo=TIMEZONE)
    )


def _assert_historical_monthly_values(
    *,
    label: str,
    index: BelpexSPP | BelpexRLP0N,
    history: list[tuple[str, float]],
    abs_tolerance: float = 0.01,
) -> None:
    starts = [dt.datetime.fromisoformat(timestamp).astimezone(TIMEZONE) for timestamp, _ in history]
    expected_map = {(start.year, start.month): expected for start, (_, expected) in zip(starts, history, strict=False)}

    start = starts[0]
    end = _next_month_start(starts[-1])

    df = index.get_values(start, end, Duration(months=1), TIMEZONE)
    actual_map = {
        (ts.astimezone(TIMEZONE).year, ts.astimezone(TIMEZONE).month): round(float(value), 2)
        for ts, value in zip(df["timestamp"], df["value"], strict=False)
    }

    issues: list[str] = []
    for month_key in expected_map:
        expected = expected_map[month_key]
        actual = actual_map.get(month_key)
        if actual is None:
            issues.append(f"{month_key[0]}-{month_key[1]:02d}: missing value")
            continue
        if abs(actual - expected) > abs_tolerance:
            issues.append(
                f"{month_key[0]}-{month_key[1]:02d}: expected={expected:.2f} actual={actual:.2f} delta={actual - expected:+.2f}"
            )

    extra_months = sorted(set(actual_map) - set(expected_map))
    for year, month in extra_months:
        issues.append(f"{year}-{month:02d}: unexpected extra month in response")

    assert not issues, f"{label} mismatches ({len(issues)}):\n" + "\n".join(issues)


@pytest.mark.parametrize(("profile", "region", "year", "month", "day", "expected"), REAL_VALUES)
def test_belpex_real_values(
    profile: str,
    region: str,
    year: int,
    month: int,
    day: int,
    expected: float,
) -> None:
    if not environ.get("ENTSOE_API_KEY"):
        skip("ENTSOE_API_KEY not set")

    if profile == "SPP":
        index = BelpexSPP(entsoe_api_key=environ.get("ENTSOE_API_KEY", ""), region=region)
    elif profile == "RLP0N":
        index = BelpexRLP0N(entsoe_api_key=environ.get("ENTSOE_API_KEY", ""), region=region)
    else:
        raise ValueError(f"Unknown profile: {profile}")

    start = dt.datetime(year, month, day, tzinfo=TIMEZONE)
    end = (
        dt.datetime(year + 1, 1, 1, tzinfo=TIMEZONE)
        if month == 12
        else dt.datetime(year, month + 1, 1, tzinfo=TIMEZONE)
    )

    df = index.get_values(start, end, Duration(months=1), TIMEZONE)

    assert len(df) == 1
    actual = round(float(df["value"].iloc[0]), 2)
    assert actual == pytest.approx(expected)


def test_belpex_spp_flanders_historical_values() -> None:
    if not environ.get("ENTSOE_API_KEY"):
        skip("ENTSOE_API_KEY not set")

    index = BelpexSPP(entsoe_api_key=environ.get("ENTSOE_API_KEY", ""), region="flanders")
    _assert_historical_monthly_values(
        label="SPP flanders",
        index=index,
        history=SPP_FLANDERS_HISTORY,
        abs_tolerance=0.01,
    )


def test_belpex_rlp0n_flanders_historical_values() -> None:
    if not environ.get("ENTSOE_API_KEY"):
        skip("ENTSOE_API_KEY not set")

    index = BelpexRLP0N(entsoe_api_key=environ.get("ENTSOE_API_KEY", ""), region="flanders")
    _assert_historical_monthly_values(
        label="RLP0N flanders",
        index=index,
        history=RLP0N_FLANDERS_HISTORY,
        abs_tolerance=0.01,
    )
