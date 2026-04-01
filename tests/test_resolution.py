import isodate
import pandas as pd
import pytest

from energy_cost.resolution import is_divisor, to_pandas_freq


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
