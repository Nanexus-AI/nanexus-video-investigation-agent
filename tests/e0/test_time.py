"""UTC microsecond normalization tests. G3-A §4.3."""

from datetime import datetime, timedelta, timezone

import pytest

from nanexus_investigation.evidence.time import (
    build_completed_interval,
    build_interval,
    datetime_to_utc_micros,
    micros_to_utc_datetime,
    unix_seconds_to_utc_micros,
)

UTC = timezone.utc


def test_equivalent_offsets_normalize_to_same_instant():
    eastern = datetime(2026, 11, 19, 15, 0, tzinfo=timezone(timedelta(hours=-5)))
    utc = datetime(2026, 11, 19, 20, 0, tzinfo=UTC)
    assert datetime_to_utc_micros(eastern) == datetime_to_utc_micros(utc)


def test_naive_datetime_rejected_without_timezone_context():
    with pytest.raises(ValueError, match="naive timestamps require assume_timezone"):
        datetime_to_utc_micros(datetime(2026, 11, 19, 20, 0))


def test_naive_datetime_with_timezone_context():
    instant = datetime_to_utc_micros(
        datetime(2026, 11, 19, 15, 0),
        assume_timezone="America/New_York",
    )
    assert micros_to_utc_datetime(instant) == datetime(2026, 11, 19, 20, 0, tzinfo=UTC)


def test_dst_ambiguous_local_time_requires_fold():
    # US fall-back ambiguous hour in America/New_York.
    local = datetime(2026, 11, 1, 1, 30)
    with pytest.raises(ValueError, match="ambiguous local time"):
        datetime_to_utc_micros(local, assume_timezone="America/New_York")

    first = datetime_to_utc_micros(
        local, assume_timezone="America/New_York", fold=0
    )
    second = datetime_to_utc_micros(
        local, assume_timezone="America/New_York", fold=1
    )
    assert first != second
    assert second.utc_micros - first.utc_micros == 3600 * 1_000_000


def test_dst_gap_local_time_rejected():
    # Spring-forward gap in America/New_York: 2026-03-08 02:30 does not exist.
    with pytest.raises(ValueError, match="does not exist"):
        datetime_to_utc_micros(
            datetime(2026, 3, 8, 2, 30),
            assume_timezone="America/New_York",
            fold=0,
        )


def test_inverted_interval_rejected():
    start = datetime(2026, 11, 19, 21, 0, tzinfo=UTC)
    end = datetime(2026, 11, 19, 20, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="inverted"):
        build_interval(start, end)


def test_missing_endpoint_remains_open():
    start = datetime(2026, 11, 19, 20, 0, tzinfo=UTC)
    interval = build_interval(start, None)
    assert interval.is_open
    assert interval.end is None


def test_touching_endpoints_are_valid_half_open_bounds():
    start = datetime(2026, 11, 19, 20, 0, tzinfo=UTC)
    mid = datetime(2026, 11, 19, 20, 5, tzinfo=UTC)
    left = build_completed_interval(start, mid)
    right = build_completed_interval(mid, datetime(2026, 11, 19, 20, 10, tzinfo=UTC))
    assert left.end == right.start


def test_microsecond_comparison_precision():
    a = datetime(2026, 11, 19, 20, 0, 0, 1, tzinfo=UTC)
    b = datetime(2026, 11, 19, 20, 0, 0, 2, tzinfo=UTC)
    assert datetime_to_utc_micros(a).utc_micros + 1 == datetime_to_utc_micros(b).utc_micros


def test_unix_seconds_quantize_to_microseconds_without_float_identity():
    first = unix_seconds_to_utc_micros(1763580000.1234567)
    second = unix_seconds_to_utc_micros("1763580000.1234567")
    assert first == second
    assert isinstance(first.utc_micros, int)


def test_degenerate_completed_interval_rejected():
    stamp = datetime(2026, 11, 19, 20, 0, tzinfo=UTC)
    with pytest.raises(ValueError, match="nondegenerate"):
        build_completed_interval(stamp, stamp)


def test_source_resolution_differs_from_internal_microseconds():
    # A source float second and an aware datetime can map to the same micros.
    dt = datetime(2026, 11, 19, 20, 0, 0, tzinfo=UTC)
    from_dt = datetime_to_utc_micros(dt)
    from_unix = unix_seconds_to_utc_micros(dt.timestamp())
    assert from_dt == from_unix
