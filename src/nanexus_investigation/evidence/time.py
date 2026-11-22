"""UTC integer-microsecond time normalization. G3-A §4.3, G3-B §7.1."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import ROUND_HALF_EVEN, Decimal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

UTC = timezone.utc
MICROS_PER_SECOND = 1_000_000


@dataclass(frozen=True, slots=True)
class InstantMicros:
    """Internal UTC instant as integer microseconds since the Unix epoch."""

    utc_micros: int

    def __post_init__(self) -> None:
        if not isinstance(self.utc_micros, int):
            raise TypeError("utc_micros must be an int")


@dataclass(frozen=True, slots=True)
class IntervalMicros:
    """Half-open UTC interval [start, end). ``end`` may be open/unknown."""

    start: InstantMicros
    end: InstantMicros | None = None

    def __post_init__(self) -> None:
        if self.end is not None and self.end.utc_micros < self.start.utc_micros:
            raise ValueError("interval end must not precede start")

    @property
    def is_open(self) -> bool:
        return self.end is None

    @property
    def is_completed(self) -> bool:
        return self.end is not None

    @property
    def is_degenerate(self) -> bool:
        return self.end is not None and self.end.utc_micros == self.start.utc_micros

    @property
    def is_nondegenerate(self) -> bool:
        return self.end is not None and self.end.utc_micros > self.start.utc_micros


@dataclass(frozen=True, slots=True)
class CompletedIntervalMicros:
    """Completed nondegenerate half-open interval. G3-B §7.1."""

    start: InstantMicros
    end: InstantMicros

    def __post_init__(self) -> None:
        if self.end.utc_micros <= self.start.utc_micros:
            raise ValueError("completed intervals must be nondegenerate")


def datetime_to_utc_micros(
    value: datetime,
    *,
    assume_timezone: str | None = None,
    fold: int | None = None,
) -> InstantMicros:
    """Normalize a timezone-aware datetime to UTC integer microseconds.

    Naive datetimes are rejected unless ``assume_timezone`` supplies a named
    zone. Ambiguous DST wall times require an explicit ``fold`` of 0 or 1.
    """
    if value.tzinfo is None or value.utcoffset() is None:
        if assume_timezone is None:
            raise ValueError("naive timestamps require assume_timezone")
        try:
            zone = ZoneInfo(assume_timezone)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown timezone: {assume_timezone}") from exc
        if fold is None:
            # Probe both folds. If they disagree, the wall time is ambiguous.
            first = value.replace(tzinfo=zone, fold=0)
            second = value.replace(tzinfo=zone, fold=1)
            if first.utcoffset() != second.utcoffset():
                raise ValueError(
                    "ambiguous local time requires an explicit fold of 0 or 1"
                )
            aware = first
        else:
            if fold not in (0, 1):
                raise ValueError("fold must be 0 or 1")
            aware = value.replace(tzinfo=zone, fold=fold)
            # Reject times that fall in a DST gap by checking round-trip.
            as_utc = aware.astimezone(UTC)
            back = as_utc.astimezone(zone)
            if (back.hour, back.minute, back.second, back.microsecond) != (
                value.hour,
                value.minute,
                value.second,
                value.microsecond,
            ):
                raise ValueError("local time does not exist in the given timezone")
        value = aware
    elif fold is not None and value.fold != fold:
        value = value.replace(fold=fold)

    utc_value = value.astimezone(UTC)
    epoch = datetime(1970, 1, 1, tzinfo=UTC)
    delta = utc_value - epoch
    micros = delta.days * 86_400 * MICROS_PER_SECOND + delta.seconds * MICROS_PER_SECOND + delta.microseconds
    return InstantMicros(utc_micros=micros)


def micros_to_utc_datetime(instant: InstantMicros) -> datetime:
    """Convert an internal instant back to a UTC-aware datetime."""
    return datetime.fromtimestamp(instant.utc_micros / MICROS_PER_SECOND, tz=UTC)


def unix_seconds_to_utc_micros(seconds: float | int | str | Decimal) -> InstantMicros:
    """Convert source Unix seconds to UTC microseconds without float identity tests.

    Source floating-point seconds are quantized to integer microseconds. Exact
    float bit patterns are not used as event identities.
    """
    quantized = (
        Decimal(str(seconds)) * Decimal(MICROS_PER_SECOND)
    ).to_integral_value(rounding=ROUND_HALF_EVEN)
    return InstantMicros(utc_micros=int(quantized))


def build_interval(
    start: datetime | InstantMicros,
    end: datetime | InstantMicros | None = None,
    *,
    assume_timezone: str | None = None,
) -> IntervalMicros:
    """Build a half-open interval. Missing end remains open; it is not invented."""
    start_instant = (
        start
        if isinstance(start, InstantMicros)
        else datetime_to_utc_micros(start, assume_timezone=assume_timezone)
    )
    if end is None:
        return IntervalMicros(start=start_instant, end=None)
    end_instant = (
        end
        if isinstance(end, InstantMicros)
        else datetime_to_utc_micros(end, assume_timezone=assume_timezone)
    )
    if end_instant.utc_micros < start_instant.utc_micros:
        raise ValueError("inverted interval: end precedes start")
    return IntervalMicros(start=start_instant, end=end_instant)


def build_completed_interval(
    start: datetime | InstantMicros,
    end: datetime | InstantMicros,
    *,
    assume_timezone: str | None = None,
) -> CompletedIntervalMicros:
    """Build a completed nondegenerate half-open interval for predicate use."""
    interval = build_interval(start, end, assume_timezone=assume_timezone)
    if interval.end is None:
        raise ValueError("completed interval requires an end")
    if interval.is_degenerate:
        raise ValueError("completed intervals must be nondegenerate")
    return CompletedIntervalMicros(start=interval.start, end=interval.end)


def gap_seconds(left: CompletedIntervalMicros, right: CompletedIntervalMicros) -> float:
    """Non-overlap gap in seconds. Zero when intervals touch or overlap."""
    forward = right.start.utc_micros - left.end.utc_micros
    backward = left.start.utc_micros - right.end.utc_micros
    gap_micros = max(forward, backward, 0)
    return gap_micros / MICROS_PER_SECOND
