"""Operator-supplied live Frigate investigation scope.

Live tests must not publish real historical camera windows. Operators set:

- ``NANEXUS_TEST_LIVE_START`` — ISO-8601 start (aware preferred)
- ``NANEXUS_TEST_LIVE_END`` — ISO-8601 end (aware preferred)
- ``NANEXUS_TEST_LIVE_TZ`` — IANA timezone used when start/end are naive
  (default ``UTC``)

Cameras come from ``NANEXUS_FRIGATE_CAMERAS`` (Frigate env / process env).
"""

from __future__ import annotations

import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest


def _parse_instant(raw: str, *, tz: ZoneInfo) -> datetime:
    text = raw.strip().replace("Z", "+00:00")
    instant = datetime.fromisoformat(text)
    if instant.tzinfo is None:
        return instant.replace(tzinfo=tz)
    return instant


def live_time_window() -> tuple[datetime, datetime] | None:
    """Return configured live (start, end) or None when unset."""
    start_raw = (os.environ.get("NANEXUS_TEST_LIVE_START") or "").strip()
    end_raw = (os.environ.get("NANEXUS_TEST_LIVE_END") or "").strip()
    if not start_raw or not end_raw:
        return None
    tz = ZoneInfo((os.environ.get("NANEXUS_TEST_LIVE_TZ") or "UTC").strip())
    start = _parse_instant(start_raw, tz=tz)
    end = _parse_instant(end_raw, tz=tz)
    if end <= start:
        raise ValueError("NANEXUS_TEST_LIVE_END must be after NANEXUS_TEST_LIVE_START")
    return start, end


def require_live_time_window() -> tuple[datetime, datetime]:
    """Fail live tests clearly when the operator has not configured a window."""
    try:
        window = live_time_window()
    except ValueError as exc:
        pytest.fail(f"Invalid live test window configuration: {exc}")
    if window is None:
        pytest.fail(
            "Live Frigate tests require operator-supplied "
            "NANEXUS_TEST_LIVE_START and NANEXUS_TEST_LIVE_END (ISO-8601). "
            "Optional NANEXUS_TEST_LIVE_TZ applies when values are timezone-naive "
            "(default UTC). Do not commit site-specific historical windows."
        )
    return window


def require_live_cameras() -> tuple[str, ...]:
    """Return authorized cameras from Frigate env (no published site defaults)."""
    from nanexus_investigation.vms.frigate.record import load_frigate_env

    env = load_frigate_env()
    cameras = tuple(
        c.strip()
        for c in (env.get("NANEXUS_FRIGATE_CAMERAS") or "").split(",")
        if c.strip()
    )
    if not cameras:
        pytest.fail(
            "Live Frigate tests require NANEXUS_FRIGATE_CAMERAS in the Frigate "
            "env file (or process environment). Do not hardcode site camera names."
        )
    return cameras
