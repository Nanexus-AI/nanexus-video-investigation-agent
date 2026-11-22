"""Live authenticated Frigate checks. Opt-in only: pytest -m live_frigate.

Requires NANEXUS_FRIGATE_* for https://<host>:8971. Does not use port 5000.
Investigation time bounds come from NANEXUS_TEST_LIVE_START / NANEXUS_TEST_LIVE_END
(see live_scope.py). Missing credentials or live window fail clearly — never a silent PASS.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nanexus_investigation.contracts.enums import ToolResultStatus
from nanexus_investigation.vms.frigate.client import FrigateReadOnlyClient
from nanexus_investigation.vms.frigate.normalize import (
    normalize_event_detail_response,
    normalize_events_response,
)
from nanexus_investigation.contracts.identity import Interval
from nanexus_investigation.vms.frigate.record import (
    config_from_env,
    live_client_ready,
    save_recorded_response,
)
from tests.e1.live_scope import require_live_time_window

pytestmark = pytest.mark.live_frigate


def _require_live_client() -> FrigateReadOnlyClient:
    from nanexus_investigation.vms.frigate.record import load_frigate_env

    env = load_frigate_env()
    ready, reason = live_client_ready(env)
    if not ready:
        pytest.fail(
            "Live authenticated Frigate access unavailable: "
            f"{reason}. Configure NANEXUS_FRIGATE_* for https://<host>:8971 "
            "in a git-ignored private env file. "
            "Do not use port 5000. Do not guess credentials."
        )
    return FrigateReadOnlyClient(config_from_env(env))


def _label_counts(events: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in events:
        label = str(event.get("label") or "unknown")
        counts[label] = counts.get(label, 0) + 1
    return counts


def test_b1_01_live_authenticated_historical_events():
    client = _require_live_client()
    version = client.get_version()
    assert version

    cameras = client.config.authorized_cameras
    assert cameras, "NANEXUS_FRIGATE_CAMERAS must list at least one authorized camera"

    start, end = require_live_time_window()
    window_name = f"{start.isoformat()}..{end.isoformat()}"

    response = client.list_events(
        cameras=cameras[:1],
        after=start.timestamp(),
        before=end.timestamp(),
        limit=min(50, client.config.max_limit),
    )
    events = json.loads(response.body.decode("utf-8"))
    assert isinstance(events, list)
    labels = _label_counts(events)
    if not events:
        pytest.fail(
            "Authenticated API succeeded but no retained events were returned in "
            f"the configured live window ({window_name}) for camera "
            f"{cameras[0]!r}. Configure a window with retained data via "
            "NANEXUS_TEST_LIVE_START / NANEXUS_TEST_LIVE_END, or confirm retention."
        )

    private_dir = Path(".local/frigate_recordings")
    response.frigate_version = version
    save_recorded_response(
        private_dir,
        "events_list_live",
        response,
        vms_instance_id=client.config.vms_instance_id,
    )

    from datetime import timezone

    bounds = Interval(
        start_utc=start.astimezone(timezone.utc),
        end_utc=end.astimezone(timezone.utc),
    )
    normalized = normalize_events_response(
        response,
        vms_instance_id=client.config.vms_instance_id,
        cameras=cameras[:1],
        time_bounds=bounds,
        page_limit=min(50, client.config.max_limit),
    )
    assert normalized.tool_result.status in {
        ToolResultStatus.OK,
        ToolResultStatus.EMPTY,
        ToolResultStatus.PARTIAL,
    }
    assert events is not None
    print(
        json.dumps(
            {
                "frigate_version": version,
                "window": window_name,
                "camera": cameras[0],
                "event_count": len(events),
                "labels": labels,
                "normalized_observations": len(normalized.observations),
            }
        )
    )


def test_b1_02_live_event_detail_from_selected_window():
    client = _require_live_client()
    start, end = require_live_time_window()
    listed = client.list_events(
        cameras=client.config.authorized_cameras[:1],
        after=start.timestamp(),
        before=end.timestamp(),
        limit=10,
    )
    events = json.loads(listed.body.decode("utf-8"))
    if not events:
        pytest.fail(
            "No retained event IDs available for detail fetch in the configured "
            "live window (NANEXUS_TEST_LIVE_START / NANEXUS_TEST_LIVE_END)."
        )
    detail = client.get_event(events[0]["id"])
    observation, evidence, tool_result = normalize_event_detail_response(
        detail, vms_instance_id=client.config.vms_instance_id
    )
    assert tool_result.status is ToolResultStatus.OK
    assert observation is not None
    assert evidence is not None
    assert observation.source_locator.native_record_id == events[0]["id"]
    assert evidence.retrieval_spec.get("native_id") == events[0]["id"]
    # Media indicators are reported when present; absence is allowed.
    print(
        json.dumps(
            {
                "event_id_redacted": events[0]["id"][:8] + "…",
                "label": events[0].get("label"),
                "has_clip": events[0].get("has_clip"),
                "has_snapshot": events[0].get("has_snapshot"),
                "detail_has_clip": evidence.retrieval_spec.get("has_clip"),
                "detail_has_snapshot": evidence.retrieval_spec.get("has_snapshot"),
            }
        )
    )
