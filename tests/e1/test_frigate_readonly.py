"""G4-E1-B1: Frigate read-only client, recording/replay, shared normalization.

Live authenticated checks require NANEXUS_FRIGATE_* env (port 8971). They fail
hard when credentials are unavailable — they do not silently skip or use port
5000. Synthetic fixtures cover normalization, scope, and failure semantics.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from nanexus_investigation.contracts.enums import (
    SearchCoverage,
    ToolErrorCode,
    ToolResultStatus,
)
from nanexus_investigation.contracts.identity import Interval
from nanexus_investigation.contracts.tool import ToolError
from nanexus_investigation.vms.frigate.client import (
    FrigateClientConfig,
    FrigateHttpResponse,
    FrigateProtocolError,
    FrigateReadOnlyClient,
    FrigateScopeError,
)
from nanexus_investigation.vms.frigate.normalize import (
    normalize_error_tool_result,
    normalize_event_detail_response,
    normalize_events_response,
)
from nanexus_investigation.vms.frigate.record import (
    FrigateRecordedTransport,
    live_client_ready,
    load_recorded_response,
    save_recorded_response,
)

FIXTURES = Path(__file__).parent / "fixtures" / "frigate"
UTC = timezone.utc
T0 = datetime(2023, 11, 14, 22, 13, 20, tzinfo=UTC)
T1 = datetime(2023, 11, 14, 23, 13, 20, tzinfo=UTC)


def _http_from_fixture(name: str, *, path: str, query: dict[str, str] | None = None) -> FrigateHttpResponse:
    body = (FIXTURES / name).read_bytes()
    return FrigateHttpResponse(
        status_code=200,
        path=path,
        query=dict(query or {}),
        body=body,
        headers={"Content-Type": "application/json"},
        retrieved_at=datetime(2026, 11, 20, 18, 0, tzinfo=UTC),
        frigate_version="0.18.2-cfa08c5-fixture",
    )


def test_b1_05_scope_enforcement_blocks_before_request():
    config = FrigateClientConfig(
        base_url="https://frigate.example.internal:8971",
        vms_instance_id="frigate-test",
        username="admin",
        password="secret",
        ca_bundle="/tmp/does-not-matter.pem",
        disable_hostname_check=True,
        authorized_cameras=("front_yard",),
        max_limit=50,
    )
    client = FrigateReadOnlyClient(config)
    with pytest.raises(FrigateScopeError, match="camera not authorized"):
        client.list_events(
            cameras=("back_alley",),
            after=T0.timestamp(),
            before=T1.timestamp(),
            limit=10,
        )
    with pytest.raises(FrigateScopeError, match="positive half-open"):
        client.list_events(
            cameras=("front_yard",),
            after=T1.timestamp(),
            before=T0.timestamp(),
            limit=10,
        )
    with pytest.raises(ValueError, match="port 5000"):
        FrigateClientConfig(
            base_url="http://10.0.0.50:5000",
            vms_instance_id="x",
            authorized_cameras=("front_yard",),
        )


def test_b1_04_and_b1_03_recorded_replay_matches_live_normalization(tmp_path: Path):
    live = _http_from_fixture(
        "events_list_sanitized.json",
        path="/api/events",
        query={"cameras": "front_yard", "limit": "25"},
    )
    meta = save_recorded_response(
        tmp_path, "events_list", live, vms_instance_id="frigate-test"
    )
    recorded = load_recorded_response(meta)
    replay = FrigateRecordedTransport(recorded).as_http_response()

    bounds = Interval(start_utc=T0, end_utc=T1)
    live_norm = normalize_events_response(
        live,
        vms_instance_id="frigate-test",
        cameras=("front_yard",),
        time_bounds=bounds,
        page_limit=25,
    )
    replay_norm = normalize_events_response(
        replay,
        vms_instance_id="frigate-test",
        cameras=("front_yard",),
        time_bounds=bounds,
        page_limit=25,
    )

    assert live_norm.tool_result.status is ToolResultStatus.OK
    assert live_norm.tool_result.status is replay_norm.tool_result.status
    assert live_norm.receipt.coverage is replay_norm.receipt.coverage
    assert live_norm.receipt.match_count == replay_norm.receipt.match_count == 2
    assert [o.source_locator.native_record_id for o in live_norm.observations] == [
        o.source_locator.native_record_id for o in replay_norm.observations
    ]
    assert [e.evidence_id for e in live_norm.evidence] == [
        e.evidence_id for e in replay_norm.evidence
    ]
    assert {o.labels for o in live_norm.observations} == {("person",), ("car",)}
    # Integrity hash must match body bytes.
    assert recorded.content_sha256
    assert len(live_norm.observations) == 2


def test_b1_06_empty_malformed_and_auth_semantics_distinguishable():
    empty = normalize_events_response(
        _http_from_fixture("events_empty.json", path="/api/events"),
        vms_instance_id="frigate-test",
        cameras=("front_yard",),
        time_bounds=Interval(start_utc=T0, end_utc=T1),
        page_limit=25,
    )
    assert empty.tool_result.status is ToolResultStatus.EMPTY
    assert empty.receipt.coverage is SearchCoverage.EXHAUSTED
    assert empty.observations == ()
    assert empty.evidence == ()

    with pytest.raises(FrigateProtocolError, match="malformed"):
        normalize_events_response(
            FrigateHttpResponse(
                status_code=200,
                path="/api/events",
                query={},
                body=b"not-json",
                headers={},
                retrieved_at=datetime.now(tz=UTC),
            ),
            vms_instance_id="frigate-test",
            cameras=("front_yard",),
            time_bounds=Interval(start_utc=T0, end_utc=T1),
        )

    denied = normalize_error_tool_result(
        status=ToolResultStatus.ERROR,
        error=ToolError(
            code=ToolErrorCode.AUTHORIZATION_DENIED,
            message="Frigate authorization denied",
        ),
        vms_instance_id="frigate-test",
    )
    unavailable = normalize_error_tool_result(
        status=ToolResultStatus.ERROR,
        error=ToolError(
            code=ToolErrorCode.SOURCE_UNAVAILABLE,
            message="Frigate unavailable",
        ),
        vms_instance_id="frigate-test",
    )
    assert denied.error is not None and unavailable.error is not None
    assert denied.error.code is ToolErrorCode.AUTHORIZATION_DENIED
    assert unavailable.error.code is ToolErrorCode.SOURCE_UNAVAILABLE
    assert denied.status is not ToolResultStatus.EMPTY
    assert unavailable.status is not ToolResultStatus.EMPTY


def test_b1_event_detail_normalization():
    detail = _http_from_fixture(
        "event_detail_sanitized.json",
        path="/api/events/1700000000.111111-abc123",
    )
    observation, evidence, tool_result = normalize_event_detail_response(
        detail, vms_instance_id="frigate-test"
    )
    assert tool_result.status is ToolResultStatus.OK
    assert observation.labels == ("person",)
    assert observation.camera_id == "front_yard"
    assert evidence.source_locator.native_record_id == "1700000000.111111-abc123"
    assert evidence.retrieval_spec.get("has_clip") is True


def test_live_client_ready_rejects_port_5000_and_missing_auth():
    ok, reason = live_client_ready(
        {
            "NANEXUS_FRIGATE_BASE_URL": "http://10.0.0.50:5000",
            "NANEXUS_FRIGATE_USERNAME": "admin",
            "NANEXUS_FRIGATE_PASSWORD": "x",
            "NANEXUS_FRIGATE_CAMERAS": "front_yard",
        }
    )
    assert not ok
    assert "5000" in reason

    ok2, reason2 = live_client_ready(
        {
            "NANEXUS_FRIGATE_BASE_URL": "https://10.0.0.50:8971",
            "NANEXUS_FRIGATE_CAMERAS": "front_yard",
        }
    )
    assert not ok2
    assert "credentials" in reason2.lower() or "PASSWORD" in reason2


def test_hostname_bypass_requires_pinned_ca_bundle():
    with pytest.raises(ValueError, match="ca_bundle"):
        FrigateClientConfig(
            base_url="https://frigate.example.internal:8971",
            vms_instance_id="frigate-test",
            username="admin",
            password="secret",
            disable_hostname_check=True,
            authorized_cameras=("front_yard",),
        )
