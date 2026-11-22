"""G4-E1-B2: Frigate search_events through LangGraph Tool Guard dispatch."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests.e1.live_scope import require_live_cameras, require_live_time_window

from nanexus_investigation.contracts.enums import (
    CaseStatus,
    SearchCoverage,
    ToolErrorCode,
    ToolResultStatus,
)
from nanexus_investigation.contracts.identity import Interval
from nanexus_investigation.contracts.tool import ToolRequest
from nanexus_investigation.graph.runtime import (
    ScenarioConfig,
    make_case,
)
from nanexus_investigation.graph.workflow import InvestigationRuntime, run_investigation
from nanexus_investigation.tools.frigate_backend import FrigateToolRegistry
from nanexus_investigation.vms.frigate.client import (
    FrigateAuthError,
    FrigateHttpResponse,
    FrigateScopeError,
    FrigateUnavailableError,
)

FIXTURES = Path(__file__).parent / "fixtures" / "frigate"
PRIVATE_RECORDING = Path(".local/frigate_recordings/events_list_live.json")
UTC = timezone.utc
VMS = "frigate-fixture-0.17.1"


def _sanitized_case(**kwargs):
    # Fixture event times are ~2023-11-14 UTC; scope must contain them.
    start = kwargs.pop("start", datetime(2023, 11, 14, 22, 0, tzinfo=UTC))
    end = kwargs.pop("end", datetime(2023, 11, 14, 23, 30, tzinfo=UTC))
    return make_case(
        case_id=kwargs.pop("case_id", "case-e1-b2-sanitized"),
        cameras=("front_yard",),
        start=start,
        end=end,
        vms_instance_id=kwargs.pop("vms_instance_id", "frigate-fixture"),
        **kwargs,
    )


def test_b2_01_graph_recorded_sanitized_fixture_through_adapter():
    """Deterministic LangGraph run via Frigate recorded transport (sanitized)."""
    case = _sanitized_case()
    scenario = ScenarioConfig(
        name="frigate_recorded_sanitized",
        tool_backend="frigate_recorded",
        frigate_recording_path=str(FIXTURES / "events_list_sanitized.json"),
        search_label=None,
        include_follow_up=False,
        max_tool_attempts=2,
    )
    runtime = InvestigationRuntime.from_fixture(case=case, scenario=scenario)
    assert isinstance(runtime.registry, FrigateToolRegistry)

    result = run_investigation(runtime, thread_id="thread-b2-01-sanitized")
    assert result is not None
    assert runtime.registry.invocations == ["search_events"]
    assert runtime._last_result is None  # noqa: SLF001 — cleared after register

    assert len(runtime.store.receipts) == 1
    receipt = next(iter(runtime.store.receipts.values()))
    assert receipt.source_id == "frigate-fixture"
    assert receipt.camera_ids == ("front_yard",)
    assert receipt.match_count == 2
    assert receipt.coverage is SearchCoverage.EXHAUSTED

    assert len(runtime.store.observations) == 2
    assert len(runtime.store.evidence) == 2
    natives = {
        obs.source_locator.native_record_id
        for obs in runtime.store.observations.values()
    }
    assert natives == {
        "1700000000.111111-abc123",
        "1700000100.333333-def456",
    }
    for obs in runtime.store.observations.values():
        assert obs.source_locator.vms_instance_id == "frigate-fixture"
        assert obs.provenance.startswith("adapter:frigate-read/")

    assert runtime.last_gate is not None
    assert runtime.last_gate.accepted is True
    assert runtime.last_gate.case_status in {
        CaseStatus.COMPLETED,
        CaseStatus.PARTIAL,
    }
    assert runtime.store.report is not None
    assert runtime.store.report.claims
    assert runtime.store.report.evidence_index
    assert runtime.store.report.limitations
    # Case status after graph should match gate decision.
    assert runtime.case.status is runtime.last_gate.case_status


def test_b2_01b_duplicate_native_ids_do_not_duplicate_evidence():
    body = json.dumps(
        [
            {
                "id": "dup-1",
                "camera": "front_yard",
                "label": "person",
                "start_time": 1700000000.0,
                "end_time": 1700000010.0,
                "has_clip": False,
                "has_snapshot": False,
            },
            {
                "id": "dup-1",
                "camera": "front_yard",
                "label": "person",
                "start_time": 1700000000.0,
                "end_time": 1700000010.0,
                "has_clip": False,
                "has_snapshot": False,
            },
        ]
    ).encode("utf-8")
    injected = FrigateHttpResponse(
        status_code=200,
        path="/api/events",
        query={},
        body=body,
        headers={},
        retrieved_at=datetime(2026, 11, 20, 18, 0, tzinfo=UTC),
        frigate_version="fixture",
    )
    case = _sanitized_case(case_id="case-b2-dup")
    scenario = ScenarioConfig(
        name="dup",
        tool_backend="frigate_recorded",
        frigate_recording_path=str(FIXTURES / "events_list_sanitized.json"),
        search_label=None,
    )
    runtime = InvestigationRuntime.from_fixture(case=case, scenario=scenario)
    assert isinstance(runtime.registry, FrigateToolRegistry)
    runtime.registry.injected_response = injected
    run_investigation(runtime, thread_id="thread-b2-dup")
    assert len(runtime.store.observations) == 1
    assert len(runtime.store.evidence) == 1


def test_b2_04_scope_rejected_before_frigate_dispatch():
    case = _sanitized_case()
    scenario = ScenarioConfig(
        name="scope_block",
        tool_backend="frigate_recorded",
        frigate_recording_path=str(FIXTURES / "events_list_sanitized.json"),
        force_camera_ids=("back_alley",),
        search_label=None,
    )
    runtime = InvestigationRuntime.from_fixture(case=case, scenario=scenario)
    result = run_investigation(runtime, thread_id="thread-b2-scope")
    assert runtime.registry.invocations == []
    assert len(runtime.store.observations) == 0
    assert runtime.case.status in {CaseStatus.FAILED, CaseStatus.PARTIAL}


def test_b2_04_related_events_unsupported_not_stubbed():
    case = _sanitized_case()
    scenario = ScenarioConfig(
        name="related_unsupported",
        tool_backend="frigate_recorded",
        frigate_recording_path=str(FIXTURES / "events_list_sanitized.json"),
        force_tool_name="search_related_events",
        search_label="vehicle",
    )
    runtime = InvestigationRuntime.from_fixture(case=case, scenario=scenario)
    run_investigation(runtime, thread_id="thread-b2-related")
    # Capability marks related unsupported → Guard rejects before invoke.
    assert runtime.registry.invocations == []
    assert len(runtime.store.observations) == 0


def test_b2_04_registry_related_events_returns_unsupported():
    registry = FrigateToolRegistry(
        mode="recorded",
        vms_instance_id="frigate-fixture",
        recorded_source=FIXTURES / "events_list_sanitized.json",
    )
    result = registry.invoke(
        ToolRequest(
            tool_name="search_related_events",
            arguments={},
            source_id="frigate-fixture",
            camera_ids=("front_yard",),
        )
    )
    assert result.status is ToolResultStatus.UNSUPPORTED
    assert result.error is not None
    assert result.error.code is ToolErrorCode.DISABLED_CAPABILITY
    assert result.data is None


def test_b2_04_empty_malformed_auth_unavailable_truncated():
    case_bounds = Interval(
        start_utc=datetime(2023, 11, 14, 22, 0, tzinfo=UTC),
        end_utc=datetime(2023, 11, 14, 23, 30, tzinfo=UTC),
    )
    request = ToolRequest(
        tool_name="search_events",
        arguments={
            "time_bounds": {
                "start_utc": case_bounds.start_utc.isoformat(),
                "end_utc": case_bounds.end_utc.isoformat(),
            },
            "limit": 50,
        },
        source_id="frigate-fixture",
        camera_ids=("front_yard",),
    )

    empty_reg = FrigateToolRegistry(
        mode="recorded",
        vms_instance_id="frigate-fixture",
        recorded_source=FIXTURES / "events_empty.json",
    )
    empty = empty_reg.invoke(request)
    assert empty.status is ToolResultStatus.EMPTY
    assert empty.coverage is SearchCoverage.EXHAUSTED
    assert empty.data == []

    malformed = FrigateToolRegistry(
        mode="recorded",
        vms_instance_id="frigate-fixture",
        recorded_source=FIXTURES / "events_list_sanitized.json",
        injected_response=FrigateHttpResponse(
            status_code=200,
            path="/api/events",
            query={},
            body=b"{not-json",
            headers={},
            retrieved_at=datetime.now(tz=UTC),
        ),
    )
    bad = malformed.invoke(request)
    assert bad.status is ToolResultStatus.ERROR
    assert bad.error is not None
    assert bad.error.code is ToolErrorCode.INVALID_RESPONSE

    auth = FrigateToolRegistry(
        mode="recorded",
        vms_instance_id="frigate-fixture",
        recorded_source=FIXTURES / "events_list_sanitized.json",
        inject_error=FrigateAuthError("Frigate login rejected"),
    )
    denied = auth.invoke(request)
    assert denied.status is ToolResultStatus.ERROR
    assert denied.error is not None
    assert denied.error.code is ToolErrorCode.AUTHORIZATION_DENIED

    down = FrigateToolRegistry(
        mode="recorded",
        vms_instance_id="frigate-fixture",
        recorded_source=FIXTURES / "events_list_sanitized.json",
        inject_error=FrigateUnavailableError("Frigate endpoint unavailable"),
    )
    unavail = down.invoke(request)
    assert unavail.status is ToolResultStatus.ERROR
    assert unavail.error is not None
    assert unavail.error.code is ToolErrorCode.SOURCE_UNAVAILABLE

    # Truncated: page limit equals returned count.
    trunc_req = ToolRequest(
        tool_name="search_events",
        arguments={
            "time_bounds": {
                "start_utc": case_bounds.start_utc.isoformat(),
                "end_utc": case_bounds.end_utc.isoformat(),
            },
            "limit": 2,
        },
        source_id="frigate-fixture",
        camera_ids=("front_yard",),
    )
    trunc_reg = FrigateToolRegistry(
        mode="recorded",
        vms_instance_id="frigate-fixture",
        recorded_source=FIXTURES / "events_list_sanitized.json",
        max_limit=50,
    )
    trunc = trunc_reg.invoke(trunc_req)
    assert trunc.status is ToolResultStatus.OK
    assert trunc.coverage is SearchCoverage.TRUNCATED
    assert isinstance(trunc.data, list) and len(trunc.data) == 2


def test_b2_04_client_scope_error_before_network():
    """Unauthorized camera on the Frigate client raises before HTTP."""
    from nanexus_investigation.vms.frigate.client import (
        FrigateClientConfig,
        FrigateReadOnlyClient,
    )

    client = FrigateReadOnlyClient(
        FrigateClientConfig(
            base_url="https://frigate.example.internal:8971",
            vms_instance_id="frigate-fixture",
            username="u",
            password="secret-password",
            ca_bundle="/tmp/does-not-matter.pem",
            disable_hostname_check=True,
            authorized_cameras=("front_yard",),
        )
    )
    with pytest.raises(FrigateScopeError, match="camera not authorized"):
        client.list_events(
            cameras=("garage",),
            after=datetime(2023, 11, 14, 22, 0, tzinfo=UTC).timestamp(),
            before=datetime(2023, 11, 14, 23, 0, tzinfo=UTC).timestamp(),
            limit=10,
        )


def test_b2_04_empty_graph_no_fabricated_observations():
    case = _sanitized_case(case_id="case-b2-empty")
    scenario = ScenarioConfig(
        name="empty_path",
        tool_backend="frigate_recorded",
        frigate_recording_path=str(FIXTURES / "events_empty.json"),
        search_label=None,
    )
    runtime = InvestigationRuntime.from_fixture(case=case, scenario=scenario)
    run_investigation(runtime, thread_id="thread-b2-empty")
    assert len(runtime.store.observations) == 0
    assert len(runtime.store.evidence) == 0
    assert len(runtime.store.receipts) == 1
    receipt = next(iter(runtime.store.receipts.values()))
    tool_result = runtime.store.tool_results_by_receipt[receipt.search_execution_id]
    assert tool_result.status is ToolResultStatus.EMPTY
    assert runtime.last_gate is not None
    # EMPTY qualified path may be PARTIAL/COMPLETED with limitations — never invent obs.
    assert runtime.store.report is not None
    assert all(
        "nothing happened in the world" not in claim.statement.lower()
        for claim in runtime.store.claims.values()
    )


@pytest.mark.live_frigate
def test_b2_01_private_recording_through_graph():
    """B2-01: private recorded transport through real Adapter + Output Gate."""
    if not PRIVATE_RECORDING.is_file():
        pytest.fail(
            "Private Frigate recording missing at "
            f"{PRIVATE_RECORDING}; re-run B1 live acquisition. "
            "Not substituting synthetic PASS."
        )
    live_cameras = require_live_cameras()[:1]
    live_start, live_end = require_live_time_window()
    case = make_case(
        case_id="case-e1-b2-recorded-private",
        question="Find activity on the authorized camera during the configured live test window.",
        cameras=live_cameras,
        start=live_start.astimezone(UTC),
        end=live_end.astimezone(UTC),
        vms_instance_id=VMS,
    )
    scenario = ScenarioConfig(
        name="frigate_recorded_private",
        tool_backend="frigate_recorded",
        frigate_recording_path=str(PRIVATE_RECORDING),
        search_label=None,
        include_follow_up=False,
        max_tool_attempts=2,
        force_time_bounds=Interval(
            start_utc=live_start.astimezone(UTC),
            end_utc=live_end.astimezone(UTC),
        ),
    )
    runtime = InvestigationRuntime.from_fixture(case=case, scenario=scenario)
    run_investigation(runtime, thread_id="thread-b2-01-private")

    assert isinstance(runtime.registry, FrigateToolRegistry)
    assert runtime.registry.invocations == ["search_events"]
    assert runtime.last_gate is not None and runtime.last_gate.accepted
    receipt = next(iter(runtime.store.receipts.values()))
    assert receipt.source_id == VMS
    assert receipt.coverage in {SearchCoverage.EXHAUSTED, SearchCoverage.TRUNCATED}
    assert len(runtime.store.observations) == receipt.match_count
    assert receipt.match_count > 0
    report = runtime.store.report
    assert report is not None
    print(
        json.dumps(
            {
                "case_status": runtime.case.status.value,
                "gate": runtime.last_gate.case_status.value,
                "tool_calls": runtime.registry.invocations,
                "match_count": receipt.match_count,
                "coverage": receipt.coverage.value,
                "observations": len(runtime.store.observations),
                "evidence": len(runtime.store.evidence),
                "receipts": len(runtime.store.receipts),
                "claims": list(report.claims),
                "limitations": list(report.limitations)[:4],
            }
        )
    )


@pytest.mark.live_frigate
def test_b2_02_live_frigate_end_to_end_graph():
    """B2-02/B2-03: authenticated live search_events through Output Gate."""
    from nanexus_investigation.vms.frigate.record import (
        config_from_env,
        live_client_ready,
        load_frigate_env,
    )
    from nanexus_investigation.vms.frigate.client import FrigateReadOnlyClient
    from nanexus_investigation.graph.runtime import _client_config_for_case

    env = load_frigate_env()
    ready, reason = live_client_ready(env)
    if not ready:
        pytest.fail(f"Live Frigate access unavailable: {reason}")

    live_cameras = require_live_cameras()[:1]
    live_start, live_end = require_live_time_window()
    case = make_case(
        case_id="case-e1-b2-live",
        question="Find activity on the authorized camera during the configured live test window.",
        cameras=live_cameras,
        start=live_start.astimezone(UTC),
        end=live_end.astimezone(UTC),
        vms_instance_id=VMS,
    )
    scenario = ScenarioConfig(
        name="frigate_live_e2e",
        tool_backend="frigate_live",
        search_label=None,
        include_follow_up=False,
        max_tool_attempts=2,
        frigate_max_limit=50,
        force_time_bounds=Interval(
            start_utc=live_start.astimezone(UTC),
            end_utc=live_end.astimezone(UTC),
        ),
    )
    cfg = _client_config_for_case(config_from_env(env), case, scenario)
    client = FrigateReadOnlyClient(cfg)
    runtime = InvestigationRuntime.from_fixture(
        case=case,
        scenario=scenario,
        frigate_client=client,
    )
    run_investigation(runtime, thread_id="thread-b2-02-live")

    assert isinstance(runtime.registry, FrigateToolRegistry)
    assert runtime.registry.mode == "live"
    assert runtime.registry.invocations == ["search_events"]
    assert runtime.last_gate is not None
    assert runtime.last_gate.accepted is True
    assert runtime.store.report is not None

    receipt = next(iter(runtime.store.receipts.values()))
    tool_result = runtime.store.tool_results_by_receipt[receipt.search_execution_id]
    assert receipt.source_id == VMS
    assert tool_result.provenance.startswith("adapter:frigate-read/")
    assert tool_result.status in {ToolResultStatus.OK, ToolResultStatus.EMPTY}
    # Report actual counts — do not force the prior B1 observation of 19.
    labels: dict[str, int] = {}
    for obs in runtime.store.observations.values():
        for label in obs.labels:
            labels[label] = labels.get(label, 0) + 1
        assert obs.source_locator.vms_instance_id == VMS
        assert obs.camera_id == live_cameras[0]

    report = runtime.store.report
    print(
        json.dumps(
            {
                "case_status": runtime.case.status.value,
                "gate_status": runtime.last_gate.case_status.value,
                "gate_accepted": runtime.last_gate.accepted,
                "tool_calls": list(runtime.registry.invocations),
                "tool_status": tool_result.status.value,
                "coverage": None if receipt.coverage is None else receipt.coverage.value,
                "match_count": receipt.match_count,
                "observations": len(runtime.store.observations),
                "evidence": len(runtime.store.evidence),
                "receipts": len(runtime.store.receipts),
                "labels": labels,
                "claims": list(report.claims),
                "evidence_index_sample": list(report.evidence_index)[:3],
                "limitations": list(report.limitations),
            }
        )
    )
