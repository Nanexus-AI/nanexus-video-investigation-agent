"""G4-E1-B2F: multi-event durable evidence registration integrity."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests.e1.live_scope import require_live_cameras, require_live_time_window

from nanexus_investigation.contracts.enums import (
    CaseStatus,
    SearchCoverage,
    ToolResultStatus,
)
from nanexus_investigation.contracts.identity import Interval
from nanexus_investigation.contracts.tool import ToolRequest
from nanexus_investigation.graph.runtime import (
    ScenarioConfig,
    make_case,
    register_tool_artifacts,
)
from nanexus_investigation.graph.workflow import InvestigationRuntime, run_investigation
from nanexus_investigation.persistence.db import InvestigationDatabase
from nanexus_investigation.persistence.errors import IdempotencyConflictError
from nanexus_investigation.persistence.repository import InvestigationRepository
from nanexus_investigation.tools.frigate_backend import FrigateToolRegistry
from nanexus_investigation.vms.frigate.client import FrigateHttpResponse

FIXTURES = Path(__file__).parent / "fixtures" / "frigate"
PRIVATE_RECORDING = Path(".local/frigate_recordings/events_list_live.json")
UTC = timezone.utc
VMS = "frigate-fixture-0.17.1"


def _fixture_case(*, case_id: str = "case-b2f"):
    return make_case(
        case_id=case_id,
        cameras=("front_yard",),
        start=datetime(2023, 11, 14, 22, 0, tzinfo=UTC),
        end=datetime(2023, 11, 14, 23, 30, tzinfo=UTC),
        vms_instance_id="frigate-fixture",
    )


def test_b2f_01_multi_event_replay_memory_and_sqlite(tmp_path: Path):
    """B2F-01: N events → N obs/evidence in memory and SQLite, 1 receipt."""
    domain_db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    case = _fixture_case(case_id="case-b2f-01")
    scenario = ScenarioConfig(
        name="b2f_multi",
        tool_backend="frigate_recorded",
        frigate_recording_path=str(FIXTURES / "events_list_sanitized.json"),
        search_label=None,
        include_follow_up=False,
        max_tool_attempts=2,
    )
    runtime = InvestigationRuntime.from_durable_fixture(
        case=case,
        scenario=scenario,
        domain_db=domain_db,
    )
    result = run_investigation(runtime, thread_id="thread-b2f-01")
    assert result is not None
    assert isinstance(runtime.registry, FrigateToolRegistry)
    assert runtime.registry.invocations == ["search_events"]

    assert len(runtime.store.observations) == 2
    assert len(runtime.store.evidence) == 2
    assert len(runtime.store.receipts) == 1
    assert runtime.last_gate is not None and runtime.last_gate.accepted

    obs_count = domain_db.conn.execute(
        "SELECT COUNT(*) AS c FROM nx_artifacts WHERE artifact_kind = 'observation'"
    ).fetchone()["c"]
    ev_count = domain_db.conn.execute(
        "SELECT COUNT(*) AS c FROM nx_artifacts WHERE artifact_kind = 'evidence_ref'"
    ).fetchone()["c"]
    receipt_count = domain_db.conn.execute(
        "SELECT COUNT(*) AS c FROM nx_artifacts WHERE artifact_kind = 'search_receipt'"
    ).fetchone()["c"]
    assert obs_count == 2
    assert ev_count == 2
    assert receipt_count == 1

    repo = InvestigationRepository(domain_db)
    groups = repo.list_registration_groups(runtime.case.case_id)
    tool_groups = [
        g for g in groups if not str(g.operation_id).endswith(":__finalize__")
    ]
    assert len(tool_groups) == 1
    group = tool_groups[0]
    assert len(group.observations) == 2
    assert len(group.evidences) == 2
    assert {o.source_locator.native_record_id for o in group.observations} == {
        "1700000000.111111-abc123",
        "1700000100.333333-def456",
    }

    # Finalize so report is durable and citation-checkable.
    from nanexus_investigation.graph.runtime import finalize_domain_investigation

    final = finalize_domain_investigation(runtime)
    assert final.terminal_status in {CaseStatus.COMPLETED, CaseStatus.PARTIAL}
    assert final.report is not None
    domain_db.close()


def test_b2f_02_durable_citation_integrity(tmp_path: Path):
    """B2F-02: every accepted report evidence id resolves in SQLite."""
    domain_db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    case = _fixture_case(case_id="case-b2f-02")
    scenario = ScenarioConfig(
        name="b2f_cite",
        tool_backend="frigate_recorded",
        frigate_recording_path=str(FIXTURES / "events_list_sanitized.json"),
        search_label=None,
        max_tool_attempts=2,
    )
    runtime = InvestigationRuntime.from_durable_fixture(
        case=case, scenario=scenario, domain_db=domain_db
    )
    run_investigation(runtime, thread_id="thread-b2f-02")
    from nanexus_investigation.graph.runtime import finalize_domain_investigation

    final = finalize_domain_investigation(runtime)
    assert final.report is not None
    repo = InvestigationRepository(domain_db)
    for evidence_id in final.report.evidence_index:
        persisted = repo.get_evidence(evidence_id)
        assert persisted is not None, f"missing durable evidence {evidence_id}"
        assert persisted.evidence_id == evidence_id
    assert len(final.report.evidence_index) == 2
    domain_db.close()


def test_b2f_03_atomicity_and_idempotency(tmp_path: Path):
    """B2F-03: failed group leaves nothing; replay does not duplicate."""
    domain_db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    case = _fixture_case(case_id="case-b2f-03")
    scenario = ScenarioConfig(
        name="b2f_atomic",
        tool_backend="frigate_recorded",
        frigate_recording_path=str(FIXTURES / "events_list_sanitized.json"),
        search_label=None,
        max_tool_attempts=3,
    )
    runtime = InvestigationRuntime.from_durable_fixture(
        case=case, scenario=scenario, domain_db=domain_db
    )
    assert runtime.domain_repo is not None

    # Build the same artifacts the graph would register, then fault mid-group.
    request = ToolRequest(
        tool_name="search_events",
        arguments={
            "label": None,
            "time_bounds": {
                "start_utc": case.effective_scope.time_bounds.start_utc.isoformat(),
                "end_utc": case.effective_scope.time_bounds.end_utc.isoformat(),
            },
            "limit": 50,
        },
        source_id="frigate-fixture",
        camera_ids=("front_yard",),
    )
    # First: force registry invoke for a real multi-event ToolResult.
    tool_result = runtime.registry.invoke(request)
    assert tool_result.status is ToolResultStatus.OK
    assert isinstance(tool_result.data, list) and len(tool_result.data) == 2

    def boom():
        raise RuntimeError("injected after_observation")

    runtime.domain_repo.set_fault_hook("after_observation", boom)
    with pytest.raises(Exception, match="after_observation|domain evidence"):
        register_tool_artifacts(
            runtime,
            request=request,
            result=tool_result,
            plan_step_id="step-1",
        )
    # Incomplete group must not remain authoritative.
    assert (
        domain_db.conn.execute(
            "SELECT COUNT(*) AS c FROM nx_registration_groups WHERE case_id = ?",
            (case.case_id,),
        ).fetchone()["c"]
        == 0
    )
    assert (
        domain_db.conn.execute(
            "SELECT COUNT(*) AS c FROM nx_artifacts WHERE case_id = ?",
            (case.case_id,),
        ).fetchone()["c"]
        == 0
    )
    assert len(runtime.store.observations) == 0

    runtime.domain_repo.set_fault_hook("after_observation", None)
    first = register_tool_artifacts(
        runtime,
        request=request,
        result=tool_result,
        plan_step_id="step-1",
    )
    second = register_tool_artifacts(
        runtime,
        request=request,
        result=tool_result,
        plan_step_id="step-1",
    )
    assert first[1] == second[1]
    assert (
        domain_db.conn.execute(
            "SELECT COUNT(*) AS c FROM nx_artifacts WHERE artifact_kind = 'observation'"
        ).fetchone()["c"]
        == 2
    )
    assert (
        domain_db.conn.execute(
            "SELECT COUNT(*) AS c FROM nx_artifacts WHERE artifact_kind = 'evidence_ref'"
        ).fetchone()["c"]
        == 2
    )
    assert (
        domain_db.conn.execute(
            "SELECT COUNT(*) AS c FROM nx_artifacts WHERE artifact_kind = 'search_receipt'"
        ).fetchone()["c"]
        == 1
    )

    # Conflicting content under same operation identity must fail closed.
    altered = tool_result.model_copy(
        update={
            "data": [
                {
                    **tool_result.data[0],
                    "native_record_id": "evt-OTHER",
                },
                tool_result.data[1],
            ]
        }
    )
    with pytest.raises((IdempotencyConflictError, Exception)):
        register_tool_artifacts(
            runtime,
            request=request,
            result=altered,
            plan_step_id="step-1",
        )
    domain_db.close()


def test_b2f_04_empty_duplicate_truncated_semantics():
    """B2F-04: empty / duplicate / truncated still behave correctly."""
    case = _fixture_case(case_id="case-b2f-04")
    empty_runtime = InvestigationRuntime.from_fixture(
        case=case,
        scenario=ScenarioConfig(
            name="empty",
            tool_backend="frigate_recorded",
            frigate_recording_path=str(FIXTURES / "events_empty.json"),
            search_label=None,
        ),
    )
    run_investigation(empty_runtime, thread_id="thread-b2f-empty")
    assert len(empty_runtime.store.observations) == 0
    receipt = next(iter(empty_runtime.store.receipts.values()))
    assert (
        empty_runtime.store.tool_results_by_receipt[receipt.search_execution_id].status
        is ToolResultStatus.EMPTY
    )

    # Duplicate native ids → one durable/in-memory pair.
    body = json.dumps(
        [
            {
                "id": "dup-native",
                "camera": "front_yard",
                "label": "person",
                "start_time": 1700000000.0,
                # Missing end_time must not be invented after dedup.
                "has_clip": False,
                "has_snapshot": False,
            },
            {
                "id": "dup-native",
                "camera": "front_yard",
                "label": "person",
                "start_time": 1700000000.0,
                "end_time": 1700000010.0,
                "has_clip": False,
                "has_snapshot": False,
            },
        ]
    ).encode()
    dup_case = _fixture_case(case_id="case-b2f-04-dup")
    dup_runtime = InvestigationRuntime.from_fixture(
        case=dup_case,
        scenario=ScenarioConfig(
            name="dup",
            tool_backend="frigate_recorded",
            frigate_recording_path=str(FIXTURES / "events_list_sanitized.json"),
            search_label=None,
        ),
    )
    assert isinstance(dup_runtime.registry, FrigateToolRegistry)
    dup_runtime.registry.injected_response = FrigateHttpResponse(
        status_code=200,
        path="/api/events",
        query={},
        body=body,
        headers={},
        retrieved_at=datetime.now(tz=UTC),
    )
    run_investigation(dup_runtime, thread_id="thread-b2f-dup")
    assert len(dup_runtime.store.observations) == 1
    obs = next(iter(dup_runtime.store.observations.values()))
    assert obs.end_utc is None  # missing end not invented

    trunc_reg = FrigateToolRegistry(
        mode="recorded",
        vms_instance_id="frigate-fixture",
        recorded_source=FIXTURES / "events_list_sanitized.json",
        max_limit=50,
    )
    trunc = trunc_reg.invoke(
        ToolRequest(
            tool_name="search_events",
            arguments={
                "time_bounds": {
                    "start_utc": datetime(2023, 11, 14, 22, 0, tzinfo=UTC).isoformat(),
                    "end_utc": datetime(2023, 11, 14, 23, 30, tzinfo=UTC).isoformat(),
                },
                "limit": 2,
            },
            source_id="frigate-fixture",
            camera_ids=("front_yard",),
        )
    )
    assert trunc.coverage is SearchCoverage.TRUNCATED
    assert isinstance(trunc.data, list) and len(trunc.data) == 2

    related = trunc_reg.invoke(
        ToolRequest(
            tool_name="search_related_events",
            arguments={},
            source_id="frigate-fixture",
            camera_ids=("front_yard",),
        )
    )
    assert related.status is ToolResultStatus.UNSUPPORTED


@pytest.mark.live_frigate
def test_b2f_05_live_frigate_durable_counts(tmp_path: Path):
    """B2F-05: live window through durable path; report actual counts."""
    from nanexus_investigation.graph.runtime import (
        _client_config_for_case,
        finalize_domain_investigation,
    )
    from nanexus_investigation.vms.frigate.client import FrigateReadOnlyClient
    from nanexus_investigation.vms.frigate.record import (
        config_from_env,
        live_client_ready,
        load_frigate_env,
    )

    env = load_frigate_env()
    ready, reason = live_client_ready(env)
    if not ready:
        pytest.fail(f"Live Frigate access unavailable: {reason}")

    domain_db = InvestigationDatabase.open(tmp_path / "domain-live.sqlite")
    case = make_case(
        case_id="case-b2f-05-live",
        question="Find activity on the authorized camera during the configured live test window.",
        cameras=require_live_cameras()[:1],
        start=require_live_time_window()[0].astimezone(UTC),
        end=require_live_time_window()[1].astimezone(UTC),
        vms_instance_id=VMS,
    )
    scenario = ScenarioConfig(
        name="b2f_live",
        tool_backend="frigate_live",
        search_label=None,
        include_follow_up=False,
        max_tool_attempts=2,
        frigate_max_limit=50,
        force_time_bounds=Interval(
            start_utc=require_live_time_window()[0].astimezone(UTC),
            end_utc=require_live_time_window()[1].astimezone(UTC),
        ),
    )
    cfg = _client_config_for_case(config_from_env(env), case, scenario)
    client = FrigateReadOnlyClient(cfg)
    runtime = InvestigationRuntime.from_durable_fixture(
        case=case,
        scenario=scenario,
        domain_db=domain_db,
        frigate_client=client,
    )
    assert isinstance(runtime.registry, FrigateToolRegistry)

    run_investigation(runtime, thread_id="thread-b2f-05-live")
    assert runtime.last_gate is not None and runtime.last_gate.accepted

    mem_obs = len(runtime.store.observations)
    mem_ev = len(runtime.store.evidence)
    mem_receipts = len(runtime.store.receipts)
    sql_obs = domain_db.conn.execute(
        "SELECT COUNT(*) AS c FROM nx_artifacts WHERE artifact_kind = 'observation'"
    ).fetchone()["c"]
    sql_ev = domain_db.conn.execute(
        "SELECT COUNT(*) AS c FROM nx_artifacts WHERE artifact_kind = 'evidence_ref'"
    ).fetchone()["c"]
    sql_receipts = domain_db.conn.execute(
        "SELECT COUNT(*) AS c FROM nx_artifacts WHERE artifact_kind = 'search_receipt'"
    ).fetchone()["c"]

    assert mem_obs == sql_obs
    assert mem_ev == sql_ev
    assert mem_receipts == sql_receipts == 1
    assert mem_obs == mem_ev
    assert mem_obs > 0

    final = finalize_domain_investigation(runtime)
    assert final.report is not None
    repo = InvestigationRepository(domain_db)
    for evidence_id in final.report.evidence_index:
        assert repo.get_evidence(evidence_id) is not None

    print(
        json.dumps(
            {
                "case_status": runtime.case.status.value,
                "gate": runtime.last_gate.case_status.value,
                "memory_observations": mem_obs,
                "memory_evidence": mem_ev,
                "memory_receipts": mem_receipts,
                "sqlite_observations": sql_obs,
                "sqlite_evidence": sql_ev,
                "sqlite_receipts": sql_receipts,
                "report_evidence_index": len(final.report.evidence_index),
                "private_recording_present": PRIVATE_RECORDING.is_file(),
            }
        )
    )
    domain_db.close()
