"""G4-V1-EB1: LangGraph + SqliteSaver + Investigation Persistence integration."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from nanexus_investigation.contracts.enums import (
    CaseStatus,
    ClaimValidationStatus,
    SearchCoverage,
    ToolErrorCode,
    ToolResultStatus,
)
from nanexus_investigation.contracts.identity import Interval
from nanexus_investigation.contracts.tool import ToolError, ToolResult
from nanexus_investigation.graph.runtime import (
    ScenarioConfig,
    empty_result,
    make_case,
    ok_event_result,
    register_tool_artifacts,
)
from nanexus_investigation.graph.workflow import (
    InvestigationRuntime,
    build_investigation_graph,
    run_investigation,
)
from nanexus_investigation.harness.durable_budget import DurableBudgetAdapter
from nanexus_investigation.persistence.budget import (
    BudgetExhaustedError,
    DurableBudgetLedger,
    ReservationStatus,
)
from nanexus_investigation.persistence.db import InvestigationDatabase
from nanexus_investigation.persistence.errors import IdempotencyConflictError
from nanexus_investigation.persistence.operation_id import (
    build_attempt_identity,
    build_operation_identity,
)
from nanexus_investigation.persistence.repository import InvestigationRepository
from nanexus_investigation.contracts.tool import ToolRequest

UTC = timezone.utc
T0 = datetime(2026, 11, 19, 20, 0, tzinfo=UTC)


def _person_ok(execution_id: str = "exec-person-1") -> ToolResult:
    return ok_event_result(
        execution_id=execution_id,
        native_id="evt-person-1",
        label="person",
        start=T0 + timedelta(minutes=10),
        end=T0 + timedelta(minutes=12),
    )


def _vehicle_ok(execution_id: str = "exec-vehicle-1") -> ToolResult:
    return ok_event_result(
        execution_id=execution_id,
        native_id="evt-vehicle-1",
        label="vehicle",
        start=T0 + timedelta(minutes=15),
        end=T0 + timedelta(minutes=17),
    )


def _open_saver(path: Path) -> tuple[sqlite3.Connection, SqliteSaver]:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return conn, saver


def test_n10_in_memory_path_still_works():
    scenario = ScenarioConfig(
        name="memory_regression",
        include_follow_up=True,
        tool_scripts={
            "search_events": [_person_ok()],
            "search_related_events": [_vehicle_ok()],
        },
    )
    runtime = InvestigationRuntime.from_fixture(case=make_case(), scenario=scenario)
    result = run_investigation(runtime, thread_id="mem-thread")
    assert not runtime.uses_durable_persistence
    assert isinstance(runtime.budget.max_attempts, int)
    assert runtime.registry.invocations == [
        "search_events",
        "search_related_events",
    ]
    assert result["report"] is not None
    assert runtime.case.status in {CaseStatus.COMPLETED, CaseStatus.PARTIAL}


def test_e2e_sqlite_two_step_investigation(tmp_path: Path):
    ckpt_path = tmp_path / "ckpt.sqlite"
    domain_path = tmp_path / "domain.sqlite"
    assert ckpt_path != domain_path

    domain_db = InvestigationDatabase.open(domain_path)
    scenario = ScenarioConfig(
        name="sqlite_e2e",
        include_follow_up=True,
        max_tool_attempts=3,
        tool_scripts={
            "search_events": [_person_ok("exec-sqlite-person")],
            "search_related_events": [_vehicle_ok("exec-sqlite-vehicle")],
        },
    )
    case = make_case(case_id="case-eb1-e2e")
    runtime = InvestigationRuntime.from_durable_fixture(
        case=case, scenario=scenario, domain_db=domain_db
    )
    conn, saver = _open_saver(ckpt_path)
    thread_id = "thread-eb1-e2e"
    result = run_investigation(
        runtime,
        thread_id=thread_id,
        checkpointer=saver,
        durability="sync",
    )
    state = result["state"]
    trace = runtime.trace

    print("\n=== G4-V1-EB1 EXECUTION TRACE ===")
    print("case_id:", runtime.case.case_id)
    print("thread_id:", thread_id)
    print("nodes:", trace.node_sequence())
    print("proposals:", trace.of_kind("tool_proposal"))
    print("guard_accepted:", trace.of_kind("guard_accepted"))
    print("budget:", trace.of_kind("budget_reservation"))
    print("executed:", trace.of_kind("tool_executed"))
    print("domain_evidence:", trace.of_kind("domain_evidence_persisted"))
    print("domain_receipts:", trace.of_kind("domain_receipt_persisted"))
    print("sufficiency:", trace.of_kind("sufficiency"))
    print("claims:", trace.of_kind("claim_assessment"))
    print("output_gate:", trace.of_kind("output_gate"))
    print("case_status:", runtime.case.status.value)
    print("report_id:", state.get("report_id"))
    print("ckpt_db:", ckpt_path.name, "domain_db:", domain_path.name)
    print("=== END TRACE ===\n")

    assert "continue_search" in [s["route"] for s in trace.of_kind("sufficiency")]
    assert runtime.registry.invocations == [
        "search_events",
        "search_related_events",
    ]
    assert len(state["evidence_ids"]) == 2
    assert all(item["reserved"] is True for item in trace.of_kind("budget_reservation"))
    assert result["report"] is not None
    assert runtime.case.status in {CaseStatus.COMPLETED, CaseStatus.PARTIAL}
    # Domain IDs resolve after reopen.
    domain_db.close()
    conn.close()

    domain2 = InvestigationDatabase.open(domain_path)
    repo2 = InvestigationRepository(domain2)
    for evidence_id in state["evidence_ids"]:
        assert repo2.get_evidence(evidence_id) is not None
    for receipt_id in state["search_receipt_ids"]:
        assert repo2.get_search_receipt(receipt_id) is not None
    ledger2 = DurableBudgetLedger(domain2)
    assert ledger2.reserved_units(runtime.case.case_id) == 2
    domain2.close()

    conn2, saver2 = _open_saver(ckpt_path)
    app2 = build_investigation_graph(runtime, checkpointer=saver2)
    recovered = app2.get_state({"configurable": {"thread_id": thread_id}})
    assert recovered.values.get("case_id") == runtime.case.case_id
    assert recovered.values.get("evidence_ids") == state["evidence_ids"]
    other = app2.get_state({"configurable": {"thread_id": "other-thread"}})
    assert not other.values.get("evidence_ids")
    conn2.close()


def test_n1_n2_checkpoint_refs_resolve_after_reconnect(tmp_path: Path):
    ckpt_path = tmp_path / "ckpt.sqlite"
    domain_path = tmp_path / "domain.sqlite"
    domain_db = InvestigationDatabase.open(domain_path)
    scenario = ScenarioConfig(
        name="n1_n2",
        tool_scripts={"search_events": [_person_ok("exec-n1")]},
    )
    runtime = InvestigationRuntime.from_durable_fixture(
        case=make_case(case_id="case-n1"),
        scenario=scenario,
        domain_db=domain_db,
    )
    conn, saver = _open_saver(ckpt_path)
    result = run_investigation(
        runtime, thread_id="t-n1", checkpointer=saver, durability="sync"
    )
    evidence_ids = list(result["state"]["evidence_ids"])
    assert evidence_ids
    domain_db.close()
    conn.close()

    repo = InvestigationRepository(InvestigationDatabase.open(domain_path))
    assert repo.get_evidence(evidence_ids[0]) is not None
    assert (
        repo.get_evidence(evidence_ids[0]).source_locator.native_record_id
        == "evt-person-1"
    )
    repo.db.close()

    conn2, saver2 = _open_saver(ckpt_path)
    app = build_investigation_graph(runtime, checkpointer=saver2)
    recovered = app.get_state({"configurable": {"thread_id": "t-n1"}})
    assert recovered.values["evidence_ids"] == evidence_ids
    conn2.close()


def test_n3_idempotent_registration(tmp_path: Path):
    domain_db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    scenario = ScenarioConfig(
        name="n3",
        tool_scripts={"search_events": [_person_ok("exec-n3")]},
    )
    runtime = InvestigationRuntime.from_durable_fixture(
        case=make_case(case_id="case-n3"),
        scenario=scenario,
        domain_db=domain_db,
    )
    request = ToolRequest(
        tool_name="search_events",
        arguments={"label": "person"},
        source_id="vms-local-1",
        camera_ids=("front_yard",),
    )
    result = _person_ok("exec-n3")
    first = register_tool_artifacts(
        runtime, request=request, result=result, plan_step_id="step-1"
    )
    second = register_tool_artifacts(
        runtime, request=request, result=result, plan_step_id="step-1"
    )
    assert first[1] == second[1]
    count = domain_db.conn.execute(
        "SELECT COUNT(*) AS c FROM nx_artifacts WHERE artifact_kind = 'evidence_ref'"
    ).fetchone()["c"]
    assert count == 1
    domain_db.close()


def test_n4_conflicting_replay_fails(tmp_path: Path):
    domain_db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    scenario = ScenarioConfig(name="n4", tool_scripts={})
    runtime = InvestigationRuntime.from_durable_fixture(
        case=make_case(case_id="case-n4"),
        scenario=scenario,
        domain_db=domain_db,
    )
    request = ToolRequest(
        tool_name="search_events",
        arguments={"label": "person"},
        source_id="vms-local-1",
        camera_ids=("front_yard",),
    )
    register_tool_artifacts(
        runtime,
        request=request,
        result=_person_ok("exec-n4a"),
        plan_step_id="step-1",
    )
    conflict = ok_event_result(
        execution_id="exec-n4b",
        native_id="evt-OTHER",
        label="person",
        start=T0 + timedelta(minutes=10),
        end=T0 + timedelta(minutes=12),
    )
    # Same plan step / request params → same operation identity; different
    # observation content under observation kind must conflict.
    # Use same execution-derived ids by forcing same obs id path: different
    # native id with same operation conflicts on put_observation content.
    with pytest.raises(IdempotencyConflictError):
        # Build observation under same operation with different fingerprint by
        # registering a result that maps to same obs id prefix differently —
        # simplest: put via repo with same operation, different observation body.
        op = build_operation_identity(
            case_id=runtime.case.case_id,
            plan_step_id="step-1",
            tool_name="search_events",
            vms_instance_id="vms-local-1",
            request_parameters={
                "arguments": request.arguments,
                "camera_ids": list(request.camera_ids),
            },
        )
        from nanexus_investigation.contracts.evidence import Observation
        from nanexus_investigation.contracts.enums import ObservationType, RecordType
        from nanexus_investigation.contracts.identity import SourceLocator

        other = Observation(
            observation_id="obs-conflict",
            source_locator=SourceLocator(
                vms_instance_id="vms-local-1",
                record_type=RecordType.EVENT,
                native_record_id="evt-OTHER",
            ),
            source_revision="rev-1",
            observation_type=ObservationType.DETECTION_EVENT,
            camera_id="front_yard",
            start_utc=T0,
            end_utc=T0 + timedelta(minutes=1),
            retrieved_at=T0,
            provenance="stub",
        )
        runtime.domain_repo.put_observation(other, operation=op)
    domain_db.close()


def test_n5_insufficient_durable_budget_blocks_stub(tmp_path: Path):
    domain_db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    scenario = ScenarioConfig(
        name="n5",
        max_tool_attempts=0,
        tool_scripts={"search_events": [_person_ok()]},
    )
    runtime = InvestigationRuntime.from_durable_fixture(
        case=make_case(case_id="case-n5"),
        scenario=scenario,
        domain_db=domain_db,
    )
    conn, saver = _open_saver(tmp_path / "ckpt.sqlite")
    result = run_investigation(
        runtime, thread_id="t-n5", checkpointer=saver, durability="sync"
    )
    assert runtime.registry.invocations == []
    assert "budget" in (result["state"].get("last_error") or "").lower() or (
        "exhausted" in (result["state"].get("last_error") or "").lower()
    )
    conn.close()
    domain_db.close()


def test_n6_unresolved_reservation_no_second_invocation(tmp_path: Path):
    domain_db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    scenario = ScenarioConfig(
        name="n6",
        max_tool_attempts=2,
        tool_scripts={"search_events": [_person_ok("exec-n6")]},
    )
    case = make_case(case_id="case-n6")
    runtime = InvestigationRuntime.from_durable_fixture(
        case=case, scenario=scenario, domain_db=domain_db
    )
    assert isinstance(runtime.budget, DurableBudgetAdapter)

    # Simulate interrupted attempt: durable reservation with unknown outcome.
    op = build_operation_identity(
        case_id=runtime.case.case_id,
        plan_step_id="step-1",
        tool_name="search_events",
        vms_instance_id="vms-local-1",
        request_parameters={
            "arguments": {"label": "person"},
            "camera_ids": ["front_yard"],
        },
    )
    # Align with graph bind parameters for step-1.
    # Graph uses decision.request.arguments including time_bounds — for this
    # test we pre-seed unresolved for the attempt index the graph will use.
    attempt = build_attempt_identity(operation_id=op.operation_id, attempt_index=1)
    # Pre-create unresolved using a reservation id the graph will also try?
    # Graph request_id is req-{case_id}-step-1-0. Seed that identity as unresolved.
    # Operation fingerprint must match what select_validate will bind.
    # Easier approach: run guard path once manually via adapter after seeding
    # unresolved with matching attempt_id from a first reserve+unresolved using
    # the exact operation the graph builds.

    # First, discover the operation identity the graph will use by dry-building
    # the same request parameters as select_validate after validate.
    from nanexus_investigation.graph.planner import build_deterministic_plan

    steps = build_deterministic_plan(
        scenario=scenario.name,
        authorized_cameras=runtime.case.effective_scope.camera_ids,
        time_bounds=runtime.case.effective_scope.time_bounds,
        include_follow_up=False,
    )
    step = steps[0]
    # Mirror ToolGuard argument enrichment for time_bounds.
    args = {
        "label": step.label,
        "time_bounds": {
            "start_utc": runtime.case.effective_scope.time_bounds.start_utc.isoformat(),
            "end_utc": runtime.case.effective_scope.time_bounds.end_utc.isoformat(),
        },
    }
    graph_op = build_operation_identity(
        case_id=runtime.case.case_id,
        plan_step_id=step.step_id,
        tool_name=step.tool_name,
        vms_instance_id=runtime.case.effective_scope.vms_instance_id,
        request_parameters={
            "arguments": args,
            "camera_ids": list(step.camera_ids or runtime.case.effective_scope.camera_ids[:1]),
        },
    )
    graph_attempt = build_attempt_identity(
        operation_id=graph_op.operation_id, attempt_index=1
    )
    reservation_id = f"req-{runtime.case.case_id}-step-1-0"
    runtime.durable_budget.reserve(
        reservation_id=reservation_id,
        case_id=runtime.case.case_id,
        operation_id=graph_op.operation_id,
        attempt_id=graph_attempt.attempt_id,
    )
    runtime.durable_budget.mark_unresolved(reservation_id)
    assert (
        runtime.durable_budget.get_reservation(reservation_id).status
        is ReservationStatus.UNRESOLVED
    )

    conn, saver = _open_saver(tmp_path / "ckpt.sqlite")
    result = run_investigation(
        runtime, thread_id="t-n6", checkpointer=saver, durability="sync"
    )
    assert runtime.registry.invocations == []
    assert result["state"].get("phase") in {
        "terminated",
        "unresolved_attempt_blocked",
        "budget_exhausted",
    }
    err = result["state"].get("last_error") or ""
    assert "unresolved" in err.lower() or "unknown execution outcome" in err.lower()
    # Allowance retained.
    assert runtime.durable_budget.reserved_units(runtime.case.case_id) == 1
    record = runtime.durable_budget.get_reservation(reservation_id)
    assert record is not None
    assert not record.is_confirmed_physical_attempt
    conn.close()
    domain_db.close()


def test_n7_guard_rejects_before_budget_reservation(tmp_path: Path):
    domain_db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    scenario = ScenarioConfig(
        name="n7",
        force_camera_ids=("back_alley",),
        tool_scripts={"search_events": [_person_ok()]},
    )
    runtime = InvestigationRuntime.from_durable_fixture(
        case=make_case(case_id="case-n7"),
        scenario=scenario,
        domain_db=domain_db,
    )
    conn, saver = _open_saver(tmp_path / "ckpt.sqlite")
    result = run_investigation(
        runtime, thread_id="t-n7", checkpointer=saver, durability="sync"
    )
    assert runtime.registry.invocations == []
    assert runtime.trace.of_kind("budget_reservation") == []
    assert any("unauthorized camera" in r for r in runtime.trace.of_kind("guard_rejected")[0]["reasons"])
    assert runtime.durable_budget.reserved_units(runtime.case.case_id) == 0
    conn.close()
    domain_db.close()


def test_n8_empty_error_unsupported_no_fabricated_observations(tmp_path: Path):
    for name, script in (
        (
            "empty",
            empty_result(execution_id="exec-empty", coverage=SearchCoverage.EXHAUSTED),
        ),
        (
            "error",
            ToolResult(
                status=ToolResultStatus.ERROR,
                execution_id="exec-err",
                error=ToolError(
                    code=ToolErrorCode.SOURCE_UNAVAILABLE, message="unavailable"
                ),
                source_id="vms-local-1",
                retrieved_at=T0,
                coverage=SearchCoverage.UNKNOWN,
                provenance="stub",
            ),
        ),
        (
            "unsupported",
            ToolResult(
                status=ToolResultStatus.UNSUPPORTED,
                execution_id="exec-unsup",
                error=ToolError(
                    code=ToolErrorCode.DISABLED_CAPABILITY, message="disabled"
                ),
                source_id="vms-local-1",
                retrieved_at=T0,
                coverage=SearchCoverage.UNKNOWN,
                provenance="stub",
            ),
        ),
    ):
        domain_db = InvestigationDatabase.open(tmp_path / f"domain-{name}.sqlite")
        scenario = ScenarioConfig(
            name=f"n8-{name}",
            tool_scripts={"search_events": [script]},
        )
        runtime = InvestigationRuntime.from_durable_fixture(
            case=make_case(case_id=f"case-n8-{name}"),
            scenario=scenario,
            domain_db=domain_db,
        )
        conn, saver = _open_saver(tmp_path / f"ckpt-{name}.sqlite")
        result = run_investigation(
            runtime,
            thread_id=f"t-n8-{name}",
            checkpointer=saver,
            durability="sync",
        )
        assert result["state"]["observation_ids"] == []
        assert result["state"]["evidence_ids"] == []
        assert result["state"]["search_receipt_ids"]
        receipt = runtime.domain_repo.get_search_receipt(
            result["state"]["search_receipt_ids"][0]
        )
        assert receipt is not None
        obs_count = domain_db.conn.execute(
            "SELECT COUNT(*) AS c FROM nx_artifacts WHERE artifact_kind = 'observation'"
        ).fetchone()["c"]
        assert obs_count == 0
        conn.close()
        domain_db.close()


def test_n9_output_gate_rejects_fabricated_evidence(tmp_path: Path):
    domain_db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    scenario = ScenarioConfig(
        name="n9",
        tool_scripts={"search_events": [_person_ok()]},
        inject_fabricated_evidence_claim=True,
    )
    runtime = InvestigationRuntime.from_durable_fixture(
        case=make_case(case_id="case-n9"),
        scenario=scenario,
        domain_db=domain_db,
    )
    conn, saver = _open_saver(tmp_path / "ckpt.sqlite")
    result = run_investigation(
        runtime, thread_id="t-n9", checkpointer=saver, durability="sync"
    )
    assert runtime.last_gate is not None
    assert runtime.last_gate.accepted is False
    assert runtime.case.status is CaseStatus.FAILED
    conn.close()
    domain_db.close()


def test_databases_remain_separate_tables(tmp_path: Path):
    ckpt_path = tmp_path / "ckpt.sqlite"
    domain_path = tmp_path / "domain.sqlite"
    domain_db = InvestigationDatabase.open(domain_path)
    conn, saver = _open_saver(ckpt_path)
    ckpt_tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    domain_tables = domain_db.table_names()
    assert "checkpoints" in ckpt_tables
    assert "nx_cases" in domain_tables
    assert ckpt_tables & domain_tables == set()
    conn.close()
    domain_db.close()


def test_f1_domain_write_failure_fails_closed(tmp_path: Path):
    """Domain commit failure must not publish uncommitted evidence IDs."""
    domain_db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    scenario = ScenarioConfig(
        name="f1_domain_fail",
        tool_scripts={"search_events": [_person_ok("exec-f1-fail")]},
    )
    runtime = InvestigationRuntime.from_durable_fixture(
        case=make_case(case_id="case-f1-fail"),
        scenario=scenario,
        domain_db=domain_db,
    )
    assert runtime.domain_repo is not None
    real_put_evidence = runtime.domain_repo.put_evidence

    def boom_put_evidence(*args, **kwargs):
        raise RuntimeError("injected domain evidence write failure")

    runtime.domain_repo.put_evidence = boom_put_evidence  # type: ignore[method-assign]

    conn, saver = _open_saver(tmp_path / "ckpt.sqlite")
    result = run_investigation(
        runtime,
        thread_id="t-f1-fail",
        checkpointer=saver,
        durability="sync",
    )
    state = result["state"]
    print("\n=== F1 DOMAIN FAILURE OUTCOME ===")
    print("phase:", state.get("phase"))
    print("last_error:", state.get("last_error"))
    print("evidence_ids:", state.get("evidence_ids"))
    print("observation_ids:", state.get("observation_ids"))
    print("search_receipt_ids:", state.get("search_receipt_ids"))
    print("report:", result["report"])
    print("case_status:", runtime.case.status.value)
    print("store_evidence_keys:", list(runtime.store.evidence.keys()))
    print("trace_failed:", runtime.trace.of_kind("domain_registration_failed"))
    print("=== END F1 OUTCOME ===\n")

    assert state.get("evidence_ids") in (None, [])
    assert state.get("observation_ids") in (None, [])
    assert state.get("phase") == "terminated"
    assert "domain evidence registration failed" in (state.get("last_error") or "")
    # Fail closed: no Output Gate acceptance and no report with uncommitted evidence.
    assert result["report"] is None
    assert runtime.last_gate is None
    assert runtime.case.status is CaseStatus.FAILED
    assert "ev-exec-f1-fail-0" not in runtime.store.evidence
    assert "obs-exec-f1-fail-0" not in runtime.store.observations
    assert runtime.trace.of_kind("domain_registration_failed")
    assert runtime.trace.of_kind("evidence_registration_aborted")
    assert "output_gate" not in runtime.trace.node_sequence()

    # Checkpoint must not retain the failed evidence IDs.
    recovered = build_investigation_graph(runtime, checkpointer=saver).get_state(
        {"configurable": {"thread_id": "t-f1-fail"}}
    )
    assert recovered.values.get("evidence_ids") in (None, [])
    assert recovered.values.get("observation_ids") in (None, [])

    # Reusing the same runtime must not expose stale in-memory evidence.
    runtime.domain_repo.put_evidence = real_put_evidence  # type: ignore[method-assign]
    assert runtime.store.evidence == {}
    assert "ev-exec-f1-fail-0" not in runtime.store.evidence

    conn.close()
    domain_db.close()
