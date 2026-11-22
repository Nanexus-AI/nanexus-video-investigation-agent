"""G4-V1-EB2C1: read-only recovery boundary assessment (R1–R10)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from nanexus_investigation.contracts.enums import (
    CaseStatus,
    EvidenceAvailability,
    EvidenceKind,
    ObservationType,
    RecordType,
    SearchCoverage,
)
from nanexus_investigation.contracts.evidence import EvidenceRef, Observation
from nanexus_investigation.contracts.identity import Interval, SourceLocator
from nanexus_investigation.contracts.search import SearchExecutionReceipt
from nanexus_investigation.graph.runtime import (
    ScenarioConfig,
    finalize_domain_investigation,
    make_case,
    ok_event_result,
)
from nanexus_investigation.graph.workflow import (
    InvestigationRuntime,
    build_investigation_graph,
    initial_state,
    run_investigation,
)
from nanexus_investigation.persistence.budget import DurableBudgetLedger, ReservationStatus
from nanexus_investigation.persistence.db import InvestigationDatabase
from nanexus_investigation.persistence.operation_id import build_operation_identity
from nanexus_investigation.persistence.recovery import (
    RecoveryDiagnostic,
    assess_recovery,
)
from nanexus_investigation.persistence.repository import InvestigationRepository

UTC = timezone.utc
T0 = datetime(2026, 11, 19, 20, 0, tzinfo=UTC)
T1 = datetime(2026, 11, 19, 23, 0, tzinfo=UTC)


def _open_saver(path: Path) -> tuple[sqlite3.Connection, SqliteSaver]:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return conn, saver


def _person(execution_id: str):
    return ok_event_result(
        execution_id=execution_id,
        native_id=f"evt-{execution_id}",
        label="person",
        start=T0 + timedelta(minutes=1),
        end=T0 + timedelta(minutes=2),
    )


def _snapshot_counts(domain: InvestigationDatabase) -> dict[str, int]:
    tables = (
        "nx_cases",
        "nx_artifacts",
        "nx_registration_groups",
        "nx_budget_reservations",
        "nx_operations",
    )
    return {
        table: domain.conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()["c"]
        for table in tables
    }


def _receipt(execution_id: str = "r1") -> SearchExecutionReceipt:
    return SearchExecutionReceipt(
        search_execution_id=f"receipt-{execution_id}",
        tool_name="search_events",
        source_id="vms-local-1",
        camera_ids=("front_yard",),
        time_bounds=Interval(start_utc=T0, end_utc=T1),
        filters={"label": "person"},
        retrieved_at=T0,
        pages_fetched=1,
        match_count=1,
        coverage=SearchCoverage.EXHAUSTED,
    )


def _observation(obs_id: str = "obs-1", native_id: str = "evt-1") -> Observation:
    return Observation(
        observation_id=obs_id,
        source_locator=SourceLocator(
            vms_instance_id="vms-local-1",
            record_type=RecordType.EVENT,
            native_record_id=native_id,
        ),
        source_revision="rev-1",
        observation_type=ObservationType.DETECTION_EVENT,
        camera_id="front_yard",
        start_utc=T0 + timedelta(minutes=10),
        end_utc=T0 + timedelta(minutes=12),
        retrieved_at=T0,
        provenance="stub:search_events",
    )


def _evidence(evidence_id: str = "ev-1", native_id: str = "evt-1") -> EvidenceRef:
    return EvidenceRef(
        evidence_id=evidence_id,
        source_locator=SourceLocator(
            vms_instance_id="vms-local-1",
            record_type=RecordType.EVENT,
            native_record_id=native_id,
        ),
        evidence_kind=EvidenceKind.EVENT_METADATA,
        camera_id="front_yard",
        interval=Interval(
            start_utc=T0 + timedelta(minutes=10),
            end_utc=T0 + timedelta(minutes=12),
        ),
        retrieval_spec={"operation": "search_events"},
        availability=EvidenceAvailability.AVAILABLE,
        provenance="stub:search_events",
    )


def test_r1_created_but_not_running(tmp_path: Path):
    domain_path = tmp_path / "domain.sqlite"
    db = InvestigationDatabase.open(domain_path)
    repo = InvestigationRepository(db)
    case = make_case(case_id="case-r1")
    repo.admit_case(case)
    assert repo.get_case("case-r1").status is CaseStatus.CREATED
    db.close()

    db2 = InvestigationDatabase.open(domain_path)
    repo2 = InvestigationRepository(db2)
    ledger = DurableBudgetLedger(db2)
    conn, saver = _open_saver(tmp_path / "ckpt.sqlite")
    assessment = assess_recovery(
        case_id="case-r1",
        thread_id="t-r1",
        repo=repo2,
        ledger=ledger,
        checkpointer=saver,
    )
    assert assessment.diagnostic is RecoveryDiagnostic.CREATED_NOT_RUNNING
    assert assessment.domain.case_status is CaseStatus.CREATED
    assert assessment.domain_finalized is False
    assert assessment.automatic_tool_redispatch_prohibited is True
    assert assessment.checkpoint.present is False
    conn.close()
    db2.close()


def test_r2_running_before_first_checkpoint(tmp_path: Path):
    domain_path = tmp_path / "domain.sqlite"
    db = InvestigationDatabase.open(domain_path)
    repo = InvestigationRepository(db)
    repo.admit_and_start_case(make_case(case_id="case-r2"))
    assert repo.get_case("case-r2").status is CaseStatus.RUNNING
    db.close()

    db2 = InvestigationDatabase.open(domain_path)
    conn, saver = _open_saver(tmp_path / "ckpt.sqlite")
    assessment = assess_recovery(
        case_id="case-r2",
        thread_id="t-r2",
        repo=InvestigationRepository(db2),
        ledger=DurableBudgetLedger(db2),
        checkpointer=saver,
    )
    assert assessment.diagnostic is RecoveryDiagnostic.RUNNING_NO_CHECKPOINT
    assert assessment.domain.case_status is CaseStatus.RUNNING
    assert any("does not prove" in item for item in assessment.uncertainties)
    assert assessment.automatic_tool_redispatch_prohibited is True
    conn.close()
    db2.close()


def test_r3_reserved_but_outcome_unknown(tmp_path: Path):
    db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    repo = InvestigationRepository(db)
    repo.admit_and_start_case(make_case(case_id="case-r3"))
    ledger = DurableBudgetLedger(db)
    ledger.configure_policy(case_id="case-r3", max_tool_attempts=3)
    op = build_operation_identity(
        case_id="case-r3",
        plan_step_id="step-1",
        tool_name="search_events",
        vms_instance_id="vms-local-1",
        request_parameters={"label": "person"},
    )
    reservation = ledger.reserve(
        reservation_id="res-r3",
        case_id="case-r3",
        operation_id=op.operation_id,
        attempt_id="attempt-r3",
    )
    assert reservation.status is ReservationStatus.RESERVED
    conn, saver = _open_saver(tmp_path / "ckpt.sqlite")
    before = _snapshot_counts(db)
    assessment = assess_recovery(
        case_id="case-r3",
        thread_id="t-r3",
        repo=repo,
        ledger=ledger,
        checkpointer=saver,
    )
    after = _snapshot_counts(db)
    assert before == after
    assert assessment.diagnostic is RecoveryDiagnostic.RESERVATION_OUTCOME_UNKNOWN
    assert assessment.domain.reservations[0].outcome_unknown is True
    assert assessment.domain.reservations[0].is_confirmed_physical_attempt is False
    assert assessment.automatic_tool_redispatch_prohibited is True
    conn.close()
    db.close()


def test_r4_attempt_recorded_result_not_registered(tmp_path: Path):
    db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    repo = InvestigationRepository(db)
    repo.admit_and_start_case(make_case(case_id="case-r4"))
    ledger = DurableBudgetLedger(db)
    ledger.configure_policy(case_id="case-r4", max_tool_attempts=3)
    op = build_operation_identity(
        case_id="case-r4",
        plan_step_id="step-1",
        tool_name="search_events",
        vms_instance_id="vms-local-1",
        request_parameters={"label": "person"},
    )
    ledger.reserve(
        reservation_id="res-r4",
        case_id="case-r4",
        operation_id=op.operation_id,
        attempt_id="attempt-r4",
    )
    ledger.mark_confirmed("res-r4", execution_id="exec-r4")
    assert repo.list_registration_groups("case-r4") == []
    conn, saver = _open_saver(tmp_path / "ckpt.sqlite")
    assessment = assess_recovery(
        case_id="case-r4",
        thread_id="t-r4",
        repo=repo,
        ledger=ledger,
        checkpointer=saver,
    )
    assert assessment.diagnostic is RecoveryDiagnostic.ATTEMPT_WITHOUT_REGISTRATION
    assert not assessment.domain.evidence_ids
    assert assessment.automatic_tool_redispatch_prohibited is True
    conn.close()
    db.close()


def test_r5_domain_group_committed_before_checkpoint(tmp_path: Path):
    domain_path = tmp_path / "domain.sqlite"
    db = InvestigationDatabase.open(domain_path)
    repo = InvestigationRepository(db)
    repo.admit_and_start_case(make_case(case_id="case-r5"))
    op = build_operation_identity(
        case_id="case-r5",
        plan_step_id="step-1",
        tool_name="search_events",
        vms_instance_id="vms-local-1",
        request_parameters={"label": "person"},
    )
    group = repo.register_artifact_group(
        operation=op,
        receipt=_receipt("r5"),
        observation=_observation("obs-r5", "evt-r5"),
        evidence=_evidence("ev-r5", "evt-r5"),
    )
    assert group.reused is False
    db.close()

    db2 = InvestigationDatabase.open(domain_path)
    conn, saver = _open_saver(tmp_path / "ckpt.sqlite")
    assessment = assess_recovery(
        case_id="case-r5",
        thread_id="t-r5",
        repo=InvestigationRepository(db2),
        ledger=DurableBudgetLedger(db2),
        checkpointer=saver,
    )
    assert assessment.diagnostic is RecoveryDiagnostic.ARTIFACT_AHEAD_OF_CHECKPOINT
    assert group.operation_id in assessment.domain.registration_operation_ids
    assert "ev-r5" in assessment.domain.evidence_ids
    assert assessment.checkpoint.present is False
    assert any("not an atomic" in note.lower() or "cross-database" in note for note in assessment.notes)
    assert assessment.automatic_tool_redispatch_prohibited is True
    conn.close()
    db2.close()


def test_r6_graph_finished_domain_still_running(tmp_path: Path):
    domain_path = tmp_path / "domain.sqlite"
    ckpt_path = tmp_path / "ckpt.sqlite"
    db = InvestigationDatabase.open(domain_path)
    runtime = InvestigationRuntime.from_durable_fixture(
        case=make_case(case_id="case-r6"),
        scenario=ScenarioConfig(
            name="r6",
            tool_scripts={"search_events": [_person("r6")]},
        ),
        domain_db=db,
    )
    conn, saver = _open_saver(ckpt_path)
    app = build_investigation_graph(runtime, checkpointer=saver)
    state = app.invoke(
        initial_state(runtime),
        config={"configurable": {"thread_id": "t-r6"}},
        durability="sync",
    )
    assert state.get("report_id") is not None
    assert runtime.domain_repo.get_case("case-r6").status is CaseStatus.RUNNING
    assert runtime.domain_repo.get_report("case-r6") is None
    db.close()
    conn.close()

    db2 = InvestigationDatabase.open(domain_path)
    conn2, saver2 = _open_saver(ckpt_path)
    assessment = assess_recovery(
        case_id="case-r6",
        thread_id="t-r6",
        repo=InvestigationRepository(db2),
        ledger=DurableBudgetLedger(db2),
        checkpointer=saver2,
    )
    assert assessment.diagnostic is RecoveryDiagnostic.GRAPH_FINISHED_DOMAIN_RUNNING
    assert assessment.domain_finalized is False
    assert assessment.authoritative_report_present is False
    assert assessment.checkpoint.report_id is not None
    assert assessment.requires_manual_review is True
    assert assessment.automatic_tool_redispatch_prohibited is True
    conn2.close()
    db2.close()


def test_r7_domain_finalized_checkpoint_behind(tmp_path: Path):
    domain_path = tmp_path / "domain.sqlite"
    ckpt_path = tmp_path / "ckpt.sqlite"
    db = InvestigationDatabase.open(domain_path)
    runtime = InvestigationRuntime.from_durable_fixture(
        case=make_case(case_id="case-r7"),
        scenario=ScenarioConfig(
            name="r7",
            tool_scripts={"search_events": [_person("r7")]},
        ),
        domain_db=db,
    )
    conn, saver = _open_saver(ckpt_path)
    result = run_investigation(
        runtime, thread_id="t-r7", checkpointer=saver, durability="sync"
    )
    assert result["domain_finalized"] is True
    # Simulate a content-behind checkpoint without mutating domain terminal records.
    app = build_investigation_graph(runtime, checkpointer=saver)
    app.update_state(
        {"configurable": {"thread_id": "t-r7"}},
        {
            "phase": "build_plan",
            "report_id": None,
            "evidence_ids": [],
            "observation_ids": [],
            "search_receipt_ids": [],
        },
    )
    domain_status = runtime.domain_repo.get_case("case-r7").status
    report_id = runtime.domain_repo.get_report("case-r7").report_id
    db.close()
    conn.close()

    db2 = InvestigationDatabase.open(domain_path)
    conn2, saver2 = _open_saver(ckpt_path)
    assessment = assess_recovery(
        case_id="case-r7",
        thread_id="t-r7",
        repo=InvestigationRepository(db2),
        ledger=DurableBudgetLedger(db2),
        checkpointer=saver2,
    )
    assert assessment.diagnostic is RecoveryDiagnostic.DOMAIN_FINALIZED_CHECKPOINT_BEHIND
    assert assessment.domain_finalized is True
    assert assessment.domain.case_status is domain_status
    assert assessment.domain.report_id == report_id
    assert assessment.checkpoint.report_id is None
    assert assessment.automatic_tool_redispatch_prohibited is True
    conn2.close()
    db2.close()


def test_r8_checkpoint_references_missing_domain_evidence(tmp_path: Path):
    domain_path = tmp_path / "domain.sqlite"
    ckpt_path = tmp_path / "ckpt.sqlite"
    db = InvestigationDatabase.open(domain_path)
    runtime = InvestigationRuntime.from_durable_fixture(
        case=make_case(case_id="case-r8"),
        scenario=ScenarioConfig(
            name="r8",
            tool_scripts={"search_events": [_person("r8")]},
        ),
        domain_db=db,
    )
    conn, saver = _open_saver(ckpt_path)
    app = build_investigation_graph(runtime, checkpointer=saver)
    state = app.invoke(
        initial_state(runtime),
        config={"configurable": {"thread_id": "t-r8"}},
        durability="sync",
    )
    evidence_ids = list(state.get("evidence_ids") or [])
    assert evidence_ids
    # Simulated interruption artifact loss (test-local DELETE, not OS crash).
    db.conn.execute("DELETE FROM nx_registration_groups WHERE case_id = ?", ("case-r8",))
    db.conn.execute("DELETE FROM nx_artifacts WHERE case_id = ?", ("case-r8",))
    db.close()
    conn.close()

    db2 = InvestigationDatabase.open(domain_path)
    conn2, saver2 = _open_saver(ckpt_path)
    assessment = assess_recovery(
        case_id="case-r8",
        thread_id="t-r8",
        repo=InvestigationRepository(db2),
        ledger=DurableBudgetLedger(db2),
        checkpointer=saver2,
    )
    assert assessment.diagnostic is RecoveryDiagnostic.CHECKPOINT_MISSING_DOMAIN_EVIDENCE
    assert assessment.inconsistencies
    assert assessment.requires_manual_review is True
    assert assessment.automatic_tool_redispatch_prohibited is True
    assert evidence_ids[0] not in assessment.domain.evidence_ids
    conn2.close()
    db2.close()


def test_r9_fully_finalized_consistent(tmp_path: Path):
    domain_path = tmp_path / "domain.sqlite"
    ckpt_path = tmp_path / "ckpt.sqlite"
    db = InvestigationDatabase.open(domain_path)
    runtime = InvestigationRuntime.from_durable_fixture(
        case=make_case(case_id="case-r9"),
        scenario=ScenarioConfig(
            name="r9",
            tool_scripts={"search_events": [_person("r9")]},
        ),
        domain_db=db,
    )
    conn, saver = _open_saver(ckpt_path)
    result = run_investigation(
        runtime, thread_id="t-r9", checkpointer=saver, durability="sync"
    )
    assert result["domain_finalized"] is True
    before = _snapshot_counts(db)
    assessment1 = assess_recovery(
        case_id="case-r9",
        thread_id="t-r9",
        repo=runtime.domain_repo,
        ledger=DurableBudgetLedger(db),
        checkpointer=saver,
    )
    assessment2 = assess_recovery(
        case_id="case-r9",
        thread_id="t-r9",
        repo=runtime.domain_repo,
        ledger=DurableBudgetLedger(db),
        checkpointer=saver,
    )
    after = _snapshot_counts(db)
    assert before == after
    assert assessment1.diagnostic is RecoveryDiagnostic.CONSISTENT_FINALIZED
    assert assessment2.diagnostic is RecoveryDiagnostic.CONSISTENT_FINALIZED
    assert assessment1.domain_finalized is True
    assert assessment1.authoritative_report_present is True
    assert assessment1.read_only is True
    assert runtime.registry.invocations == ["search_events"]
    # Assessment must not create reservations or invoke tools.
    assert len(DurableBudgetLedger(db).list_reservations("case-r9")) == before[
        "nx_budget_reservations"
    ]
    conn.close()
    db.close()


def test_r10_failed_terminal_case(tmp_path: Path):
    domain_path = tmp_path / "domain.sqlite"
    db = InvestigationDatabase.open(domain_path)
    runtime = InvestigationRuntime.from_durable_fixture(
        case=make_case(case_id="case-r10"),
        scenario=ScenarioConfig(name="r10", tool_scripts={}),
        domain_db=db,
    )
    runtime.last_gate = None
    finalized = finalize_domain_investigation(runtime)
    assert finalized.terminal_status is CaseStatus.FAILED
    assert finalized.report is None
    db.close()

    db2 = InvestigationDatabase.open(domain_path)
    conn, saver = _open_saver(tmp_path / "ckpt.sqlite")
    assessment = assess_recovery(
        case_id="case-r10",
        thread_id="t-r10",
        repo=InvestigationRepository(db2),
        ledger=DurableBudgetLedger(db2),
        checkpointer=saver,
    )
    assert assessment.diagnostic is RecoveryDiagnostic.CONSISTENT_FAILED
    assert assessment.domain.case_status is CaseStatus.FAILED
    assert assessment.authoritative_report_present is False
    assert assessment.domain_finalized is True
    assert assessment.automatic_tool_redispatch_prohibited is True
    conn.close()
    db2.close()


def test_case_thread_mismatch_flagged(tmp_path: Path):
    db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    runtime = InvestigationRuntime.from_durable_fixture(
        case=make_case(case_id="case-match-a"),
        scenario=ScenarioConfig(
            name="mismatch",
            tool_scripts={"search_events": [_person("mm")]},
        ),
        domain_db=db,
    )
    conn, saver = _open_saver(tmp_path / "ckpt.sqlite")
    result = run_investigation(
        runtime, thread_id="t-match-a", checkpointer=saver, durability="sync"
    )
    other_evidence = list(result["state"].get("evidence_ids") or [])
    assert other_evidence
    # Assess a different Case against the other Case's thread checkpoint.
    InvestigationRepository(db).admit_and_start_case(make_case(case_id="case-match-b"))
    assessment = assess_recovery(
        case_id="case-match-b",
        thread_id="t-match-a",
        repo=InvestigationRepository(db),
        ledger=DurableBudgetLedger(db),
        checkpointer=saver,
    )
    assert assessment.diagnostic is RecoveryDiagnostic.CASE_THREAD_MISMATCH
    assert assessment.requires_manual_review is True
    assert assessment.domain.case_id == "case-match-b"
    # Other Case's evidence must not appear as this Case's domain evidence.
    assert not set(other_evidence).intersection(assessment.domain.evidence_ids)
    assert assessment.checkpoint.case_id == "case-match-a"
    conn.close()
    db.close()


def test_f1_correct_pair_and_missing_vs_unbound_checkpoint(tmp_path: Path):
    domain_path = tmp_path / "domain.sqlite"
    ckpt_path = tmp_path / "ckpt.sqlite"
    db = InvestigationDatabase.open(domain_path)
    runtime = InvestigationRuntime.from_durable_fixture(
        case=make_case(case_id="case-f1-bind"),
        scenario=ScenarioConfig(
            name="f1bind",
            tool_scripts={"search_events": [_person("f1b")]},
        ),
        domain_db=db,
    )
    conn, saver = _open_saver(ckpt_path)
    result = run_investigation(
        runtime, thread_id="t-f1-bind", checkpointer=saver, durability="sync"
    )
    assert result["domain_finalized"] is True
    ok = assess_recovery(
        case_id="case-f1-bind",
        thread_id="t-f1-bind",
        repo=runtime.domain_repo,
        ledger=DurableBudgetLedger(db),
        checkpointer=saver,
    )
    assert ok.diagnostic is RecoveryDiagnostic.CONSISTENT_FINALIZED
    assert "checkpoint_case_id_matches_requested_case" in ok.invariants_checked

    # Missing checkpoint for another thread (distinct from mismatch).
    missing = assess_recovery(
        case_id="case-f1-bind",
        thread_id="t-f1-absent",
        repo=runtime.domain_repo,
        ledger=DurableBudgetLedger(db),
        checkpointer=saver,
    )
    assert missing.checkpoint.present is False
    assert missing.diagnostic is RecoveryDiagnostic.DOMAIN_FINALIZED_CHECKPOINT_BEHIND
    assert any("missing checkpoint" in note for note in missing.notes)

    # Checkpoint without case_id: thread alone is not positive binding.
    app = build_investigation_graph(runtime, checkpointer=saver)
    app.update_state(
        {"configurable": {"thread_id": "t-f1-unbound"}},
        {
            "case_id": None,
            "phase": "completed",
            "report_id": "candidate-only",
            "evidence_ids": ["ev-should-not-bind"],
        },
    )
    unbound = assess_recovery(
        case_id="case-f1-bind",
        thread_id="t-f1-unbound",
        repo=runtime.domain_repo,
        ledger=DurableBudgetLedger(db),
        checkpointer=saver,
    )
    assert unbound.checkpoint.present is True
    assert not unbound.checkpoint.case_id
    assert any("thread_id alone" in item for item in unbound.uncertainties)
    assert unbound.diagnostic is not RecoveryDiagnostic.CASE_THREAD_MISMATCH
    assert unbound.diagnostic is RecoveryDiagnostic.DOMAIN_FINALIZED_CHECKPOINT_BEHIND
    assert "ev-should-not-bind" not in unbound.domain.evidence_ids
    conn.close()
    db.close()


def test_f2_combined_terminal_missing_evidence_keeps_reservation_visible(tmp_path: Path):
    domain_path = tmp_path / "domain.sqlite"
    ckpt_path = tmp_path / "ckpt.sqlite"
    db = InvestigationDatabase.open(domain_path)
    runtime = InvestigationRuntime.from_durable_fixture(
        case=make_case(case_id="case-f2-combo"),
        scenario=ScenarioConfig(
            name="f2combo",
            tool_scripts={"search_events": [_person("f2c")]},
        ),
        domain_db=db,
    )
    conn, saver = _open_saver(ckpt_path)
    result = run_investigation(
        runtime, thread_id="t-f2-combo", checkpointer=saver, durability="sync"
    )
    assert result["domain_finalized"] is True
    ledger = DurableBudgetLedger(db)
    # Additional unresolved reservation alongside a terminal Case (same policy).
    op = build_operation_identity(
        case_id="case-f2-combo",
        plan_step_id="step-extra",
        tool_name="search_events",
        vms_instance_id="vms-local-1",
        request_parameters={"label": "vehicle"},
    )
    ledger.reserve(
        reservation_id="res-f2-extra",
        case_id="case-f2-combo",
        operation_id=op.operation_id,
        attempt_id="attempt-f2-extra",
    )
    # Remove committed evidence while leaving checkpoint refs intact.
    db.conn.execute(
        "DELETE FROM nx_artifacts WHERE case_id = ? AND artifact_kind = 'evidence_ref'",
        ("case-f2-combo",),
    )
    assessment = assess_recovery(
        case_id="case-f2-combo",
        thread_id="t-f2-combo",
        repo=InvestigationRepository(db),
        ledger=ledger,
        checkpointer=saver,
    )
    assert assessment.diagnostic is RecoveryDiagnostic.INCONSISTENT
    assert assessment.diagnostic is not RecoveryDiagnostic.CONSISTENT_FINALIZED
    assert (
        RecoveryDiagnostic.RESERVATION_OUTCOME_UNKNOWN
        in assessment.secondary_diagnostics
        or any(r.outcome_unknown for r in assessment.domain.reservations)
    )
    assert any("unpersisted" in item or "not committed" in item for item in assessment.inconsistencies)
    assert assessment.automatic_tool_redispatch_prohibited is True
    conn.close()
    db.close()


def test_f3_logical_records_unchanged_after_repeated_assessment(tmp_path: Path):
    domain_path = tmp_path / "domain.sqlite"
    ckpt_path = tmp_path / "ckpt.sqlite"
    db = InvestigationDatabase.open(domain_path)
    runtime = InvestigationRuntime.from_durable_fixture(
        case=make_case(case_id="case-f3-ro"),
        scenario=ScenarioConfig(
            name="f3ro",
            tool_scripts={"search_events": [_person("f3")]},
        ),
        domain_db=db,
    )
    conn, saver = _open_saver(ckpt_path)
    run_investigation(
        runtime, thread_id="t-f3-ro", checkpointer=saver, durability="sync"
    )

    def logical_snapshot() -> dict:
        repo = InvestigationRepository(db)
        ledger = DurableBudgetLedger(db)
        case = repo.get_case("case-f3-ro")
        report = repo.get_report("case-f3-ro")
        return {
            "case_status": None if case is None else case.status.value,
            "case_fp": None
            if case is None
            else db.conn.execute(
                "SELECT payload_fingerprint FROM nx_cases WHERE case_id=?",
                ("case-f3-ro",),
            ).fetchone()["payload_fingerprint"],
            "report_id": None if report is None else report.report_id,
            "artifacts": repo.list_artifact_refs("case-f3-ro"),
            "groups": [
                g.operation_id for g in repo.list_registration_groups("case-f3-ro")
            ],
            "reservations": [
                (r.reservation_id, r.status.value, r.execution_id)
                for r in ledger.list_reservations("case-f3-ro")
            ],
        }

    before = logical_snapshot()
    db.close()
    conn.close()

    db2 = InvestigationDatabase.open(domain_path)
    conn2, saver2 = _open_saver(ckpt_path)
    a1 = assess_recovery(
        case_id="case-f3-ro",
        thread_id="t-f3-ro",
        repo=InvestigationRepository(db2),
        ledger=DurableBudgetLedger(db2),
        checkpointer=saver2,
    )
    a2 = assess_recovery(
        case_id="case-f3-ro",
        thread_id="t-f3-ro",
        repo=InvestigationRepository(db2),
        ledger=DurableBudgetLedger(db2),
        checkpointer=saver2,
    )
    after = {
        "case_status": InvestigationRepository(db2).get_case("case-f3-ro").status.value,
        "case_fp": db2.conn.execute(
            "SELECT payload_fingerprint FROM nx_cases WHERE case_id=?",
            ("case-f3-ro",),
        ).fetchone()["payload_fingerprint"],
        "report_id": InvestigationRepository(db2).get_report("case-f3-ro").report_id,
        "artifacts": InvestigationRepository(db2).list_artifact_refs("case-f3-ro"),
        "groups": [
            g.operation_id
            for g in InvestigationRepository(db2).list_registration_groups("case-f3-ro")
        ],
        "reservations": [
            (r.reservation_id, r.status.value, r.execution_id)
            for r in DurableBudgetLedger(db2).list_reservations("case-f3-ro")
        ],
    }
    assert before == after
    assert a1.diagnostic == a2.diagnostic == RecoveryDiagnostic.CONSISTENT_FINALIZED
    assert a1.read_only and a2.read_only
    conn2.close()
    db2.close()


def test_assessment_never_invokes_stub_tools(tmp_path: Path):
    db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    repo = InvestigationRepository(db)
    repo.admit_and_start_case(make_case(case_id="case-noredispatch"))
    ledger = DurableBudgetLedger(db)
    ledger.configure_policy(case_id="case-noredispatch", max_tool_attempts=2)
    op = build_operation_identity(
        case_id="case-noredispatch",
        plan_step_id="step-1",
        tool_name="search_events",
        vms_instance_id="vms-local-1",
        request_parameters={"label": "person"},
    )
    ledger.reserve(
        reservation_id="res-nd",
        case_id="case-noredispatch",
        operation_id=op.operation_id,
        attempt_id="attempt-nd",
    )
    runtime = InvestigationRuntime.from_durable_fixture(
        case=make_case(case_id="case-noredispatch-runtime"),
        scenario=ScenarioConfig(
            name="nd",
            tool_scripts={"search_events": [_person("nd")]},
        ),
        domain_db=db,
    )
    before_invocations = list(runtime.registry.invocations)
    conn, saver = _open_saver(tmp_path / "ckpt.sqlite")
    assessment = assess_recovery(
        case_id="case-noredispatch",
        thread_id="t-nd",
        repo=repo,
        ledger=ledger,
        checkpointer=saver,
    )
    assert assessment.automatic_tool_redispatch_prohibited is True
    assert runtime.registry.invocations == before_invocations
    assert ledger.get_reservation("res-nd").status is ReservationStatus.RESERVED
    conn.close()
    db.close()
