"""G4-V1-EB2B: atomic terminal Case/report finalization."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from nanexus_investigation.contracts.claim import InvestigationReport
from nanexus_investigation.contracts.enums import CaseStatus
from nanexus_investigation.graph.runtime import (
    ScenarioConfig,
    empty_result,
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
from nanexus_investigation.persistence.budget import DurableBudgetLedger
from nanexus_investigation.persistence.db import InvestigationDatabase
from nanexus_investigation.persistence.errors import (
    CaseLifecycleError,
    DomainFinalizationError,
    IdempotencyConflictError,
)
from nanexus_investigation.persistence.repository import (
    AcceptedReportBinding,
    InvestigationRepository,
)

UTC = timezone.utc
T0 = datetime(2026, 11, 19, 20, 0, tzinfo=UTC)


def _person(execution_id: str) -> object:
    return ok_event_result(
        execution_id=execution_id,
        native_id=f"evt-{execution_id}",
        label="person",
        start=T0 + timedelta(minutes=1),
        end=T0 + timedelta(minutes=2),
    )


def _open_saver(path: Path) -> tuple[sqlite3.Connection, SqliteSaver]:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return conn, saver


def _run_happy(tmp_path: Path, case_id: str, execution_id: str):
    domain_db = InvestigationDatabase.open(tmp_path / f"{case_id}-domain.sqlite")
    runtime = InvestigationRuntime.from_durable_fixture(
        case=make_case(case_id=case_id),
        scenario=ScenarioConfig(
            name=case_id,
            tool_scripts={"search_events": [_person(execution_id)]},
        ),
        domain_db=domain_db,
    )
    conn, saver = _open_saver(tmp_path / f"{case_id}-ckpt.sqlite")
    result = run_investigation(
        runtime, thread_id=f"t-{case_id}", checkpointer=saver, durability="sync"
    )
    return domain_db, runtime, result, conn


def test_l05_complete_finalization_atomic(tmp_path: Path):
    domain_db, runtime, result, conn = _run_happy(tmp_path, "case-l05", "l05")
    assert result["domain_finalized"] is True
    assert result["finalization_error"] is None
    assert result["domain_report"] is not None
    assert runtime.case.status in {CaseStatus.COMPLETED, CaseStatus.PARTIAL}
    repo = InvestigationRepository(domain_db)
    domain_case = repo.get_case(runtime.case.case_id)
    assert domain_case is not None
    assert domain_case.status is runtime.case.status
    domain_report = repo.get_report(runtime.case.case_id)
    assert domain_report is not None
    assert domain_report.report_id == result["domain_report"].report_id
    assert domain_report.execution_status is domain_case.status
    conn.close()
    domain_db.close()


def test_l06_partial_finalization_atomic(tmp_path: Path):
    # EMPTY search yields qualified/partial path via Output Gate.
    domain_db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    runtime = InvestigationRuntime.from_durable_fixture(
        case=make_case(case_id="case-l06"),
        scenario=ScenarioConfig(
            name="l06",
            tool_scripts={"search_events": [empty_result(execution_id="exec-l06")]},
        ),
        domain_db=domain_db,
    )
    conn, saver = _open_saver(tmp_path / "ckpt.sqlite")
    result = run_investigation(
        runtime, thread_id="t-l06", checkpointer=saver, durability="sync"
    )
    assert result["domain_finalized"] is True, result.get("finalization_error")
    domain_case = runtime.domain_repo.get_case("case-l06")
    assert domain_case is not None
    assert domain_case.status in {CaseStatus.PARTIAL, CaseStatus.FAILED}
    if domain_case.status is CaseStatus.PARTIAL:
        report = runtime.domain_repo.get_report("case-l06")
        assert report is not None
        assert report.execution_status is CaseStatus.PARTIAL
        assert result["domain_report"] is not None
    else:
        assert runtime.domain_repo.get_report("case-l06") is None
    conn.close()
    domain_db.close()


def test_l07_invalid_partial_report_rejected(tmp_path: Path):
    db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    repo = InvestigationRepository(db)
    case = make_case(case_id="case-l07")
    repo.admit_and_start_case(case)
    bogus = InvestigationReport(
        report_id="report-bogus",
        case_id="case-l07",
        generated_at=T0,
        scope_summary="scope",
        execution_status=CaseStatus.PARTIAL,
        claims=(),
        evidence_index=(),
        limitations=(),
        unresolved_questions=("unknown",),
        validation_summary="ungrounded",
        report_version="e0-v1",
    )
    with pytest.raises(DomainFinalizationError, match="AcceptedReportBinding"):
        repo.finalize_case(
            case_id="case-l07",
            terminal_status=CaseStatus.PARTIAL,
            accepted=None,
        )
    with pytest.raises(DomainFinalizationError, match="evidence-grounded"):
        binding = AcceptedReportBinding.from_gate_report(
            case_id="case-l07",
            terminal_status=CaseStatus.PARTIAL,
            report=bogus,
        )
        repo.finalize_case(
            case_id="case-l07",
            terminal_status=CaseStatus.PARTIAL,
            accepted=binding,
        )
    assert repo.get_case("case-l07").status is CaseStatus.RUNNING
    assert repo.get_report("case-l07") is None
    db.close()


def test_l08_report_write_failure_rolls_back(tmp_path: Path):
    db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    runtime = InvestigationRuntime.from_durable_fixture(
        case=make_case(case_id="case-l08"),
        scenario=ScenarioConfig(
            name="l08",
            tool_scripts={"search_events": [_person("l08")]},
        ),
        domain_db=db,
    )
    conn, saver = _open_saver(tmp_path / "ckpt.sqlite")
    app = build_investigation_graph(runtime, checkpointer=saver)
    app_state = app.invoke(
        initial_state(runtime),
        config={"configurable": {"thread_id": "t-l08"}},
        durability="sync",
    )
    assert runtime.last_gate is not None and runtime.last_gate.accepted
    runtime.domain_repo.set_fault_hook(
        "before_report_write",
        lambda: (_ for _ in ()).throw(RuntimeError("injected report write failure")),
    )
    with pytest.raises(RuntimeError, match="report write failure"):
        finalize_domain_investigation(runtime)
    assert runtime.domain_repo.get_case("case-l08").status is CaseStatus.RUNNING
    assert runtime.domain_repo.get_report("case-l08") is None
    assert app_state.get("report_id") is not None  # graph candidate only
    conn.close()
    db.close()


def test_l09_status_update_failure_rolls_back_report(tmp_path: Path):
    db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    runtime = InvestigationRuntime.from_durable_fixture(
        case=make_case(case_id="case-l09"),
        scenario=ScenarioConfig(
            name="l09",
            tool_scripts={"search_events": [_person("l09")]},
        ),
        domain_db=db,
    )
    conn, saver = _open_saver(tmp_path / "ckpt.sqlite")
    app = build_investigation_graph(runtime, checkpointer=saver)
    app.invoke(
        initial_state(runtime),
        config={"configurable": {"thread_id": "t-l09"}},
        durability="sync",
    )
    assert runtime.last_gate is not None and runtime.last_gate.accepted
    runtime.domain_repo.set_fault_hook(
        "before_status_update",
        lambda: (_ for _ in ()).throw(RuntimeError("injected status failure")),
    )
    with pytest.raises(RuntimeError, match="status failure"):
        finalize_domain_investigation(runtime)
    assert runtime.domain_repo.get_case("case-l09").status is CaseStatus.RUNNING
    assert runtime.domain_repo.get_report("case-l09") is None
    conn.close()
    db.close()


def test_l10_duplicate_finalization_reuses_report(tmp_path: Path):
    domain_db, runtime, result, conn = _run_happy(tmp_path, "case-l10", "l10")
    assert result["domain_finalized"] is True
    first_report = result["domain_report"]
    second = finalize_domain_investigation(runtime)
    assert second.reused is True
    assert second.report is not None
    assert second.report.report_id == first_report.report_id
    count = domain_db.conn.execute(
        """
        SELECT COUNT(*) AS c FROM nx_artifacts
        WHERE case_id = ? AND artifact_kind = 'investigation_report'
        """,
        ("case-l10",),
    ).fetchone()["c"]
    assert count == 1
    conn.close()
    domain_db.close()


def test_l11_conflicting_finalization_rejected(tmp_path: Path):
    domain_db, runtime, result, conn = _run_happy(tmp_path, "case-l11", "l11")
    assert result["domain_finalized"] is True
    original = result["domain_report"]
    conflicting = original.model_copy(
        update={
            "report_id": "report-other",
            "validation_summary": "different authoritative content",
        }
    )
    with pytest.raises(IdempotencyConflictError):
        runtime.domain_repo.finalize_case(
            case_id="case-l11",
            terminal_status=original.execution_status,
            accepted=AcceptedReportBinding.from_gate_report(
                case_id="case-l11",
                terminal_status=original.execution_status,
                report=conflicting,
            ),
            claims_by_id=dict(runtime.store.claims),
        )
    loaded = runtime.domain_repo.get_report("case-l11")
    assert loaded.report_id == original.report_id
    assert loaded.validation_summary == original.validation_summary
    conn.close()
    domain_db.close()


def test_l12_terminal_case_protection(tmp_path: Path):
    domain_db, runtime, result, conn = _run_happy(tmp_path, "case-l12", "l12")
    assert result["domain_finalized"] is True
    with pytest.raises(CaseLifecycleError, match="terminal|running"):
        runtime.domain_repo.transition_case_to_running("case-l12")
    with pytest.raises(IdempotencyConflictError):
        runtime.domain_repo.admit_case(make_case(case_id="case-l12"))
    # Cannot finalize as a different terminal status.
    other_status = (
        CaseStatus.FAILED
        if runtime.case.status is not CaseStatus.FAILED
        else CaseStatus.CANCELLED
    )
    with pytest.raises(IdempotencyConflictError):
        runtime.domain_repo.finalize_case(
            case_id="case-l12",
            terminal_status=other_status,
            accepted=None,
        )
    conn.close()
    domain_db.close()


def test_l13_reopen_preserves_finalization(tmp_path: Path):
    domain_db, runtime, result, conn = _run_happy(tmp_path, "case-l13", "l13")
    path = tmp_path / "case-l13-domain.sqlite"
    report_id = result["domain_report"].report_id
    status = runtime.case.status
    budget_before = DurableBudgetLedger(domain_db).reserved_units("case-l13")
    domain_db.close()
    conn.close()

    db2 = InvestigationDatabase.open(path)
    repo2 = InvestigationRepository(db2)
    case2 = repo2.get_case("case-l13")
    report2 = repo2.get_report("case-l13")
    assert case2 is not None
    assert case2.status is status
    assert report2 is not None
    assert report2.report_id == report_id
    assert DurableBudgetLedger(db2).reserved_units("case-l13") == budget_before
    db2.close()


def test_f2_output_gate_binding_required_not_bare_boolean(tmp_path: Path):
    """Completed/partial finalization requires AcceptedReportBinding, not a flag."""
    db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    repo = InvestigationRepository(db)
    repo.admit_and_start_case(make_case(case_id="case-f2-bind"))
    report = InvestigationReport(
        report_id="report-f2",
        case_id="case-f2-bind",
        generated_at=T0,
        scope_summary="scope",
        execution_status=CaseStatus.COMPLETED,
        claims=("claim-x",),
        evidence_index=("ev-missing",),
        limitations=("none",),
        unresolved_questions=(),
        validation_summary="ungrounded",
        report_version="e0-v1",
    )
    with pytest.raises(DomainFinalizationError, match="AcceptedReportBinding"):
        repo.finalize_case(
            case_id="case-f2-bind",
            terminal_status=CaseStatus.COMPLETED,
            accepted=None,
        )
    binding = AcceptedReportBinding.from_gate_report(
        case_id="case-f2-bind",
        terminal_status=CaseStatus.COMPLETED,
        report=report,
    )
    substituted = report.model_copy(
        update={"validation_summary": "attacker substituted content"}
    )
    with pytest.raises(DomainFinalizationError, match="does not match"):
        binding.assert_matches(substituted)
    # Binding with unpersisted evidence still cannot finalize.
    with pytest.raises(DomainFinalizationError, match="unpersisted evidence"):
        repo.finalize_case(
            case_id="case-f2-bind",
            terminal_status=CaseStatus.COMPLETED,
            accepted=binding,
            claims_by_id={},
        )
    assert repo.get_case("case-f2-bind").status is CaseStatus.RUNNING
    assert repo.get_report("case-f2-bind") is None
    db.close()


def test_f4_failed_finalization_replay_and_protection(tmp_path: Path):
    db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    runtime = InvestigationRuntime.from_durable_fixture(
        case=make_case(case_id="case-f4"),
        scenario=ScenarioConfig(
            name="f4",
            tool_scripts={
                "search_events": [
                    ok_event_result(
                        execution_id="f4",
                        native_id="evt-f4",
                        label="person",
                        start=T0 + timedelta(minutes=1),
                        end=T0 + timedelta(minutes=2),
                    )
                ]
            },
        ),
        domain_db=db,
    )
    # Force durable failure finalization without an accepted report.
    runtime.last_gate = None
    first = runtime.domain_repo.finalize_case(
        case_id="case-f4",
        terminal_status=CaseStatus.FAILED,
        accepted=None,
    )
    assert first.report is None
    assert first.case.status is CaseStatus.FAILED
    second = runtime.domain_repo.finalize_case(
        case_id="case-f4",
        terminal_status=CaseStatus.FAILED,
        accepted=None,
    )
    assert second.reused is True
    assert second.report is None
    # Cannot promote a failed Case to completed.
    bogus = InvestigationReport(
        report_id="report-promote",
        case_id="case-f4",
        generated_at=T0,
        scope_summary="scope",
        execution_status=CaseStatus.COMPLETED,
        claims=("c1",),
        evidence_index=(),
        limitations=(),
        unresolved_questions=(),
        validation_summary="nope",
        report_version="e0-v1",
    )
    with pytest.raises(IdempotencyConflictError):
        runtime.domain_repo.finalize_case(
            case_id="case-f4",
            terminal_status=CaseStatus.COMPLETED,
            accepted=AcceptedReportBinding.from_gate_report(
                case_id="case-f4",
                terminal_status=CaseStatus.COMPLETED,
                report=bogus,
            ),
        )
    with pytest.raises(CaseLifecycleError, match="terminal"):
        runtime.domain_repo.transition_case_to_running("case-f4")
    assert runtime.domain_repo.get_report("case-f4") is None
    assert runtime.domain_repo.get_case("case-f4").status is CaseStatus.FAILED
    db.close()


def test_l14_graph_checkpoint_alone_not_domain_finalized(tmp_path: Path):
    db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    runtime = InvestigationRuntime.from_durable_fixture(
        case=make_case(case_id="case-l14"),
        scenario=ScenarioConfig(
            name="l14",
            tool_scripts={"search_events": [_person("l14")]},
        ),
        domain_db=db,
    )
    conn, saver = _open_saver(tmp_path / "ckpt.sqlite")
    app = build_investigation_graph(runtime, checkpointer=saver)
    state = app.invoke(
        initial_state(runtime),
        config={"configurable": {"thread_id": "t-l14"}},
        durability="sync",
    )
    # Graph candidate may exist; domain Case must remain running until finalize commits.
    assert state.get("report_id") is not None
    assert runtime.store.report is not None
    assert runtime.domain_repo.get_case("case-l14").status is CaseStatus.RUNNING
    assert runtime.domain_repo.get_report("case-l14") is None

    runtime.case = runtime.domain_repo.get_case("case-l14")
    runtime.domain_repo.set_fault_hook(
        "before_finalize_commit",
        lambda: (_ for _ in ()).throw(RuntimeError("injected finalize commit failure")),
    )
    with pytest.raises(RuntimeError, match="finalize commit failure"):
        finalize_domain_investigation(runtime)
    assert runtime.domain_repo.get_case("case-l14").status is CaseStatus.RUNNING
    assert runtime.domain_repo.get_report("case-l14") is None
    conn.close()
    db.close()
