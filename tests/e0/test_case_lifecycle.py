"""G4-V1-EB2B: durable Case admission (created → running) boundaries."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver

from nanexus_investigation.contracts.enums import CaseStatus
from nanexus_investigation.graph.runtime import ScenarioConfig, make_case, ok_event_result
from nanexus_investigation.graph.workflow import InvestigationRuntime, run_investigation
from nanexus_investigation.persistence.db import InvestigationDatabase
from nanexus_investigation.persistence.errors import (
    CaseLifecycleError,
    IdempotencyConflictError,
)
from nanexus_investigation.persistence.repository import InvestigationRepository

UTC = timezone.utc
T0 = datetime(2026, 11, 19, 20, 0, tzinfo=UTC)


def _person(execution_id: str = "exec-life") -> object:
    return ok_event_result(
        execution_id=execution_id,
        native_id="evt-life",
        label="person",
        start=T0 + timedelta(minutes=1),
        end=T0 + timedelta(minutes=2),
    )


def test_l01_admission_created_then_running_before_graph(tmp_path: Path):
    db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    repo = InvestigationRepository(db)
    case = make_case(case_id="case-l01")
    graph_started = {"value": False}

    admitted = repo.admit_case(case)
    assert admitted.status is CaseStatus.CREATED
    persisted = repo.get_case(case.case_id)
    assert persisted is not None
    assert persisted.status is CaseStatus.CREATED

    running = repo.transition_case_to_running(case.case_id)
    assert running.status is CaseStatus.RUNNING
    assert repo.get_case(case.case_id).status is CaseStatus.RUNNING

    scenario = ScenarioConfig(
        name="l01",
        tool_scripts={"search_events": [_person("exec-l01")]},
    )
    runtime = InvestigationRuntime.from_durable_fixture(
        case=make_case(case_id="case-l01b"),
        scenario=scenario,
        domain_db=db,
    )
    assert runtime.case.status is CaseStatus.RUNNING
    assert repo.get_case("case-l01b").status is CaseStatus.RUNNING
    assert any(e["kind"] == "domain_case_admitted" for e in runtime.trace.events)

    conn = sqlite3.connect(str(tmp_path / "ckpt.sqlite"), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    graph_started["value"] = True
    result = run_investigation(
        runtime, thread_id="t-l01", checkpointer=saver, durability="sync"
    )
    assert graph_started["value"] is True
    assert result["domain_finalized"] is True
    conn.close()
    db.close()


def test_l02_creation_failure_does_not_start_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    repo = InvestigationRepository(db)
    case = make_case(case_id="case-l02")
    repo.set_fault_hook(
        "before_case_admit",
        lambda: (_ for _ in ()).throw(RuntimeError("injected create failure")),
    )
    with pytest.raises(RuntimeError, match="create failure"):
        repo.admit_case(case)
    assert repo.get_case(case.case_id) is None

    def boom_admit(self, case):  # noqa: ANN001
        raise RuntimeError("injected create failure")

    monkeypatch.setattr(InvestigationRepository, "admit_case", boom_admit)
    with pytest.raises(RuntimeError, match="create failure"):
        InvestigationRuntime.from_durable_fixture(
            case=make_case(case_id="case-l02b"),
            scenario=ScenarioConfig(name="l02", tool_scripts={}),
            domain_db=db,
        )
    assert InvestigationRepository(db).get_case("case-l02b") is None
    db.close()


def test_l03_running_transition_failure_keeps_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    repo = InvestigationRepository(db)
    case = make_case(case_id="case-l03")
    repo.admit_case(case)
    assert repo.get_case(case.case_id).status is CaseStatus.CREATED

    repo.set_fault_hook(
        "before_running_transition",
        lambda: (_ for _ in ()).throw(RuntimeError("injected running failure")),
    )
    with pytest.raises(RuntimeError, match="running failure"):
        repo.transition_case_to_running(case.case_id)

    still = repo.get_case(case.case_id)
    assert still is not None
    assert still.status is CaseStatus.CREATED
    assert still.question == case.question

    def boom_running(self, case_id: str):
        raise RuntimeError("injected running failure")

    monkeypatch.setattr(
        InvestigationRepository, "transition_case_to_running", boom_running
    )
    with pytest.raises(RuntimeError, match="running failure"):
        InvestigationRuntime.from_durable_fixture(
            case=make_case(case_id="case-l03"),
            scenario=ScenarioConfig(name="l03", tool_scripts={}),
            domain_db=db,
        )
    # admit is idempotent; running still fails — Case remains created.
    assert repo.get_case(case.case_id).status is CaseStatus.CREATED
    db.close()


def test_f1_admission_requires_explicit_effective_scope(tmp_path: Path):
    db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    repo = InvestigationRepository(db)
    case = make_case(case_id="case-f1-scope")
    # Fixture helper documents trusted scope; admission still enforces containment.
    assert case.effective_scope.camera_ids
    assert set(case.effective_scope.camera_ids).issubset(set(case.requested_scope.camera_ids))
    admitted = repo.admit_case(case)
    assert admitted.status is CaseStatus.CREATED
    doc = InvestigationRuntime.from_durable_fixture.__doc__ or ""
    assert "Not a production" in doc
    assert "not live VMS" in doc
    assert "authorization" in doc
    db.close()


def test_l04_duplicate_and_conflicting_admission(tmp_path: Path):
    db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    repo = InvestigationRepository(db)
    case = make_case(case_id="case-l04")
    first = repo.admit_case(case)
    second = repo.admit_case(case)
    assert first.status is CaseStatus.CREATED
    assert second.status is CaseStatus.CREATED
    assert second.case_id == first.case_id

    conflict = make_case(case_id="case-l04", question="Different question entirely.")
    with pytest.raises(IdempotencyConflictError, match="different content"):
        repo.admit_case(conflict)

    running = repo.transition_case_to_running(case.case_id)
    assert running.status is CaseStatus.RUNNING
    # Re-admission of equivalent content while running is safe/idempotent.
    again = repo.admit_case(case)
    assert again.status is CaseStatus.RUNNING
    db.close()
