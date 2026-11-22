"""Native SqliteSaver and Investigation Persistence round-trips. G4-V1-EA."""

from __future__ import annotations

import importlib.metadata
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict

import pytest
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from nanexus_investigation.contracts.case import InvestigationCase
from nanexus_investigation.contracts.enums import (
    CaseStatus,
    EvidenceAvailability,
    EvidenceKind,
    ObservationType,
    RecordType,
    SearchCoverage,
)
from nanexus_investigation.contracts.evidence import EvidenceRef, Observation
from nanexus_investigation.contracts.identity import Interval, SourceLocator, TimeContext
from nanexus_investigation.contracts.scope import InvestigationScope
from nanexus_investigation.contracts.search import SearchExecutionReceipt
from nanexus_investigation.persistence.db import InvestigationDatabase
from nanexus_investigation.persistence.errors import IdempotencyConflictError
from nanexus_investigation.persistence.operation_id import build_operation_identity
from nanexus_investigation.persistence.repository import InvestigationRepository
from nanexus_investigation.persistence.schema import APP_TABLE_PREFIX, CHECKPOINT_TABLES

UTC = timezone.utc
T0 = datetime(2026, 11, 19, 20, 0, tzinfo=UTC)
T1 = datetime(2026, 11, 19, 23, 0, tzinfo=UTC)


class _CounterState(TypedDict):
    n: int
    label: str


def _build_mini_graph():
    def bump(state: _CounterState) -> dict:
        return {"n": state["n"] + 1, "label": state["label"]}

    graph = StateGraph(_CounterState)
    graph.add_node("bump", bump)
    graph.add_edge(START, "bump")
    graph.add_edge("bump", END)
    return graph


def test_native_sqlite_saver_package_is_available():
    assert importlib.metadata.version("langgraph-checkpoint-sqlite") == "3.1.3"
    assert importlib.metadata.version("langgraph-checkpoint") == "4.3.2"
    assert importlib.metadata.version("sqlite-vec") == "0.1.9"
    assert SqliteSaver.__module__.startswith("langgraph.checkpoint.sqlite")


def test_sqlite_saver_roundtrip_survives_connection_recreation(tmp_path: Path):
    ckpt_path = tmp_path / "langgraph_checkpoints.sqlite"
    assert not str(ckpt_path).startswith("/media/")  # disposable test-local only

    graph = _build_mini_graph()
    conn = sqlite3.connect(str(ckpt_path), check_same_thread=False)
    saver = SqliteSaver(conn)
    assert type(saver).__name__ == "SqliteSaver"
    saver.setup()
    app = graph.compile(checkpointer=saver)

    config = {"configurable": {"thread_id": "case-thread-1"}}
    result = app.invoke(
        {"n": 0, "label": "alpha"},
        config,
        durability="sync",
    )
    assert result == {"n": 1, "label": "alpha"}
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert "checkpoints" in tables
    assert "writes" in tables
    conn.close()

    conn2 = sqlite3.connect(str(ckpt_path), check_same_thread=False)
    saver2 = SqliteSaver(conn2)
    saver2.setup()
    app2 = graph.compile(checkpointer=saver2)
    recovered = app2.get_state(config)
    assert recovered.values == {"n": 1, "label": "alpha"}

    other = app2.get_state({"configurable": {"thread_id": "case-thread-2"}})
    assert other.values == {} or other.values.get("n") is None
    conn2.close()


def test_checkpoint_and_domain_databases_are_separate(tmp_path: Path):
    ckpt_path = tmp_path / "ckpt.sqlite"
    domain_path = tmp_path / "investigation.sqlite"

    conn = sqlite3.connect(str(ckpt_path), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    ckpt_tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    conn.close()

    db = InvestigationDatabase.open(domain_path)
    domain_tables = db.table_names()
    db.close()

    assert ckpt_tables & domain_tables == set()
    assert all(name.startswith(APP_TABLE_PREFIX) or name == "sqlite_sequence" for name in domain_tables if name != "sqlite_sequence")
    assert "nx_cases" in domain_tables
    assert "checkpoints" not in domain_tables
    # Native saver tables remain only in the checkpoint file.
    assert "checkpoints" in ckpt_tables
    assert CHECKPOINT_TABLES & domain_tables == set()


def _sample_case(case_id: str = "case-ea-1") -> InvestigationCase:
    scope = InvestigationScope(
        vms_instance_id="vms-local-1",
        camera_ids=("front_yard",),
        time_bounds=Interval(start_utc=T0, end_utc=T1),
    )
    return InvestigationCase(
        case_id=case_id,
        question="Find person activity.",
        requested_scope=scope,
        effective_scope=scope,
        time_context=TimeContext(
            timezone="America/New_York",
            start_utc=T0,
            end_utc=T1,
        ),
        status=CaseStatus.CREATED,
        policy_ref="policy:e0",
        created_at=datetime(2026, 11, 20, 12, 0, tzinfo=UTC),
        budget_profile_ref="budget:ea",
    )


def test_domain_record_survives_new_connection(tmp_path: Path):
    path = tmp_path / "investigation.sqlite"
    case = _sample_case()
    db = InvestigationDatabase.open(path)
    repo = InvestigationRepository(db)
    repo.upsert_case(case)

    op = build_operation_identity(
        case_id=case.case_id,
        plan_step_id="step-1",
        tool_name="search_events",
        vms_instance_id="vms-local-1",
        request_parameters={"label": "person", "camera_ids": ["front_yard"]},
    )
    observation = Observation(
        observation_id="obs-1",
        source_locator=SourceLocator(
            vms_instance_id="vms-local-1",
            record_type=RecordType.EVENT,
            native_record_id="evt-1",
        ),
        source_revision="rev-1",
        observation_type=ObservationType.DETECTION_EVENT,
        camera_id="front_yard",
        start_utc=T0,
        end_utc=T0.replace(minute=5),
        retrieved_at=T0,
        provenance="stub:search_events",
    )
    evidence = EvidenceRef(
        evidence_id="ev-1",
        source_locator=observation.source_locator,
        evidence_kind=EvidenceKind.EVENT_METADATA,
        camera_id="front_yard",
        interval=Interval(start_utc=T0, end_utc=T0.replace(minute=5)),
        retrieval_spec={"operation": "search_events"},
        availability=EvidenceAvailability.AVAILABLE,
        provenance="stub:search_events",
    )
    receipt = SearchExecutionReceipt(
        search_execution_id="receipt-1",
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
    repo.put_observation(observation, operation=op)
    repo.put_evidence(evidence, operation=op)
    # Receipt needs its own operation identity to avoid UNIQUE (operation_id, kind) clash
    # when multiple artifact kinds share one operation — schema allows one kind per op.
    # Observation and evidence share op but different kinds — OK.
    # Putting second evidence with same op would conflict on kind uniqueness for evidence_ref.
    op_receipt = build_operation_identity(
        case_id=case.case_id,
        plan_step_id="step-1-receipt",
        tool_name="search_events",
        vms_instance_id="vms-local-1",
        request_parameters={"label": "person", "kind": "receipt"},
    )
    repo.put_search_receipt(receipt, operation=op_receipt)
    db.close()

    db2 = InvestigationDatabase.open(path)
    repo2 = InvestigationRepository(db2)
    loaded_case = repo2.get_case(case.case_id)
    assert loaded_case is not None
    assert loaded_case.case_id == case.case_id
    assert loaded_case.effective_scope.vms_instance_id == "vms-local-1"
    loaded_obs = repo2.get_observation("obs-1")
    assert loaded_obs is not None
    assert loaded_obs.source_locator.native_record_id == "evt-1"
    loaded_ev = repo2.get_evidence("ev-1")
    assert loaded_ev is not None
    assert loaded_ev.source_locator == observation.source_locator
    loaded_receipt = repo2.get_search_receipt("receipt-1")
    assert loaded_receipt is not None
    assert loaded_receipt.coverage is SearchCoverage.EXHAUSTED
    db2.close()


def test_conflicting_case_content_rejected(tmp_path: Path):
    path = tmp_path / "investigation.sqlite"
    db = InvestigationDatabase.open(path)
    repo = InvestigationRepository(db)
    case = _sample_case()
    repo.upsert_case(case)
    conflict = case.model_copy(update={"question": "Different question."})
    with pytest.raises(IdempotencyConflictError):
        repo.upsert_case(conflict)
    db.close()


def test_no_credentials_or_media_in_payload(tmp_path: Path):
    path = tmp_path / "investigation.sqlite"
    db = InvestigationDatabase.open(path)
    repo = InvestigationRepository(db)
    case = _sample_case()
    repo.upsert_case(case)
    op = build_operation_identity(
        case_id=case.case_id,
        plan_step_id="bad",
        tool_name="search_events",
        vms_instance_id="vms-local-1",
        request_parameters={},
    )
    # Build a valid Observation then attempt raw insert with forbidden keys via put_artifact
    observation = Observation(
        observation_id="obs-bad",
        source_locator=SourceLocator(
            vms_instance_id="vms-local-1",
            record_type=RecordType.EVENT,
            native_record_id="evt-bad",
        ),
        source_revision="rev-1",
        observation_type=ObservationType.DETECTION_EVENT,
        camera_id="front_yard",
        start_utc=T0,
        end_utc=T0.replace(minute=1),
        retrieved_at=T0,
        provenance="stub",
        attributes={"note": "ok"},
    )
    # attributes cannot hold credentials through contract easily; test helper directly
    from nanexus_investigation.persistence.repository import _forbidden_payload_keys

    with pytest.raises(ValueError, match="secrets or media"):
        _forbidden_payload_keys({"api_key": "x"})
    with pytest.raises(ValueError, match="raw bytes"):
        _forbidden_payload_keys({"blob": b"abc"})
    repo.put_observation(observation, operation=op)
    db.close()
