"""G4-V1-EB2A: atomic domain artifact registration groups."""

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
    ToolErrorCode,
    ToolResultStatus,
)
from nanexus_investigation.contracts.evidence import EvidenceRef, Observation
from nanexus_investigation.contracts.identity import Interval, SourceLocator
from nanexus_investigation.contracts.search import SearchExecutionReceipt
from nanexus_investigation.contracts.tool import ToolError, ToolResult
from nanexus_investigation.graph.runtime import (
    DomainRegistrationError,
    ScenarioConfig,
    empty_result,
    make_case,
    ok_event_result,
    register_tool_artifacts,
)
from nanexus_investigation.graph.workflow import InvestigationRuntime, run_investigation
from nanexus_investigation.persistence.budget import DurableBudgetLedger, ReservationStatus
from nanexus_investigation.persistence.db import InvestigationDatabase
from nanexus_investigation.persistence.errors import IdempotencyConflictError
from nanexus_investigation.persistence.operation_id import build_operation_identity
from nanexus_investigation.persistence.repository import InvestigationRepository
from nanexus_investigation.persistence.schema import SCHEMA_VERSION
from nanexus_investigation.contracts.tool import ToolRequest

UTC = timezone.utc
T0 = datetime(2026, 11, 19, 20, 0, tzinfo=UTC)
T1 = datetime(2026, 11, 19, 23, 0, tzinfo=UTC)


def _case_and_repo(tmp_path: Path, case_id: str = "case-atomic-1"):
    db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    case = make_case(case_id=case_id)
    if case.status.value == "created":
        case = case.transition_to(CaseStatus.RUNNING)
    repo = InvestigationRepository(db)
    repo.upsert_case(case)
    return db, repo, case


def _op(case_id: str, step: str = "step-1"):
    return build_operation_identity(
        case_id=case_id,
        plan_step_id=step,
        tool_name="search_events",
        vms_instance_id="vms-local-1",
        request_parameters={"label": "person"},
    )


def _receipt(execution_id: str = "exec-1", match_count: int = 1) -> SearchExecutionReceipt:
    return SearchExecutionReceipt(
        search_execution_id=f"receipt-{execution_id}",
        tool_name="search_events",
        source_id="vms-local-1",
        camera_ids=("front_yard",),
        time_bounds=Interval(start_utc=T0, end_utc=T1),
        filters={"label": "person"},
        retrieved_at=T0,
        pages_fetched=1,
        match_count=match_count,
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


def _counts(db: InvestigationDatabase, operation_id: str) -> dict[str, int]:
    artifacts = db.conn.execute(
        "SELECT COUNT(*) AS c FROM nx_artifacts WHERE operation_id = ?",
        (operation_id,),
    ).fetchone()["c"]
    groups = db.conn.execute(
        "SELECT COUNT(*) AS c FROM nx_registration_groups WHERE operation_id = ?",
        (operation_id,),
    ).fetchone()["c"]
    ops = db.conn.execute(
        "SELECT COUNT(*) AS c FROM nx_operations WHERE operation_id = ?",
        (operation_id,),
    ).fetchone()["c"]
    return {"artifacts": artifacts, "groups": groups, "operations": ops}


def test_a1_successful_group_survives_reopen(tmp_path: Path):
    db, repo, case = _case_and_repo(tmp_path)
    op = _op(case.case_id)
    group = repo.register_artifact_group(
        operation=op,
        receipt=_receipt(),
        observation=_observation(),
        evidence=_evidence(),
    )
    assert group.reused is False
    assert group.observation_id == "obs-1"
    assert group.evidence_id == "ev-1"
    db.close()

    db2 = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    repo2 = InvestigationRepository(db2)
    loaded = repo2.get_registration_group(op.operation_id)
    assert loaded is not None
    assert loaded.receipt.search_execution_id == "receipt-exec-1"
    assert loaded.observation is not None
    assert loaded.evidence is not None
    assert loaded.observation.source_locator.native_record_id == "evt-1"
    assert loaded.evidence.source_locator.native_record_id == "evt-1"
    assert int(
        db2.conn.execute(
            "SELECT value FROM nx_schema_meta WHERE key='schema_version'"
        ).fetchone()["value"]
    ) == SCHEMA_VERSION
    db2.close()


def test_a2_failure_before_first_write_commits_nothing(tmp_path: Path):
    db, repo, case = _case_and_repo(tmp_path, "case-a2")
    op = _op(case.case_id)

    def boom():
        raise RuntimeError("injected before_first_write")

    repo.set_fault_hook("before_first_write", boom)
    with pytest.raises(RuntimeError, match="before_first_write"):
        repo.register_artifact_group(
            operation=op,
            receipt=_receipt("a2"),
            observation=_observation("obs-a2"),
            evidence=_evidence("ev-a2"),
        )
    assert _counts(db, op.operation_id) == {
        "artifacts": 0,
        "groups": 0,
        "operations": 0,
    }
    db.close()


def test_a3_failure_after_observation_rolls_back_group(tmp_path: Path):
    db, repo, case = _case_and_repo(tmp_path, "case-a3")
    op = _op(case.case_id)

    def boom():
        raise RuntimeError("injected after_observation")

    repo.set_fault_hook("after_observation", boom)
    with pytest.raises(RuntimeError, match="after_observation"):
        repo.register_artifact_group(
            operation=op,
            receipt=_receipt("a3"),
            observation=_observation("obs-a3"),
            evidence=_evidence("ev-a3"),
        )
    assert _counts(db, op.operation_id) == {
        "artifacts": 0,
        "groups": 0,
        "operations": 0,
    }
    db.close()


def test_a4_failure_after_evidence_ref_rolls_back_group(tmp_path: Path):
    db, repo, case = _case_and_repo(tmp_path, "case-a4")
    op = _op(case.case_id)

    def boom():
        raise RuntimeError("injected after_evidence")

    repo.set_fault_hook("after_evidence", boom)
    with pytest.raises(RuntimeError, match="after_evidence"):
        repo.register_artifact_group(
            operation=op,
            receipt=_receipt("a4"),
            observation=_observation("obs-a4"),
            evidence=_evidence("ev-a4"),
        )
    assert _counts(db, op.operation_id) == {
        "artifacts": 0,
        "groups": 0,
        "operations": 0,
    }
    db.close()


def test_a5_failure_before_final_commit_rolls_back(tmp_path: Path):
    db, repo, case = _case_and_repo(tmp_path, "case-a5")
    op = _op(case.case_id)

    def boom():
        raise RuntimeError("injected before_commit")

    repo.set_fault_hook("before_commit", boom)
    with pytest.raises(RuntimeError, match="before_commit"):
        repo.register_artifact_group(
            operation=op,
            receipt=_receipt("a5"),
            observation=_observation("obs-a5"),
            evidence=_evidence("ev-a5"),
        )
    assert _counts(db, op.operation_id) == {
        "artifacts": 0,
        "groups": 0,
        "operations": 0,
    }
    db.close()


def test_a6_same_operation_replay_returns_original_ids(tmp_path: Path):
    db, repo, case = _case_and_repo(tmp_path, "case-a6")
    op = _op(case.case_id)
    first = repo.register_artifact_group(
        operation=op,
        receipt=_receipt("a6"),
        observation=_observation("obs-a6"),
        evidence=_evidence("ev-a6"),
    )
    second = repo.register_artifact_group(
        operation=op,
        receipt=_receipt("a6"),
        observation=_observation("obs-a6"),
        evidence=_evidence("ev-a6"),
    )
    assert second.reused is True
    assert second.receipt_id == first.receipt_id
    assert second.observation_id == first.observation_id
    assert second.evidence_id == first.evidence_id
    assert _counts(db, op.operation_id)["artifacts"] == 3
    assert _counts(db, op.operation_id)["groups"] == 1
    db.close()


def test_a7_conflicting_replay_rejected(tmp_path: Path):
    db, repo, case = _case_and_repo(tmp_path, "case-a7")
    op = _op(case.case_id)
    original = repo.register_artifact_group(
        operation=op,
        receipt=_receipt("a7"),
        observation=_observation("obs-a7", "evt-a7"),
        evidence=_evidence("ev-a7", "evt-a7"),
    )
    with pytest.raises(IdempotencyConflictError):
        repo.register_artifact_group(
            operation=op,
            receipt=_receipt("a7"),
            observation=_observation("obs-a7", "evt-OTHER"),
            evidence=_evidence("ev-a7", "evt-OTHER"),
        )
    loaded = repo.get_registration_group(op.operation_id)
    assert loaded is not None
    assert loaded.observation is not None
    assert loaded.observation.source_locator.native_record_id == "evt-a7"
    assert loaded.group_fingerprint == original.group_fingerprint
    db.close()


def test_a8_empty_error_unsupported_no_fabricated_observations(tmp_path: Path):
    db, repo, case = _case_and_repo(tmp_path, "case-a8")
    for label, receipt_kwargs, status in (
        ("empty", {"execution_id": "empty", "match_count": 0}, ToolResultStatus.EMPTY),
        ("error", {"execution_id": "error", "match_count": 0}, ToolResultStatus.ERROR),
        (
            "unsupported",
            {"execution_id": "unsup", "match_count": 0},
            ToolResultStatus.UNSUPPORTED,
        ),
    ):
        op = _op(case.case_id, step=f"step-{label}")
        receipt = _receipt(**receipt_kwargs)
        if status is ToolResultStatus.ERROR:
            receipt = receipt.model_copy(
                update={
                    "coverage": SearchCoverage.UNKNOWN,
                    "error": ToolError(
                        code=ToolErrorCode.SOURCE_UNAVAILABLE, message="unavailable"
                    ),
                }
            )
        if status is ToolResultStatus.UNSUPPORTED:
            receipt = receipt.model_copy(
                update={
                    "coverage": SearchCoverage.UNKNOWN,
                    "error": ToolError(
                        code=ToolErrorCode.DISABLED_CAPABILITY, message="disabled"
                    ),
                }
            )
        group = repo.register_artifact_group(operation=op, receipt=receipt)
        assert group.observation is None
        assert group.evidence is None
        assert group.receipt_id.startswith("receipt-")
        obs = db.conn.execute(
            """
            SELECT COUNT(*) AS c FROM nx_artifacts
            WHERE operation_id = ? AND artifact_kind = 'observation'
            """,
            (op.operation_id,),
        ).fetchone()["c"]
        assert obs == 0
    db.close()


def test_a9_graph_failure_boundary_no_published_ids(tmp_path: Path):
    domain_db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    scenario = ScenarioConfig(
        name="a9",
        tool_scripts={
            "search_events": [
                ok_event_result(
                    execution_id="exec-a9",
                    native_id="evt-a9",
                    label="person",
                    start=T0 + timedelta(minutes=1),
                    end=T0 + timedelta(minutes=2),
                )
            ]
        },
    )
    runtime = InvestigationRuntime.from_durable_fixture(
        case=make_case(case_id="case-a9"),
        scenario=scenario,
        domain_db=domain_db,
    )
    assert runtime.domain_repo is not None
    runtime.domain_repo.set_fault_hook(
        "after_observation",
        lambda: (_ for _ in ()).throw(RuntimeError("injected graph registration fault")),
    )
    conn = sqlite3.connect(str(tmp_path / "ckpt.sqlite"), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    result = run_investigation(
        runtime, thread_id="t-a9", checkpointer=saver, durability="sync"
    )
    assert result["state"].get("evidence_ids") in (None, [])
    assert result["state"].get("observation_ids") in (None, [])
    assert result["report"] is None
    assert runtime.case.status is CaseStatus.FAILED
    assert "ev-exec-a9-0" not in runtime.store.evidence
    # No committed registration group for the failed operation.
    groups = domain_db.conn.execute(
        "SELECT COUNT(*) AS c FROM nx_registration_groups"
    ).fetchone()["c"]
    assert groups == 0
    artifacts = domain_db.conn.execute(
        "SELECT COUNT(*) AS c FROM nx_artifacts"
    ).fetchone()["c"]
    assert artifacts == 0
    conn.close()
    domain_db.close()


def test_a10_durable_two_step_and_in_memory_still_pass(tmp_path: Path):
    # In-memory path
    mem = InvestigationRuntime.from_fixture(
        case=make_case(case_id="case-a10-mem"),
        scenario=ScenarioConfig(
            name="a10_mem",
            include_follow_up=True,
            tool_scripts={
                "search_events": [
                    ok_event_result(
                        execution_id="mem-p",
                        native_id="evt-p",
                        label="person",
                        start=T0 + timedelta(minutes=1),
                        end=T0 + timedelta(minutes=2),
                    )
                ],
                "search_related_events": [
                    ok_event_result(
                        execution_id="mem-v",
                        native_id="evt-v",
                        label="vehicle",
                        start=T0 + timedelta(minutes=5),
                        end=T0 + timedelta(minutes=6),
                    )
                ],
            },
        ),
    )
    mem_result = run_investigation(mem, thread_id="t-a10-mem")
    assert mem_result["report"] is not None
    assert mem.registry.invocations == ["search_events", "search_related_events"]

    # Durable two-step
    domain_db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    durable = InvestigationRuntime.from_durable_fixture(
        case=make_case(case_id="case-a10-dur"),
        scenario=ScenarioConfig(
            name="a10_dur",
            include_follow_up=True,
            tool_scripts={
                "search_events": [
                    ok_event_result(
                        execution_id="dur-p",
                        native_id="evt-p",
                        label="person",
                        start=T0 + timedelta(minutes=1),
                        end=T0 + timedelta(minutes=2),
                    )
                ],
                "search_related_events": [
                    ok_event_result(
                        execution_id="dur-v",
                        native_id="evt-v",
                        label="vehicle",
                        start=T0 + timedelta(minutes=5),
                        end=T0 + timedelta(minutes=6),
                    )
                ],
            },
        ),
        domain_db=domain_db,
    )
    conn = sqlite3.connect(str(tmp_path / "ckpt.sqlite"), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    dur_result = run_investigation(
        durable, thread_id="t-a10-dur", checkpointer=saver, durability="sync"
    )
    assert dur_result["report"] is not None
    assert durable.case.status in {CaseStatus.COMPLETED, CaseStatus.PARTIAL}
    groups = domain_db.conn.execute(
        "SELECT COUNT(*) AS c FROM nx_registration_groups"
    ).fetchone()["c"]
    assert groups == 2
    conn.close()
    domain_db.close()


def test_budget_reservation_survives_registration_rollback(tmp_path: Path):
    domain_db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    case = make_case(case_id="case-budget-atomic")
    runtime = InvestigationRuntime.from_durable_fixture(
        case=case,
        scenario=ScenarioConfig(
            name="budget_atomic",
            max_tool_attempts=2,
            tool_scripts={
                "search_events": [
                    ok_event_result(
                        execution_id="exec-bud",
                        native_id="evt-bud",
                        label="person",
                        start=T0 + timedelta(minutes=1),
                        end=T0 + timedelta(minutes=2),
                    )
                ]
            },
        ),
        domain_db=domain_db,
    )
    assert runtime.domain_repo is not None
    runtime.domain_repo.set_fault_hook(
        "after_receipt",
        lambda: (_ for _ in ()).throw(RuntimeError("registration rollback")),
    )
    conn = sqlite3.connect(str(tmp_path / "ckpt.sqlite"), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    result = run_investigation(
        runtime, thread_id="t-budget-atomic", checkpointer=saver, durability="sync"
    )
    assert result["report"] is None
    assert runtime.case.status is CaseStatus.FAILED
    # Tool ran (reservation confirmed) but registration rolled back.
    assert runtime.registry.invocations == ["search_events"]
    ledger = DurableBudgetLedger(domain_db)
    assert ledger.reserved_units(runtime.case.case_id) == 1
    reservation = ledger.get_reservation(f"req-{runtime.case.case_id}-step-1-0")
    assert reservation is not None
    assert reservation.status is ReservationStatus.CONFIRMED
    assert reservation.is_confirmed_physical_attempt
    # No refund and no second dispatch.
    assert domain_db.conn.execute(
        "SELECT COUNT(*) AS c FROM nx_registration_groups"
    ).fetchone()["c"] == 0
    conn.close()
    domain_db.close()


def test_schema_v1_database_upgrades_additively(tmp_path: Path):
    """Opening a v1-era file applies registration_groups without destroying cases."""
    path = tmp_path / "legacy.sqlite"
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE nx_schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO nx_schema_meta(key, value) VALUES ('schema_version', '1');
        CREATE TABLE nx_cases (
            case_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            question TEXT NOT NULL,
            policy_ref TEXT NOT NULL,
            created_at TEXT NOT NULL,
            vms_instance_id TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            payload_fingerprint TEXT NOT NULL,
            schema_version INTEGER NOT NULL DEFAULT 1
        );
        """
    )
    conn.execute(
        """
        INSERT INTO nx_cases VALUES (
            'legacy-case', 'running', 'q', 'policy', '2026-01-01T00:00:00+00:00',
            'vms-local-1', '{}', 'fp', 1
        )
        """
    )
    conn.commit()
    conn.close()

    db = InvestigationDatabase.open(path)
    assert "nx_registration_groups" in db.table_names()
    assert (
        db.conn.execute(
            "SELECT value FROM nx_schema_meta WHERE key='schema_version'"
        ).fetchone()["value"]
        == str(SCHEMA_VERSION)
    )
    assert (
        db.conn.execute(
            "SELECT case_id FROM nx_cases WHERE case_id='legacy-case'"
        ).fetchone()["case_id"]
        == "legacy-case"
    )
    # Reopen must not destroy upgraded data or rewrite version destructively.
    db.close()
    db2 = InvestigationDatabase.open(path)
    assert (
        db2.conn.execute(
            "SELECT value FROM nx_schema_meta WHERE key='schema_version'"
        ).fetchone()["value"]
        == str(SCHEMA_VERSION)
    )
    assert (
        db2.conn.execute(
            "SELECT case_id FROM nx_cases WHERE case_id='legacy-case'"
        ).fetchone()["case_id"]
        == "legacy-case"
    )
    db2.close()


def test_schema_v2_upgrades_to_v3_preserving_single_event_group(tmp_path: Path):
    """v2 UNIQUE(operation_id, artifact_kind) DBs migrate to v3 without data loss."""
    from nanexus_investigation.persistence.repository import (
        _legacy_single_artifact_group_fingerprint,
    )

    path = tmp_path / "v2.sqlite"
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE nx_schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO nx_schema_meta(key, value) VALUES ('schema_version', '2');
        CREATE TABLE nx_cases (
            case_id TEXT PRIMARY KEY NOT NULL,
            status TEXT NOT NULL,
            question TEXT NOT NULL,
            policy_ref TEXT NOT NULL,
            created_at TEXT NOT NULL,
            vms_instance_id TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            payload_fingerprint TEXT NOT NULL,
            schema_version INTEGER NOT NULL DEFAULT 2
        );
        CREATE TABLE nx_operations (
            operation_id TEXT PRIMARY KEY NOT NULL,
            case_id TEXT NOT NULL REFERENCES nx_cases(case_id),
            plan_step_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            content_fingerprint TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE (case_id, plan_step_id, content_fingerprint)
        );
        CREATE TABLE nx_artifacts (
            artifact_kind TEXT NOT NULL,
            artifact_id TEXT NOT NULL,
            operation_id TEXT NOT NULL REFERENCES nx_operations(operation_id),
            case_id TEXT NOT NULL REFERENCES nx_cases(case_id),
            payload_json TEXT NOT NULL,
            payload_fingerprint TEXT NOT NULL,
            native_record_id TEXT,
            vms_instance_id TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (artifact_kind, artifact_id),
            UNIQUE (operation_id, artifact_kind)
        );
        CREATE TABLE nx_registration_groups (
            operation_id TEXT PRIMARY KEY NOT NULL REFERENCES nx_operations(operation_id),
            case_id TEXT NOT NULL REFERENCES nx_cases(case_id),
            group_fingerprint TEXT NOT NULL,
            member_kinds TEXT NOT NULL,
            receipt_id TEXT NOT NULL,
            observation_id TEXT,
            evidence_id TEXT,
            status TEXT NOT NULL CHECK (status = 'committed'),
            created_at TEXT NOT NULL
        );
        CREATE TABLE nx_budget_policies (
            case_id TEXT PRIMARY KEY NOT NULL REFERENCES nx_cases(case_id),
            max_tool_attempts INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE nx_budget_reservations (
            reservation_id TEXT PRIMARY KEY NOT NULL,
            case_id TEXT NOT NULL REFERENCES nx_cases(case_id),
            operation_id TEXT NOT NULL,
            attempt_id TEXT NOT NULL,
            units INTEGER NOT NULL,
            status TEXT NOT NULL,
            payload_fingerprint TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            execution_id TEXT
        );
        """
    )
    case = make_case(case_id="case-v2-upgrade")
    case = case.transition_to(CaseStatus.RUNNING)
    op = _op(case.case_id, step="step-v2")
    receipt = _receipt("v2")
    observation = _observation("obs-v2", "evt-v2")
    evidence = _evidence("ev-v2", "evt-v2")
    legacy_fp = _legacy_single_artifact_group_fingerprint(
        operation_id=op.operation_id,
        receipt=receipt,
        observation=observation,
        evidence=evidence,
    )
    from nanexus_investigation.persistence.operation_id import payload_fingerprint
    import json

    def _insert_artifact(kind: str, artifact_id: str, model, native: str | None):
        payload = model.model_dump(mode="json")
        conn.execute(
            """
            INSERT INTO nx_artifacts(
                artifact_kind, artifact_id, operation_id, case_id,
                payload_json, payload_fingerprint, native_record_id,
                vms_instance_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                kind,
                artifact_id,
                op.operation_id,
                case.case_id,
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
                payload_fingerprint(payload),
                native,
                "vms-local-1",
                T0.isoformat(),
            ),
        )

    conn.execute(
        """
        INSERT INTO nx_cases(
            case_id, status, question, policy_ref, created_at, vms_instance_id,
            payload_json, payload_fingerprint, schema_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 2)
        """,
        (
            case.case_id,
            case.status.value,
            case.question,
            case.policy_ref,
            case.created_at.isoformat(),
            case.effective_scope.vms_instance_id,
            case.model_dump_json(),
            "case-fp",
        ),
    )
    conn.execute(
        """
        INSERT INTO nx_operations(
            operation_id, case_id, plan_step_id, tool_name,
            content_fingerprint, created_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            op.operation_id,
            case.case_id,
            op.plan_step_id,
            op.tool_name,
            op.content_fingerprint,
            T0.isoformat(),
        ),
    )
    _insert_artifact("search_receipt", receipt.search_execution_id, receipt, None)
    _insert_artifact(
        "observation",
        observation.observation_id,
        observation,
        observation.source_locator.native_record_id,
    )
    _insert_artifact(
        "evidence_ref",
        evidence.evidence_id,
        evidence,
        evidence.source_locator.native_record_id,
    )
    conn.execute(
        """
        INSERT INTO nx_registration_groups(
            operation_id, case_id, group_fingerprint, member_kinds,
            receipt_id, observation_id, evidence_id, status, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, 'committed', ?)
        """,
        (
            op.operation_id,
            case.case_id,
            legacy_fp,
            json.dumps(
                ["search_receipt", "observation", "evidence_ref"],
                separators=(",", ":"),
            ),
            receipt.search_execution_id,
            observation.observation_id,
            evidence.evidence_id,
            T0.isoformat(),
        ),
    )
    conn.commit()
    conn.close()

    db = InvestigationDatabase.open(path)
    assert (
        db.conn.execute(
            "SELECT value FROM nx_schema_meta WHERE key='schema_version'"
        ).fetchone()["value"]
        == "3"
    )
    ddl = db.conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='nx_artifacts'"
    ).fetchone()["sql"]
    assert "UNIQUE (operation_id, artifact_kind)" not in " ".join(ddl.split())
    repo = InvestigationRepository(db)
    loaded = repo.get_registration_group(op.operation_id)
    assert loaded is not None
    assert loaded.reused is True
    assert loaded.observation_id == "obs-v2"
    assert loaded.evidence_id == "ev-v2"
    assert loaded.observation is not None
    assert loaded.observation.source_locator.native_record_id == "evt-v2"

    # After upgrade, a new multi-event group can persist both pairs.
    multi_op = _op(case.case_id, step="step-multi")
    multi = repo.register_artifact_group(
        operation=multi_op,
        receipt=_receipt("multi"),
        observations=(
            _observation("obs-m1", "evt-m1"),
            _observation("obs-m2", "evt-m2"),
        ),
        evidences=(
            _evidence("ev-m1", "evt-m1"),
            _evidence("ev-m2", "evt-m2"),
        ),
    )
    assert len(multi.observations) == 2
    assert len(multi.evidences) == 2
    assert (
        db.conn.execute(
            "SELECT COUNT(*) AS c FROM nx_artifacts WHERE operation_id = ? "
            "AND artifact_kind = 'observation'",
            (multi_op.operation_id,),
        ).fetchone()["c"]
        == 2
    )
    db.close()


def test_fresh_database_initializes_at_schema_version_3(tmp_path: Path):
    db = InvestigationDatabase.open(tmp_path / "fresh.sqlite")
    assert (
        db.conn.execute(
            "SELECT value FROM nx_schema_meta WHERE key='schema_version'"
        ).fetchone()["value"]
        == str(SCHEMA_VERSION)
    )
    assert SCHEMA_VERSION == 3
    assert "nx_registration_groups" in db.table_names()
    # Multi-event groups require no UNIQUE(operation_id, artifact_kind).
    ddl = db.conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='nx_artifacts'"
    ).fetchone()["sql"]
    assert "UNIQUE (operation_id, artifact_kind)" not in " ".join(ddl.split())
    db.close()


def test_unsupported_schema_version_is_rejected(tmp_path: Path):
    path = tmp_path / "future.sqlite"
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE nx_schema_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO nx_schema_meta(key, value) VALUES ('schema_version', '99');
        """
    )
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError, match="unsupported schema_version=99"):
        InvestigationDatabase.open(path)


def test_orphan_artifacts_without_group_are_not_complete(tmp_path: Path):
    db, repo, case = _case_and_repo(tmp_path, "case-orphan")
    op = _op(case.case_id, step="step-orphan")
    repo.ensure_operation(op)
    repo.put_search_receipt(_receipt("orphan"), operation=op)
    assert _counts(db, op.operation_id)["artifacts"] == 1
    assert _counts(db, op.operation_id)["groups"] == 0
    assert repo.get_registration_group(op.operation_id) is None
    with pytest.raises(IdempotencyConflictError, match="incomplete artifacts"):
        repo.register_artifact_group(
            operation=op,
            receipt=_receipt("orphan"),
            observation=_observation("obs-orphan"),
            evidence=_evidence("ev-orphan"),
        )
    db.close()


def test_incomplete_group_marker_rejected_on_load(tmp_path: Path):
    db, repo, case = _case_and_repo(tmp_path, "case-incomplete-marker")
    op = _op(case.case_id, step="step-incomplete")
    repo.ensure_operation(op)
    with db.transaction() as conn:
        conn.execute(
            """
            INSERT INTO nx_registration_groups(
                operation_id, case_id, group_fingerprint, member_kinds,
                receipt_id, observation_id, evidence_id, status, created_at
            ) VALUES (?, ?, 'bogus-fp', '["search_receipt"]', 'receipt-missing',
                      NULL, NULL, 'committed', ?)
            """,
            (op.operation_id, case.case_id, T0.isoformat()),
        )
    with pytest.raises(IdempotencyConflictError, match="missing search receipt"):
        repo.get_registration_group(op.operation_id)
    db.close()


def test_distinct_operations_create_distinct_groups(tmp_path: Path):
    db, repo, case = _case_and_repo(tmp_path, "case-distinct")
    op1 = _op(case.case_id, step="step-a")
    op2 = _op(case.case_id, step="step-b")
    g1 = repo.register_artifact_group(
        operation=op1,
        receipt=_receipt("da"),
        observation=_observation("obs-da"),
        evidence=_evidence("ev-da"),
    )
    g2 = repo.register_artifact_group(
        operation=op2,
        receipt=_receipt("db"),
        observation=_observation("obs-db"),
        evidence=_evidence("ev-db"),
    )
    assert g1.operation_id != g2.operation_id
    assert g1.receipt_id != g2.receipt_id
    assert _counts(db, op1.operation_id)["groups"] == 1
    assert _counts(db, op2.operation_id)["groups"] == 1
    db.close()


def test_post_failure_txn_conn_cleared_and_reusable(tmp_path: Path):
    db, repo, case = _case_and_repo(tmp_path, "case-reuse-txn")
    op_fail = _op(case.case_id, step="step-fail-reuse")
    repo.set_fault_hook(
        "after_observation",
        lambda: (_ for _ in ()).throw(RuntimeError("injected after_observation")),
    )
    with pytest.raises(RuntimeError, match="after_observation"):
        repo.register_artifact_group(
            operation=op_fail,
            receipt=_receipt("reuse-fail"),
            observation=_observation("obs-reuse-fail"),
            evidence=_evidence("ev-reuse-fail"),
        )
    assert repo._txn_conn is None  # noqa: SLF001 — lifecycle contract under review
    assert _counts(db, op_fail.operation_id) == {
        "artifacts": 0,
        "groups": 0,
        "operations": 0,
    }
    repo.clear_fault_hooks()
    op_ok = _op(case.case_id, step="step-ok-reuse")
    group = repo.register_artifact_group(
        operation=op_ok,
        receipt=_receipt("reuse-ok"),
        observation=_observation("obs-reuse-ok"),
        evidence=_evidence("ev-reuse-ok"),
    )
    assert group.reused is False
    assert repo._txn_conn is None  # noqa: SLF001
    assert _counts(db, op_ok.operation_id)["groups"] == 1
    db.close()


def test_print_success_and_failure_evidence(tmp_path: Path):
    db, repo, case = _case_and_repo(tmp_path, "case-trace")
    op = _op(case.case_id)
    ok = repo.register_artifact_group(
        operation=op,
        receipt=_receipt("ok"),
        observation=_observation("obs-ok"),
        evidence=_evidence("ev-ok"),
    )
    print("\n=== EB2A SUCCESSFUL REGISTRATION ===")
    print("operation_id:", ok.operation_id)
    print("receipt_id:", ok.receipt_id)
    print("observation_id:", ok.observation_id)
    print("evidence_id:", ok.evidence_id)
    print("group_fingerprint:", ok.group_fingerprint)
    print("=== END SUCCESS ===\n")

    op2 = _op(case.case_id, step="step-fail")
    repo.set_fault_hook(
        "after_receipt",
        lambda: (_ for _ in ()).throw(RuntimeError("forced rollback")),
    )
    with pytest.raises(RuntimeError, match="forced rollback"):
        repo.register_artifact_group(
            operation=op2,
            receipt=_receipt("fail"),
            observation=_observation("obs-fail"),
            evidence=_evidence("ev-fail"),
        )
    print("=== EB2A FAILED REGISTRATION ===")
    print("operation_id:", op2.operation_id)
    print("post_rollback_counts:", _counts(db, op2.operation_id))
    print("=== END FAILURE ===\n")
    assert _counts(db, op2.operation_id)["groups"] == 0
    db.close()
