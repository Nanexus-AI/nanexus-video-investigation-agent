"""Stable operation identity and idempotent domain writes. G4-V1-EA / G3-C1 §8."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from nanexus_investigation.contracts.case import InvestigationCase
from nanexus_investigation.contracts.enums import (
    CaseStatus,
    ObservationType,
    RecordType,
)
from nanexus_investigation.contracts.evidence import Observation
from nanexus_investigation.contracts.identity import Interval, SourceLocator, TimeContext
from nanexus_investigation.contracts.scope import InvestigationScope
from nanexus_investigation.persistence.db import InvestigationDatabase
from nanexus_investigation.persistence.errors import IdempotencyConflictError
from nanexus_investigation.persistence.operation_id import (
    build_attempt_identity,
    build_operation_identity,
)
from nanexus_investigation.persistence.repository import InvestigationRepository

UTC = timezone.utc
T0 = datetime(2026, 11, 19, 20, 0, tzinfo=UTC)
T1 = datetime(2026, 11, 19, 23, 0, tzinfo=UTC)


def _case() -> InvestigationCase:
    scope = InvestigationScope(
        vms_instance_id="vms-local-1",
        camera_ids=("front_yard",),
        time_bounds=Interval(start_utc=T0, end_utc=T1),
    )
    return InvestigationCase(
        case_id="case-idem-1",
        question="Find person activity.",
        requested_scope=scope,
        effective_scope=scope,
        time_context=TimeContext(
            timezone="UTC",
            start_utc=T0,
            end_utc=T1,
        ),
        status=CaseStatus.CREATED,
        policy_ref="policy:e0",
        created_at=datetime(2026, 11, 20, 12, 0, tzinfo=UTC),
    )


def _observation(obs_id: str, native_id: str) -> Observation:
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
        start_utc=T0,
        end_utc=T0.replace(minute=3),
        retrieved_at=T0,
        provenance="stub:search_events",
    )


def _repo(tmp_path: Path) -> InvestigationRepository:
    db = InvestigationDatabase.open(tmp_path / "investigation.sqlite")
    repo = InvestigationRepository(db)
    repo.upsert_case(_case())
    return repo


def test_same_logical_write_returns_original_artifact(tmp_path: Path):
    repo = _repo(tmp_path)
    op = build_operation_identity(
        case_id="case-idem-1",
        plan_step_id="step-1",
        tool_name="search_events",
        vms_instance_id="vms-local-1",
        request_parameters={"label": "person"},
    )
    first = repo.put_observation(_observation("obs-1", "evt-1"), operation=op)
    second = repo.put_observation(_observation("obs-1", "evt-1"), operation=op)
    assert first.observation_id == second.observation_id == "obs-1"
    assert first.source_locator.native_record_id == "evt-1"
    # Only one row for this operation/kind.
    count = repo.db.conn.execute(
        "SELECT COUNT(*) AS c FROM nx_artifacts WHERE operation_id = ?",
        (op.operation_id,),
    ).fetchone()["c"]
    assert count == 1
    repo.db.close()


def test_same_identity_conflicting_content_raises(tmp_path: Path):
    repo = _repo(tmp_path)
    op = build_operation_identity(
        case_id="case-idem-1",
        plan_step_id="step-1",
        tool_name="search_events",
        vms_instance_id="vms-local-1",
        request_parameters={"label": "person"},
    )
    repo.put_observation(_observation("obs-1", "evt-1"), operation=op)
    conflict = _observation("obs-1", "evt-DIFFERENT")
    with pytest.raises(IdempotencyConflictError):
        repo.put_observation(conflict, operation=op)
    repo.db.close()


def test_deliberate_new_operation_identity_creates_separate_artifact(tmp_path: Path):
    repo = _repo(tmp_path)
    op1 = build_operation_identity(
        case_id="case-idem-1",
        plan_step_id="step-1",
        tool_name="search_events",
        vms_instance_id="vms-local-1",
        request_parameters={"label": "person"},
    )
    op2 = build_operation_identity(
        case_id="case-idem-1",
        plan_step_id="step-2",
        tool_name="search_events",
        vms_instance_id="vms-local-1",
        request_parameters={"label": "person"},
    )
    assert op1.operation_id != op2.operation_id
    repo.put_observation(_observation("obs-1", "evt-1"), operation=op1)
    repo.put_observation(_observation("obs-2", "evt-2"), operation=op2)
    assert repo.get_observation("obs-1") is not None
    assert repo.get_observation("obs-2") is not None
    count = repo.db.conn.execute("SELECT COUNT(*) AS c FROM nx_artifacts").fetchone()["c"]
    assert count == 2
    repo.db.close()


def test_restart_preserves_uniqueness(tmp_path: Path):
    path = tmp_path / "investigation.sqlite"
    op = build_operation_identity(
        case_id="case-idem-1",
        plan_step_id="step-1",
        tool_name="search_events",
        vms_instance_id="vms-local-1",
        request_parameters={"label": "person"},
    )
    db1 = InvestigationDatabase.open(path)
    repo1 = InvestigationRepository(db1)
    repo1.upsert_case(_case())
    repo1.put_observation(_observation("obs-1", "evt-1"), operation=op)
    db1.close()

    db2 = InvestigationDatabase.open(path)
    repo2 = InvestigationRepository(db2)
    # Replay after process/connection restart returns original, no duplicate.
    again = repo2.put_observation(_observation("obs-1", "evt-1"), operation=op)
    assert again.observation_id == "obs-1"
    count = repo2.db.conn.execute("SELECT COUNT(*) AS c FROM nx_artifacts").fetchone()["c"]
    assert count == 1
    with pytest.raises(IdempotencyConflictError):
        repo2.put_observation(_observation("obs-1", "evt-OTHER"), operation=op)
    db2.close()


def test_attempt_identity_distinguishes_physical_attempts():
    op = build_operation_identity(
        case_id="case-idem-1",
        plan_step_id="step-1",
        tool_name="search_events",
        vms_instance_id="vms-local-1",
        request_parameters={"label": "person"},
    )
    a1 = build_attempt_identity(operation_id=op.operation_id, attempt_index=1)
    a2 = build_attempt_identity(operation_id=op.operation_id, attempt_index=2)
    assert a1.attempt_id != a2.attempt_id
    assert a1.operation_id == a2.operation_id == op.operation_id


def test_operation_identity_stable_across_equivalent_parameters():
    kwargs = dict(
        case_id="case-idem-1",
        plan_step_id="step-1",
        tool_name="search_events",
        vms_instance_id="vms-local-1",
        request_parameters={"label": "person", "camera_ids": ["front_yard"]},
    )
    first = build_operation_identity(**kwargs)
    second = build_operation_identity(**kwargs)
    assert first.operation_id == second.operation_id
    assert first.content_fingerprint == second.content_fingerprint
