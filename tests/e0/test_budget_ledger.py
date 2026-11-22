"""Durable SQLite budget reservation accounting. G4-V1-EA / G3-C1 §9."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from nanexus_investigation.contracts.case import InvestigationCase
from nanexus_investigation.contracts.enums import CaseStatus
from nanexus_investigation.contracts.identity import Interval, TimeContext
from nanexus_investigation.contracts.scope import InvestigationScope
from nanexus_investigation.persistence.budget import (
    BudgetExhaustedError,
    DurableBudgetLedger,
    ReservationStatus,
    reserve_then_invoke,
)
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


def _setup(tmp_path: Path, *, max_attempts: int = 2):
    path = tmp_path / "investigation.sqlite"
    db = InvestigationDatabase.open(path)
    repo = InvestigationRepository(db)
    scope = InvestigationScope(
        vms_instance_id="vms-local-1",
        camera_ids=("front_yard",),
        time_bounds=Interval(start_utc=T0, end_utc=T1),
    )
    case = InvestigationCase(
        case_id="case-budget-1",
        question="Budget test",
        requested_scope=scope,
        effective_scope=scope,
        time_context=TimeContext(timezone="UTC", start_utc=T0, end_utc=T1),
        status=CaseStatus.CREATED,
        policy_ref="policy:e0",
        created_at=datetime(2026, 11, 20, 12, 0, tzinfo=UTC),
    )
    repo.upsert_case(case)
    ledger = DurableBudgetLedger(db)
    ledger.configure_policy(case_id=case.case_id, max_tool_attempts=max_attempts)
    op = build_operation_identity(
        case_id=case.case_id,
        plan_step_id="step-1",
        tool_name="search_events",
        vms_instance_id="vms-local-1",
        request_parameters={"label": "person"},
    )
    return path, db, ledger, case.case_id, op


def test_successful_durable_reservation(tmp_path: Path):
    _, db, ledger, case_id, op = _setup(tmp_path)
    attempt = build_attempt_identity(operation_id=op.operation_id, attempt_index=1)
    record = ledger.reserve(
        reservation_id="res-1",
        case_id=case_id,
        operation_id=op.operation_id,
        attempt_id=attempt.attempt_id,
    )
    assert record.status is ReservationStatus.RESERVED
    assert ledger.remaining(case_id) == 1
    assert not record.is_confirmed_physical_attempt
    db.close()


def test_insufficient_budget_rejection(tmp_path: Path):
    _, db, ledger, case_id, op = _setup(tmp_path, max_attempts=1)
    a1 = build_attempt_identity(operation_id=op.operation_id, attempt_index=1)
    a2 = build_attempt_identity(operation_id=op.operation_id, attempt_index=2)
    ledger.reserve(
        reservation_id="res-1",
        case_id=case_id,
        operation_id=op.operation_id,
        attempt_id=a1.attempt_id,
    )
    with pytest.raises(BudgetExhaustedError):
        ledger.reserve(
            reservation_id="res-2",
            case_id=case_id,
            operation_id=op.operation_id,
            attempt_id=a2.attempt_id,
        )
    assert ledger.remaining(case_id) == 0
    db.close()


def test_same_identity_replay_does_not_double_charge(tmp_path: Path):
    _, db, ledger, case_id, op = _setup(tmp_path, max_attempts=1)
    attempt = build_attempt_identity(operation_id=op.operation_id, attempt_index=1)
    first = ledger.reserve(
        reservation_id="res-1",
        case_id=case_id,
        operation_id=op.operation_id,
        attempt_id=attempt.attempt_id,
    )
    second = ledger.reserve(
        reservation_id="res-1",
        case_id=case_id,
        operation_id=op.operation_id,
        attempt_id=attempt.attempt_id,
    )
    assert first.reservation_id == second.reservation_id
    assert ledger.reserved_units(case_id) == 1
    db.close()


def test_conflicting_reservation_content_rejected(tmp_path: Path):
    _, db, ledger, case_id, op = _setup(tmp_path)
    attempt = build_attempt_identity(operation_id=op.operation_id, attempt_index=1)
    ledger.reserve(
        reservation_id="res-1",
        case_id=case_id,
        operation_id=op.operation_id,
        attempt_id=attempt.attempt_id,
        units=1,
    )
    with pytest.raises(IdempotencyConflictError):
        ledger.reserve(
            reservation_id="res-1",
            case_id=case_id,
            operation_id=op.operation_id,
            attempt_id=attempt.attempt_id,
            units=2,
        )
    db.close()


def test_new_connection_reads_committed_reservation(tmp_path: Path):
    path, db, ledger, case_id, op = _setup(tmp_path)
    attempt = build_attempt_identity(operation_id=op.operation_id, attempt_index=1)
    ledger.reserve(
        reservation_id="res-1",
        case_id=case_id,
        operation_id=op.operation_id,
        attempt_id=attempt.attempt_id,
    )
    db.close()

    db2 = InvestigationDatabase.open(path)
    ledger2 = DurableBudgetLedger(db2)
    loaded = ledger2.get_reservation("res-1")
    assert loaded is not None
    assert loaded.status is ReservationStatus.RESERVED
    assert ledger2.remaining(case_id) == 1
    db2.close()


def test_unresolved_reservation_after_simulated_interruption(tmp_path: Path):
    _, db, ledger, case_id, op = _setup(tmp_path)
    attempt = build_attempt_identity(operation_id=op.operation_id, attempt_index=1)
    ledger.reserve(
        reservation_id="res-1",
        case_id=case_id,
        operation_id=op.operation_id,
        attempt_id=attempt.attempt_id,
    )
    unresolved = ledger.mark_unresolved("res-1")
    assert unresolved.status is ReservationStatus.UNRESOLVED
    assert unresolved.holds_allowance
    assert not unresolved.is_confirmed_physical_attempt
    # Not refunded.
    assert ledger.remaining(case_id) == 1
    db.close()


def test_confirmed_attempt_recording(tmp_path: Path):
    _, db, ledger, case_id, op = _setup(tmp_path)
    attempt = build_attempt_identity(operation_id=op.operation_id, attempt_index=1)
    ledger.reserve(
        reservation_id="res-1",
        case_id=case_id,
        operation_id=op.operation_id,
        attempt_id=attempt.attempt_id,
    )
    confirmed = ledger.mark_confirmed("res-1", execution_id="exec-1")
    assert confirmed.status is ReservationStatus.CONFIRMED
    assert confirmed.execution_id == "exec-1"
    assert confirmed.is_confirmed_physical_attempt
    db.close()


def test_failed_invocation_retains_reservation(tmp_path: Path):
    _, db, ledger, case_id, op = _setup(tmp_path, max_attempts=1)
    attempt = build_attempt_identity(operation_id=op.operation_id, attempt_index=1)

    def boom():
        raise RuntimeError("stub failed")

    with pytest.raises(RuntimeError, match="stub failed"):
        reserve_then_invoke(
            ledger,
            reservation_id="res-1",
            case_id=case_id,
            operation_id=op.operation_id,
            attempt_id=attempt.attempt_id,
            stub=boom,
        )
    record = ledger.get_reservation("res-1")
    assert record is not None
    assert record.status is ReservationStatus.FAILED
    assert ledger.remaining(case_id) == 0
    db.close()


def test_multiple_reservations_against_limited_allowance(tmp_path: Path):
    _, db, ledger, case_id, op = _setup(tmp_path, max_attempts=2)
    a1 = build_attempt_identity(operation_id=op.operation_id, attempt_index=1)
    a2 = build_attempt_identity(operation_id=op.operation_id, attempt_index=2)
    a3 = build_attempt_identity(operation_id=op.operation_id, attempt_index=3)
    ledger.reserve(
        reservation_id="res-1",
        case_id=case_id,
        operation_id=op.operation_id,
        attempt_id=a1.attempt_id,
    )
    ledger.reserve(
        reservation_id="res-2",
        case_id=case_id,
        operation_id=op.operation_id,
        attempt_id=a2.attempt_id,
    )
    assert ledger.remaining(case_id) == 0
    with pytest.raises(BudgetExhaustedError):
        ledger.reserve(
            reservation_id="res-3",
            case_id=case_id,
            operation_id=op.operation_id,
            attempt_id=a3.attempt_id,
        )
    db.close()


def test_no_stub_invocation_when_reservation_cannot_commit(tmp_path: Path):
    _, db, ledger, case_id, op = _setup(tmp_path, max_attempts=0)
    attempt = build_attempt_identity(operation_id=op.operation_id, attempt_index=1)
    calls: list[str] = []

    def stub():
        calls.append("invoked")
        return "ok"

    with pytest.raises(BudgetExhaustedError):
        reserve_then_invoke(
            ledger,
            reservation_id="res-1",
            case_id=case_id,
            operation_id=op.operation_id,
            attempt_id=attempt.attempt_id,
            stub=stub,
        )
    assert calls == []
    assert ledger.get_reservation("res-1") is None
    db.close()


def test_reserve_then_invoke_success_sequence(tmp_path: Path):
    _, db, ledger, case_id, op = _setup(tmp_path)
    attempt = build_attempt_identity(operation_id=op.operation_id, attempt_index=1)
    order: list[str] = []

    def stub():
        # Reservation must already be durable before stub body runs.
        existing = ledger.get_reservation("res-1")
        assert existing is not None
        assert existing.status is ReservationStatus.RESERVED
        order.append("stub")
        return {"execution_id": "exec-ok"}

    reservation, result = reserve_then_invoke(
        ledger,
        reservation_id="res-1",
        case_id=case_id,
        operation_id=op.operation_id,
        attempt_id=attempt.attempt_id,
        stub=stub,
    )
    assert order == ["stub"]
    assert reservation.status is ReservationStatus.RESERVED
    assert result["execution_id"] == "exec-ok"
    confirmed = ledger.mark_confirmed("res-1", execution_id="exec-ok")
    assert confirmed.is_confirmed_physical_attempt
    db.close()
