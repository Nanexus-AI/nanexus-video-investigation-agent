"""Durable SQLite budget ledger. G3-C1 §9.

Validate admitted operation → Commit reservation → Permit external attempt.

This ledger is Investigation Persistence authority for tool-attempt accounting.
It is not a retry scheduler and does not enable technical retries.
Checkpoint and domain transactions remain separate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import StrEnum

from nanexus_investigation.persistence.db import InvestigationDatabase
from nanexus_investigation.persistence.errors import IdempotencyConflictError
from nanexus_investigation.persistence.operation_id import payload_fingerprint

UTC = timezone.utc


class ReservationStatus(StrEnum):
    """Internal budget accounting statuses. Not ToolResult statuses."""

    RESERVED = "reserved"
    CONFIRMED = "confirmed"
    FAILED = "failed"
    UNRESOLVED = "unresolved"


class BudgetExhaustedError(RuntimeError):
    """Insufficient remaining budget to commit a reservation."""


@dataclass(frozen=True, slots=True)
class ReservationRecord:
    reservation_id: str
    case_id: str
    operation_id: str
    attempt_id: str
    units: int
    status: ReservationStatus
    payload_fingerprint: str
    created_at: str
    updated_at: str
    execution_id: str | None = None

    @property
    def is_confirmed_physical_attempt(self) -> bool:
        return self.status is ReservationStatus.CONFIRMED

    @property
    def holds_allowance(self) -> bool:
        return self.status in {
            ReservationStatus.RESERVED,
            ReservationStatus.CONFIRMED,
            ReservationStatus.FAILED,
            ReservationStatus.UNRESOLVED,
        }


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _row_to_record(row) -> ReservationRecord:
    return ReservationRecord(
        reservation_id=row["reservation_id"],
        case_id=row["case_id"],
        operation_id=row["operation_id"],
        attempt_id=row["attempt_id"],
        units=int(row["units"]),
        status=ReservationStatus(row["status"]),
        payload_fingerprint=row["payload_fingerprint"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        execution_id=row["execution_id"],
    )


class DurableBudgetLedger:
    """SQLite-backed reserve-before-execution accounting for one Case allowance."""

    def __init__(self, db: InvestigationDatabase) -> None:
        self._db = db

    def configure_policy(self, *, case_id: str, max_tool_attempts: int) -> None:
        if max_tool_attempts < 0:
            raise ValueError("max_tool_attempts must be >= 0")
        with self._db.transaction() as conn:
            existing = conn.execute(
                "SELECT max_tool_attempts FROM nx_budget_policies WHERE case_id = ?",
                (case_id,),
            ).fetchone()
            if existing is not None:
                if int(existing["max_tool_attempts"]) != max_tool_attempts:
                    raise IdempotencyConflictError(
                        f"budget policy for {case_id} already set to "
                        f"{existing['max_tool_attempts']}"
                    )
                return
            conn.execute(
                """
                INSERT INTO nx_budget_policies(case_id, max_tool_attempts, created_at)
                VALUES (?, ?, ?)
                """,
                (case_id, max_tool_attempts, _now()),
            )

    def max_attempts(self, case_id: str) -> int:
        row = self._db.conn.execute(
            "SELECT max_tool_attempts FROM nx_budget_policies WHERE case_id = ?",
            (case_id,),
        ).fetchone()
        if row is None:
            raise LookupError(f"no budget policy for case {case_id}")
        return int(row["max_tool_attempts"])

    def reserved_units(self, case_id: str) -> int:
        row = self._db.conn.execute(
            """
            SELECT COALESCE(SUM(units), 0) AS total
            FROM nx_budget_reservations
            WHERE case_id = ?
            """,
            (case_id,),
        ).fetchone()
        return int(row["total"])

    def remaining(self, case_id: str) -> int:
        return max(0, self.max_attempts(case_id) - self.reserved_units(case_id))

    def get_reservation(self, reservation_id: str) -> ReservationRecord | None:
        row = self._db.conn.execute(
            "SELECT * FROM nx_budget_reservations WHERE reservation_id = ?",
            (reservation_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_record(row)

    def get_by_attempt(self, *, case_id: str, attempt_id: str) -> ReservationRecord | None:
        row = self._db.conn.execute(
            """
            SELECT * FROM nx_budget_reservations
            WHERE case_id = ? AND attempt_id = ?
            """,
            (case_id, attempt_id),
        ).fetchone()
        if row is None:
            return None
        return _row_to_record(row)

    def list_reservations(self, case_id: str) -> list[ReservationRecord]:
        """Read-only listing of budget reservations for one Case."""
        rows = self._db.conn.execute(
            """
            SELECT * FROM nx_budget_reservations
            WHERE case_id = ?
            ORDER BY created_at ASC, reservation_id ASC
            """,
            (case_id,),
        ).fetchall()
        return [_row_to_record(row) for row in rows]

    def reserve(
        self,
        *,
        reservation_id: str,
        case_id: str,
        operation_id: str,
        attempt_id: str,
        units: int = 1,
        request_fingerprint: str | None = None,
    ) -> ReservationRecord:
        """Atomically commit a durable reservation before external execution.

        Replaying the same reservation/attempt identity with equivalent content
        returns the original record without double-charging.
        """
        if units <= 0:
            raise ValueError("units must be positive")
        fingerprint = request_fingerprint or payload_fingerprint(
            {
                "reservation_id": reservation_id,
                "case_id": case_id,
                "operation_id": operation_id,
                "attempt_id": attempt_id,
                "units": units,
            }
        )
        with self._db.transaction() as conn:
            policy = conn.execute(
                "SELECT max_tool_attempts FROM nx_budget_policies WHERE case_id = ?",
                (case_id,),
            ).fetchone()
            if policy is None:
                raise LookupError(f"no budget policy for case {case_id}")

            existing = conn.execute(
                """
                SELECT * FROM nx_budget_reservations
                WHERE reservation_id = ? OR (case_id = ? AND attempt_id = ?)
                """,
                (reservation_id, case_id, attempt_id),
            ).fetchone()
            if existing is not None:
                if existing["payload_fingerprint"] != fingerprint:
                    raise IdempotencyConflictError(
                        f"reservation identity conflict for {reservation_id}/{attempt_id}"
                    )
                if existing["reservation_id"] != reservation_id:
                    raise IdempotencyConflictError(
                        f"attempt_id {attempt_id} already bound to a different reservation"
                    )
                return _row_to_record(existing)

            used = conn.execute(
                """
                SELECT COALESCE(SUM(units), 0) AS total
                FROM nx_budget_reservations
                WHERE case_id = ?
                """,
                (case_id,),
            ).fetchone()
            remaining = int(policy["max_tool_attempts"]) - int(used["total"])
            if remaining < units:
                raise BudgetExhaustedError(
                    f"insufficient budget for case {case_id}: "
                    f"need {units}, remaining {remaining}"
                )

            now = _now()
            conn.execute(
                """
                INSERT INTO nx_budget_reservations(
                    reservation_id, case_id, operation_id, attempt_id, units,
                    status, payload_fingerprint, created_at, updated_at, execution_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    reservation_id,
                    case_id,
                    operation_id,
                    attempt_id,
                    units,
                    ReservationStatus.RESERVED.value,
                    fingerprint,
                    now,
                    now,
                ),
            )
            row = conn.execute(
                "SELECT * FROM nx_budget_reservations WHERE reservation_id = ?",
                (reservation_id,),
            ).fetchone()
            return _row_to_record(row)

    def mark_unresolved(self, reservation_id: str) -> ReservationRecord:
        """Record unknown execution outcome. Does not refund the reservation."""
        return self._set_status(
            reservation_id,
            ReservationStatus.UNRESOLVED,
            allow_from={ReservationStatus.RESERVED, ReservationStatus.UNRESOLVED},
        )

    def mark_confirmed(
        self,
        reservation_id: str,
        *,
        execution_id: str,
    ) -> ReservationRecord:
        """Record a confirmed physical attempt when execution evidence exists."""
        if not execution_id.strip():
            raise ValueError("execution_id required to confirm an attempt")
        return self._set_status(
            reservation_id,
            ReservationStatus.CONFIRMED,
            allow_from={
                ReservationStatus.RESERVED,
                ReservationStatus.UNRESOLVED,
                ReservationStatus.CONFIRMED,
            },
            execution_id=execution_id,
        )

    def mark_failed(
        self,
        reservation_id: str,
        *,
        execution_id: str | None = None,
    ) -> ReservationRecord:
        """Failed invocation retains the reservation (no silent erase/refund)."""
        return self._set_status(
            reservation_id,
            ReservationStatus.FAILED,
            allow_from={
                ReservationStatus.RESERVED,
                ReservationStatus.UNRESOLVED,
                ReservationStatus.FAILED,
            },
            execution_id=execution_id,
        )

    def _set_status(
        self,
        reservation_id: str,
        status: ReservationStatus,
        *,
        allow_from: set[ReservationStatus],
        execution_id: str | None = None,
    ) -> ReservationRecord:
        with self._db.transaction() as conn:
            row = conn.execute(
                "SELECT * FROM nx_budget_reservations WHERE reservation_id = ?",
                (reservation_id,),
            ).fetchone()
            if row is None:
                raise LookupError(f"unknown reservation {reservation_id}")
            current = ReservationStatus(row["status"])
            if current not in allow_from:
                raise ValueError(
                    f"cannot transition reservation {reservation_id} "
                    f"from {current} to {status}"
                )
            if current is status and (
                execution_id is None or row["execution_id"] == execution_id
            ):
                return _row_to_record(row)
            conn.execute(
                """
                UPDATE nx_budget_reservations
                SET status = ?, updated_at = ?, execution_id = COALESCE(?, execution_id)
                WHERE reservation_id = ?
                """,
                (status.value, _now(), execution_id, reservation_id),
            )
            updated = conn.execute(
                "SELECT * FROM nx_budget_reservations WHERE reservation_id = ?",
                (reservation_id,),
            ).fetchone()
            return _row_to_record(updated)


def reserve_then_invoke(
    ledger: DurableBudgetLedger,
    *,
    reservation_id: str,
    case_id: str,
    operation_id: str,
    attempt_id: str,
    stub,
) -> tuple[ReservationRecord, object]:
    """Mandatory E0 sequence: commit reservation, then permit stub invocation.

    If reservation cannot commit, ``BudgetExhaustedError`` is raised and the
    stub is never called. Failed stub invocations retain the reservation.
    """
    reservation = ledger.reserve(
        reservation_id=reservation_id,
        case_id=case_id,
        operation_id=operation_id,
        attempt_id=attempt_id,
    )
    try:
        result = stub()
    except Exception:
        ledger.mark_failed(reservation_id)
        raise
    return reservation, result
