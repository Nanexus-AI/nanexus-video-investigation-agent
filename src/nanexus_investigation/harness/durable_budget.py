"""Adapter: DurableBudgetLedger → Tool Guard budget interface.

Used only on the SQLite-backed investigation path. The in-memory ledger remains
the default for existing E0 graph tests.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nanexus_investigation.persistence.budget import (
    BudgetExhaustedError,
    DurableBudgetLedger,
    ReservationRecord,
    ReservationStatus,
)
from nanexus_investigation.persistence.operation_id import (
    AttemptIdentity,
    OperationIdentity,
)


@dataclass
class _BoundAttempt:
    request_id: str
    operation: OperationIdentity
    attempt: AttemptIdentity
    fresh: bool = True


@dataclass
class DurableBudgetAdapter:
    """Presents durable reservations through the Guard's reserve/can_reserve API.

    Replay of the same reservation identity does not double-charge. An existing
    unresolved (or prior reserved-without-resolution) attempt fails closed and
    must not authorize automatic stub re-dispatch.
    """

    ledger: DurableBudgetLedger
    case_id: str
    max_attempts: int
    _bound: _BoundAttempt | None = None
    _request_map: dict[str, _BoundAttempt] = field(default_factory=dict)
    _block_invoke: dict[str, str] = field(default_factory=dict)
    _confirmed_requests: set[str] = field(default_factory=set)
    last_error: str | None = None

    @property
    def reserved(self) -> int:
        return self.ledger.reserved_units(self.case_id)

    @property
    def consumed(self) -> int:
        rows = self.ledger._db.conn.execute(  # noqa: SLF001 — E0 accounting summary
            """
            SELECT COUNT(*) AS c FROM nx_budget_reservations
            WHERE case_id = ? AND status IN ('confirmed', 'failed')
            """,
            (self.case_id,),
        ).fetchone()
        return int(rows["c"])

    @property
    def remaining(self) -> int:
        return self.ledger.remaining(self.case_id)

    def can_reserve(self) -> bool:
        return self.remaining > 0

    def bind(
        self,
        *,
        request_id: str,
        operation: OperationIdentity,
        attempt: AttemptIdentity,
    ) -> None:
        self._bound = _BoundAttempt(
            request_id=request_id,
            operation=operation,
            attempt=attempt,
            fresh=True,
        )

    def reserve(self, request_id: str) -> bool:
        """Commit durable reservation. Returns False when exhausted or blocked."""
        self.last_error = None
        bound = self._bound
        if bound is None or bound.request_id != request_id:
            self.last_error = "budget adapter missing bound attempt context"
            return False

        existing = self.ledger.get_by_attempt(
            case_id=self.case_id, attempt_id=bound.attempt.attempt_id
        )
        if existing is not None:
            return self._handle_existing(request_id, bound, existing)

        try:
            record = self.ledger.reserve(
                reservation_id=request_id,
                case_id=self.case_id,
                operation_id=bound.operation.operation_id,
                attempt_id=bound.attempt.attempt_id,
            )
        except BudgetExhaustedError:
            self.last_error = "tool-attempt budget exhausted"
            return False

        bound.fresh = True
        self._request_map[request_id] = bound
        self._block_invoke.pop(request_id, None)
        assert record.status is ReservationStatus.RESERVED
        return True

    def _handle_existing(
        self,
        request_id: str,
        bound: _BoundAttempt,
        existing: ReservationRecord,
    ) -> bool:
        if existing.status is ReservationStatus.UNRESOLVED:
            self.last_error = (
                "unresolved reservation blocks automatic re-dispatch; "
                "fail closed pending recovery handling"
            )
            self._block_invoke[request_id] = "unresolved_reservation"
            self._request_map[request_id] = bound
            return False

        if existing.status is ReservationStatus.CONFIRMED:
            # Already charged and confirmed — do not authorize another dispatch.
            self.last_error = "reservation already confirmed; refusing re-dispatch"
            self._block_invoke[request_id] = "already_confirmed"
            self._request_map[request_id] = bound
            return False

        if existing.status is ReservationStatus.FAILED:
            self.last_error = "reservation already failed; refusing re-dispatch"
            self._block_invoke[request_id] = "already_failed"
            self._request_map[request_id] = bound
            return False

        # RESERVED from a prior process without resolution: fail closed (N6).
        # Same-identity replay within a single reserved row still holds allowance
        # but must not silently mean "safe to invoke again" after interruption.
        if existing.reservation_id != request_id:
            self.last_error = "attempt identity bound to a different reservation"
            return False

        # Treat prior reserved-without-fresh-bind as unsafe for auto-dispatch.
        # A brand-new bind in this process that finds RESERVED from earlier work
        # is not a fresh commit in this turn.
        self.last_error = (
            "prior reserved attempt has unknown execution outcome; "
            "refusing automatic re-dispatch"
        )
        self._block_invoke[request_id] = "prior_reserved_unresolved"
        self._request_map[request_id] = bound
        return False

    def invoke_blocked_reason(self, request_id: str) -> str | None:
        return self._block_invoke.get(request_id)

    def mark_consumed(self, request_id: str) -> None:
        """Confirm physical attempt when stub returned an execution_id."""
        bound = self._request_map.get(request_id)
        if bound is None:
            return
        # Confirmation requires execution evidence supplied by execute_tool.
        # This method alone is not proof of dispatch; execute_tool calls
        # mark_confirmed_with_execution after a real stub result.
        self._confirmed_requests.add(request_id)

    def mark_confirmed_with_execution(
        self, request_id: str, *, execution_id: str
    ) -> ReservationRecord:
        return self.ledger.mark_confirmed(request_id, execution_id=execution_id)

    def mark_failed(self, request_id: str, *, execution_id: str | None = None) -> None:
        if self.ledger.get_reservation(request_id) is None:
            return
        self.ledger.mark_failed(request_id, execution_id=execution_id)

    def mark_unresolved(self, request_id: str) -> ReservationRecord:
        return self.ledger.mark_unresolved(request_id)
