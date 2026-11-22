"""In-memory tool-attempt budget ledger for E0. Not crash-safe or durable."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InMemoryBudgetLedger:
    """Tracks reserved and consumed tool attempts for a single investigation.

    This ledger is a development substitute only. It is not Investigation
    Persistence and must not be described as durable recovery.
    """

    max_attempts: int
    reserved: int = 0
    consumed: int = 0
    reservations: list[str] = field(default_factory=list)

    @property
    def remaining(self) -> int:
        return max(0, self.max_attempts - self.reserved)

    def can_reserve(self) -> bool:
        return self.remaining > 0

    def reserve(self, request_id: str) -> bool:
        """Pre-execution reservation. Failed or unresolved attempts stay charged."""
        if not self.can_reserve():
            return False
        self.reserved += 1
        self.reservations.append(request_id)
        return True

    def mark_consumed(self, request_id: str) -> None:
        if request_id in self.reservations:
            self.consumed += 1
