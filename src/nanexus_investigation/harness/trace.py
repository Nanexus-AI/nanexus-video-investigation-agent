"""In-memory execution trace for E0 assertions. Not durable audit storage."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExecutionTrace:
    """Ordered investigation events for deterministic test assertions."""

    events: list[dict[str, Any]] = field(default_factory=list)

    def record(self, kind: str, **payload: Any) -> None:
        self.events.append({"kind": kind, **payload})

    def of_kind(self, kind: str) -> list[dict[str, Any]]:
        return [event for event in self.events if event["kind"] == kind]

    def node_sequence(self) -> list[str]:
        return [event["node"] for event in self.events if event["kind"] == "node"]
