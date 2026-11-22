"""Tool Guard: validate requests; never own workflow routing or retries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol
from uuid import uuid4


from nanexus_investigation.contracts.capability import CapabilitySnapshot
from nanexus_investigation.contracts.enums import CapabilityAvailability
from nanexus_investigation.contracts.identity import Interval
from nanexus_investigation.contracts.scope import InvestigationScope
from nanexus_investigation.contracts.tool import ToolRequest
from nanexus_investigation.tools.protocol import ToolRegistryView


class BudgetLedgerView(Protocol):
    """Minimal budget surface used by Tool Guard. In-memory or durable adapter."""

    max_attempts: int

    @property
    def remaining(self) -> int: ...

    @property
    def reserved(self) -> int: ...

    @property
    def consumed(self) -> int: ...

    def can_reserve(self) -> bool: ...

    def reserve(self, request_id: str) -> bool: ...

    def mark_consumed(self, request_id: str) -> None: ...


@dataclass(frozen=True, slots=True)
class GuardDecision:
    """Structured Tool Guard outcome. Rejection never reaches the tool body."""

    accepted: bool
    request_id: str
    reasons: tuple[str, ...] = ()
    request: ToolRequest | None = None

    @property
    def rejected(self) -> bool:
        return not self.accepted


class ToolGuard:
    """Cross-cutting request validation and pre-execution budget reservation.

    Mandatory order for accepted calls:
    proposal → Guard validation → budget reservation → registry invocation.
    LangGraph node retry is not enabled by this component.
    """

    def __init__(
        self,
        *,
        registry: ToolRegistryView,
        budget: BudgetLedgerView,
        effective_scope: InvestigationScope,
        capabilities: CapabilitySnapshot,
    ) -> None:
        self._registry = registry
        self._budget = budget
        self._scope = effective_scope
        self._capabilities = capabilities

    def validate(self, proposal: dict[str, Any]) -> GuardDecision:
        request_id = proposal.get("request_id") or f"req-{uuid4().hex[:12]}"
        tool_name = proposal.get("tool_name", "")
        reasons: list[str] = []

        if not self._registry.is_registered(tool_name):
            reasons.append(f"unknown or unregistered tool: {tool_name}")

        camera_ids = tuple(proposal.get("camera_ids") or self._scope.camera_ids)
        for camera_id in camera_ids:
            if camera_id not in self._scope.camera_ids:
                reasons.append(f"unauthorized camera: {camera_id}")

        source_id = proposal.get("source_id") or self._scope.vms_instance_id
        if source_id != self._scope.vms_instance_id:
            reasons.append(f"unauthorized vms instance: {source_id}")

        time_bounds = proposal.get("time_bounds")
        if time_bounds is None:
            time_bounds = self._scope.time_bounds
        elif isinstance(time_bounds, dict):
            time_bounds = Interval(
                start_utc=time_bounds["start_utc"],
                end_utc=time_bounds["end_utc"],
            )
        if not isinstance(time_bounds, Interval):
            reasons.append("time_bounds must be an Interval")
        else:
            if not self._scope.contains_interval(time_bounds):
                reasons.append("requested time window is outside effective scope")

        capability = self._capabilities.for_operation(tool_name)
        if capability is None:
            reasons.append(f"capability unknown for tool: {tool_name}")
        elif capability.availability is not CapabilityAvailability.SUPPORTED:
            reasons.append(
                f"capability not supported for tool: {tool_name} "
                f"({capability.availability})"
            )
        elif capability.authorized_camera_ids is not None:
            for camera_id in camera_ids:
                if camera_id not in capability.authorized_camera_ids:
                    reasons.append(
                        f"camera not authorized for capability {tool_name}: {camera_id}"
                    )

        if not self._budget.can_reserve():
            reasons.append("tool-attempt budget exhausted")

        if reasons:
            return GuardDecision(
                accepted=False,
                request_id=request_id,
                reasons=tuple(reasons),
            )

        arguments = dict(proposal.get("arguments") or {})
        if isinstance(time_bounds, Interval):
            arguments.setdefault(
                "time_bounds",
                {
                    "start_utc": time_bounds.start_utc.isoformat(),
                    "end_utc": (
                        time_bounds.end_utc.isoformat()
                        if time_bounds.end_utc is not None
                        else None
                    ),
                },
            )
        arguments.setdefault("label", proposal.get("label"))

        request = ToolRequest(
            tool_name=tool_name,
            arguments={k: v for k, v in arguments.items() if v is not None},
            source_id=source_id,
            camera_ids=camera_ids,
        )
        return GuardDecision(
            accepted=True,
            request_id=request_id,
            request=request,
        )

    def reserve(self, decision: GuardDecision) -> bool:
        """Reserve budget only after validation acceptance."""
        if decision.rejected or decision.request is None:
            return False
        return self._budget.reserve(decision.request_id)
