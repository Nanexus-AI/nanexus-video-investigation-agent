"""Fixed allowlist of deterministic stub tools. Not a live VMS Adapter."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from nanexus_investigation.contracts.enums import (
    SearchCoverage,
    ToolErrorCode,
    ToolResultStatus,
)
from nanexus_investigation.contracts.tool import ToolError, ToolRequest, ToolResult

ALLOWED_TOOLS = frozenset(
    {
        "search_events",
        "search_related_events",
        "analyze_temporal_relations",
        "resolve_media",
        "observe_media",
    }
)

# Local deterministic analysis tools (not VMS HTTP adapters).
ANALYSIS_TOOLS = frozenset({"analyze_temporal_relations"})
MEDIA_TOOLS = frozenset({"resolve_media"})
# Local VLM observation over retained media (not a VMS HTTP adapter).
VISUAL_TOOLS = frozenset({"observe_media"})

ToolHandler = Callable[[ToolRequest], ToolResult]
UTC = timezone.utc


@dataclass
class StubToolRegistry:
    """Scripted tool bodies for E0. No network or filesystem access.

    Planning and tool responses are deterministic test behavior, not model
    autonomy. Temporal analysis uses the bound Case store when present.
    """

    scripts: dict[str, list[ToolResult | dict[str, Any]]] = field(default_factory=dict)
    call_counts: dict[str, int] = field(default_factory=dict)
    invocations: list[str] = field(default_factory=list)
    # Bound after InvestigationRuntime construction (Case observation store).
    analysis_store: Any | None = None
    analysis_scope: Any | None = None
    analysis_max_pairs: int = 100
    analysis_max_window_seconds: float = 3600.0
    media_store: Any | None = None
    media_case_id: str | None = None
    visual_provider: Any | None = None

    def bind_case_store(self, store: Any, scope: Any) -> None:
        self.analysis_store = store
        self.analysis_scope = scope

    def bind_media_store(self, media_store: Any, *, case_id: str) -> None:
        self.media_store = media_store
        self.media_case_id = case_id

    def bind_visual_provider(self, visual_provider: Any | None) -> None:
        self.visual_provider = visual_provider

    def is_registered(self, tool_name: str) -> bool:
        return tool_name in ALLOWED_TOOLS

    def invoke(self, request: ToolRequest) -> ToolResult:
        name = request.tool_name
        if name not in ALLOWED_TOOLS:
            raise RuntimeError(f"unregistered tool body invoked: {name}")
        self.invocations.append(name)
        index = self.call_counts.get(name, 0)
        self.call_counts[name] = index + 1

        if name in ANALYSIS_TOOLS:
            return self._invoke_temporal(request)

        if name in VISUAL_TOOLS:
            return self._invoke_observe_media(request)

        scripted = self.scripts.get(name, [])
        if index >= len(scripted):
            return ToolResult(
                status=ToolResultStatus.EMPTY,
                execution_id=f"exec-{uuid4().hex[:12]}",
                data=[],
                source_id=request.source_id,
                retrieved_at=datetime.now(tz=UTC),
                coverage=SearchCoverage.EXHAUSTED,
                provenance="stub:default-empty",
            )
        item = scripted[index]
        if isinstance(item, ToolResult):
            return item
        status = ToolResultStatus(item["status"])
        error = None
        if "error_code" in item:
            error = ToolError(
                code=ToolErrorCode(item["error_code"]),
                message=item.get("error_message", "stub error"),
            )
        return ToolResult(
            status=status,
            execution_id=item.get("execution_id", f"exec-{uuid4().hex[:12]}"),
            data=item.get("data"),
            error=error,
            source_id=request.source_id or item.get("source_id"),
            retrieved_at=item.get("retrieved_at", datetime.now(tz=UTC)),
            coverage=(
                SearchCoverage(item["coverage"])
                if item.get("coverage") is not None
                else None
            ),
            provenance=item.get("provenance", f"stub:{name}"),
            usage=item.get("usage", {"tool_attempts": 1}),
        )

    def _invoke_temporal(self, request: ToolRequest) -> ToolResult:
        from nanexus_investigation.tools.temporal_analysis import (
            analyze_temporal_relations,
        )

        # Prefer scripted temporal results for deterministic negative tests.
        scripted = self.scripts.get(request.tool_name, [])
        index = self.call_counts.get(request.tool_name, 1) - 1
        if index < len(scripted):
            item = scripted[index]
            if isinstance(item, ToolResult):
                return item

        if self.analysis_store is None or self.analysis_scope is None:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                execution_id=f"exec-temporal-{uuid4().hex[:12]}",
                error=ToolError(
                    code=ToolErrorCode.DISABLED_CAPABILITY,
                    message="temporal analysis store is not bound for this investigation",
                ),
                source_id=request.source_id,
                retrieved_at=datetime.now(tz=UTC),
                coverage=SearchCoverage.UNKNOWN,
                provenance="stub:temporal-unbound",
            )
        return analyze_temporal_relations(
            request,
            observations=self.analysis_store.observations,
            scope=self.analysis_scope,
            max_pairs=self.analysis_max_pairs,
            max_window_seconds=self.analysis_max_window_seconds,
        )

    def _invoke_observe_media(self, request: ToolRequest) -> ToolResult:
        from nanexus_investigation.tools.media_observe import observe_media

        # Prefer scripted observe results for deterministic negative tests.
        scripted = self.scripts.get(request.tool_name, [])
        index = self.call_counts.get(request.tool_name, 1) - 1
        if index < len(scripted):
            item = scripted[index]
            if isinstance(item, ToolResult):
                return item

        if self.analysis_store is None or self.analysis_scope is None:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                execution_id=f"exec-observe-{uuid4().hex[:12]}",
                error=ToolError(
                    code=ToolErrorCode.DISABLED_CAPABILITY,
                    message="observe_media store is not bound for this investigation",
                ),
                source_id=request.source_id,
                retrieved_at=datetime.now(tz=UTC),
                coverage=SearchCoverage.UNKNOWN,
                provenance="stub:observe-unbound",
            )
        if self.media_store is None or not self.media_case_id:
            return ToolResult(
                status=ToolResultStatus.ERROR,
                execution_id=f"exec-observe-{uuid4().hex[:12]}",
                error=ToolError(
                    code=ToolErrorCode.DISABLED_CAPABILITY,
                    message="media retention store is not bound for this investigation",
                ),
                source_id=request.source_id,
                retrieved_at=datetime.now(tz=UTC),
                coverage=SearchCoverage.UNKNOWN,
                provenance="stub:observe-unbound-media",
            )
        return observe_media(
            request,
            evidence_by_id=self.analysis_store.evidence,
            scope=self.analysis_scope,
            case_id=self.media_case_id,
            media_store=self.media_store,
            visual_provider=self.visual_provider,
        )
