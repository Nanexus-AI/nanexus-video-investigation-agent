"""InvestigationState TypedDict compatible with G3-B §4.2."""

from __future__ import annotations

import operator
from typing import Annotated, Any, NotRequired, TypedDict


def _merge_unique(left: list[str], right: list[str]) -> list[str]:
    seen = set(left)
    merged = list(left)
    for item in right:
        if item not in seen:
            seen.add(item)
            merged.append(item)
    return merged


class InvestigationState(TypedDict):
    """Small serializable workflow values and durable-record references.

    Raw media, credentials, and unrestricted tool payloads are forbidden.
    """

    schema_version: str
    case_id: str
    phase: str
    effective_scope_ref: str
    time_context_ref: str
    capability_ref: NotRequired[str]
    plan: list[dict[str, Any]]
    plan_index: int
    pending_tool_request: dict[str, Any] | None
    tool_execution_ids: Annotated[list[str], _merge_unique]
    observation_ids: Annotated[list[str], _merge_unique]
    evidence_ids: Annotated[list[str], _merge_unique]
    search_receipt_ids: Annotated[list[str], _merge_unique]
    relation_ids: Annotated[list[str], _merge_unique]
    candidate_claim_ids: Annotated[list[str], _merge_unique]
    validated_claim_ids: Annotated[list[str], _merge_unique]
    coverage_summary: list[str]
    budget_usage: dict[str, int]
    attempt_summary: dict[str, int]
    last_error: str | None
    pending_clarification: NotRequired[str | None]
    time_clarification_applied: NotRequired[bool]
    report_id: str | None
    route: NotRequired[str]
    last_tool_status: NotRequired[str | None]
    last_match_count: NotRequired[int]
    max_plan_steps: int
    model_proposal: NotRequired[dict[str, Any] | None]
