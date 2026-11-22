"""Clarification resume validation for native LangGraph interrupt/resume.

A clarification answer may only narrow the authorized investigation within the
existing effective Evidence Grant. It is not permission to expand cameras,
VMS instance, time window, or tools.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from nanexus_investigation.contracts.identity import Interval
from nanexus_investigation.contracts.scope import InvestigationScope
from nanexus_investigation.graph.runtime import InvestigationRuntime


def authorized_time_envelope(runtime: InvestigationRuntime) -> Interval | None:
    """Return the completed time envelope that may not be expanded by resume.

    Completed effective bounds are authoritative. When effective end is open,
    a completed requested/time-context interval may supply the missing end, but
    must not override a narrower effective start (or otherwise widen effective
    authorization).
    """
    case = runtime.case
    effective = case.effective_scope.time_bounds
    if effective.end_utc is not None:
        return effective

    for bounds in (
        case.requested_scope.time_bounds,
        Interval(start_utc=case.time_context.start_utc, end_utc=case.time_context.end_utc),
    ):
        if bounds.end_utc is None:
            continue
        start = max(effective.start_utc, bounds.start_utc)
        end = bounds.end_utc
        if end <= start:
            continue
        return Interval(start_utc=start, end_utc=end)
    return None


def time_constraint_unresolved(runtime: InvestigationRuntime) -> bool:
    """True when intake cannot proceed without a clarified time interval."""
    if runtime.scenario.require_time_clarification:
        return True
    return runtime.case.effective_scope.time_bounds.end_utc is None


def clarification_interrupt_payload(runtime: InvestigationRuntime) -> dict[str, Any]:
    envelope = authorized_time_envelope(runtime)
    case = runtime.case
    return {
        "case_id": case.case_id,
        "clarification_id": "time_bounds",
        "question": (
            "Provide a completed half-open UTC time interval [start_utc, end_utc) "
            "within the authorized investigation scope."
        ),
        "required_fields": ["start_utc", "end_utc"],
        "reason": "investigation time constraint is unresolved",
        "authorized_vms_instance_id": case.effective_scope.vms_instance_id,
        "authorized_camera_ids": list(case.effective_scope.camera_ids),
        "authorized_time_envelope": None
        if envelope is None or envelope.end_utc is None
        else {
            "start_utc": envelope.start_utc.isoformat(),
            "end_utc": envelope.end_utc.isoformat(),
        },
    }


def _parse_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise ValueError("timestamp must be an ISO-8601 string or datetime")


def validate_and_apply_time_clarification(
    runtime: InvestigationRuntime,
    answer: Any,
) -> tuple[bool, str | None]:
    """Validate resume payload and narrow effective scope time bounds in-memory.

    Returns ``(True, None)`` on success. On failure returns ``(False, reason)``
    without widening scope or authorizing tools.
    """
    if not isinstance(answer, dict):
        return False, "clarification resume payload must be an object"

    forbidden_expansions = []
    if "tool_name" in answer or "tools" in answer:
        forbidden_expansions.append("clarification must not introduce tools")
    if "vms_instance_id" in answer:
        requested_vms = str(answer["vms_instance_id"])
        if requested_vms != runtime.case.effective_scope.vms_instance_id:
            forbidden_expansions.append("clarification must not change VMS instance")
    if "camera_ids" in answer:
        cameras = answer["camera_ids"]
        if not isinstance(cameras, (list, tuple)):
            return False, "camera_ids must be a list when provided"
        unauthorized = [
            camera_id
            for camera_id in cameras
            if camera_id not in runtime.case.effective_scope.camera_ids
        ]
        if unauthorized:
            forbidden_expansions.append(
                f"clarification requests unauthorized cameras: {unauthorized}"
            )
    if forbidden_expansions:
        return False, "; ".join(forbidden_expansions)

    try:
        start = _parse_datetime(answer.get("start_utc"))
        end = _parse_datetime(answer.get("end_utc"))
        clarified = Interval(start_utc=start, end_utc=end)
    except Exception as exc:  # noqa: BLE001 — surface validation to caller
        return False, f"invalid time clarification: {exc}"

    envelope = authorized_time_envelope(runtime)
    if envelope is None or envelope.end_utc is None:
        return False, "no completed authorized time envelope is available"

    # Build a temporary scope with the envelope to reuse contains_interval.
    grant = InvestigationScope(
        vms_instance_id=runtime.case.effective_scope.vms_instance_id,
        camera_ids=runtime.case.effective_scope.camera_ids,
        time_bounds=envelope,
    )
    if not grant.contains_interval(clarified):
        return False, "clarified time interval expands or escapes authorized scope"

    new_scope = runtime.case.effective_scope.model_copy(
        update={"time_bounds": clarified}
    )
    runtime.case = runtime.case.model_copy(update={"effective_scope": new_scope})
    runtime.store.case = runtime.case
    runtime.trace.record(
        "clarification_accepted",
        clarification_id="time_bounds",
        start_utc=clarified.start_utc.isoformat(),
        end_utc=clarified.end_utc.isoformat(),
    )
    return True, None
