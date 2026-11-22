"""StateGraph node functions for the E0 deterministic investigation slice."""

from __future__ import annotations

from typing import Any, Literal

from langgraph.types import interrupt

from nanexus_investigation.contracts.enums import CaseStatus, ToolResultStatus
from nanexus_investigation.graph.clarification import (
    clarification_interrupt_payload,
    time_constraint_unresolved,
    validate_and_apply_time_clarification,
)
from nanexus_investigation.graph.runtime import (
    DomainRegistrationError,
    InvestigationRuntime,
    assemble_report,
    draft_candidate_claims,
    evaluate_temporal,
    register_tool_artifacts,
    validate_claims,
)
from nanexus_investigation.graph.state import InvestigationState
from nanexus_investigation.harness.durable_budget import DurableBudgetAdapter
from nanexus_investigation.harness.output_gate import run_output_gate
from nanexus_investigation.harness.tool_guard import ToolGuard
from nanexus_investigation.persistence.operation_id import (
    build_attempt_identity,
    build_operation_identity,
)


RouteName = Literal["continue_search", "proceed_to_report", "terminate"]
IntakeRoute = Literal["discover_capabilities", "terminate"]


def _node(runtime: InvestigationRuntime, name: str) -> None:
    runtime.trace.record("node", node=name)


def validate_intake(runtime: InvestigationRuntime, state: InvestigationState) -> dict[str, Any]:
    """Validate intake; interrupt for unresolved time before any tool work.

    Replay-safe: no tool invocation or budget reservation occurs in this node.
    ``interrupt()`` may re-enter; clarification application is idempotent.
    """
    _node(runtime, "validate_intake")
    case = runtime.case
    if case.status is CaseStatus.CREATED:
        runtime.case = case.transition_to(CaseStatus.RUNNING)
        runtime.store.case = runtime.case
        case = runtime.case

    if not runtime.case.effective_scope.camera_ids:
        return {
            "phase": "intake_failed",
            "last_error": "effective scope requires cameras",
            "route": "terminate",
        }

    clarification_done = bool(state.get("time_clarification_applied"))
    if time_constraint_unresolved(runtime) and not clarification_done:
        # Mark awaiting_input before interrupt; no external tool side effects.
        if runtime.case.status is CaseStatus.RUNNING:
            runtime.case = runtime.case.transition_to(CaseStatus.AWAITING_INPUT)
            runtime.store.case = runtime.case
        payload = clarification_interrupt_payload(runtime)
        runtime.trace.record(
            "clarification_interrupt",
            clarification_id=payload["clarification_id"],
            reason=payload["reason"],
            case_id=payload["case_id"],
        )
        # Native LangGraph pattern: re-interrupt after rejection so the same
        # Case/thread can accept a corrected clarification without terminating
        # into a stranded awaiting_input Case.
        while True:
            answer = interrupt(payload)
            accepted, error = validate_and_apply_time_clarification(runtime, answer)
            if accepted:
                break
            runtime.trace.record(
                "clarification_rejected",
                reason=error,
                case_id=runtime.case.case_id,
            )
            payload = {
                **clarification_interrupt_payload(runtime),
                "last_rejection": error or "clarification rejected",
            }
            runtime.trace.record(
                "clarification_interrupt",
                clarification_id=payload["clarification_id"],
                reason="retry_after_rejection",
                case_id=payload["case_id"],
            )

        if runtime.case.status is CaseStatus.AWAITING_INPUT:
            runtime.case = runtime.case.transition_to(CaseStatus.RUNNING)
            runtime.store.case = runtime.case
        # Rebuild Tool Guard against the narrowed effective scope.
        runtime.guard = ToolGuard(
            registry=runtime.registry,
            budget=runtime.budget,
            effective_scope=runtime.case.effective_scope,
            capabilities=runtime.capabilities,
        )
        return {
            "phase": "intake_validated",
            "effective_scope_ref": f"scope:{runtime.case.case_id}",
            "time_context_ref": f"time:{runtime.case.case_id}",
            "pending_clarification": None,
            "time_clarification_applied": True,
            "last_error": None,
            "route": None,
        }

    if runtime.case.effective_scope.time_bounds.end_utc is None:
        return {
            "phase": "intake_failed",
            "last_error": "effective scope requires completed time bounds",
            "route": "terminate",
        }

    return {
        "phase": "intake_validated",
        "effective_scope_ref": f"scope:{runtime.case.case_id}",
        "time_context_ref": f"time:{runtime.case.case_id}",
        "last_error": None,
        "route": None,
    }


def route_after_intake(state: InvestigationState) -> IntakeRoute:
    if state.get("route") == "terminate" or state.get("phase") in {
        "intake_failed",
        "clarification_rejected",
    }:
        return "terminate"
    return "discover_capabilities"


def discover_capabilities(
    runtime: InvestigationRuntime, state: InvestigationState
) -> dict[str, Any]:
    _node(runtime, "discover_capabilities")
    runtime.case = runtime.case.model_copy(
        update={"capability_snapshot": runtime.capabilities}
    )
    runtime.store.case = runtime.case
    runtime.trace.record(
        "capabilities",
        operations=[op.operation for op in runtime.capabilities.operations],
    )
    return {
        "phase": "capabilities_discovered",
        "capability_ref": f"caps:{runtime.case.case_id}",
    }


def _authorized_supported_tools(runtime: InvestigationRuntime) -> tuple[str, ...]:
    from nanexus_investigation.contracts.enums import CapabilityAvailability
    from nanexus_investigation.tools.registry import ALLOWED_TOOLS

    return tuple(
        sorted(
            tool
            for tool in ALLOWED_TOOLS
            if (
                (cap := runtime.capabilities.for_operation(tool)) is not None
                and cap.availability is CapabilityAvailability.SUPPORTED
            )
        )
    )


def _proposal_context_base(
    runtime: InvestigationRuntime,
    *,
    prior_tool_name: str | None,
    prior_tool_status: str | None,
    prior_match_count: int,
    prior_result_summary: str,
    evidence_notes: tuple[str, ...] = (),
    coverage_notes: tuple[str, ...] = (),
):
    from nanexus_investigation.model.provider import (
        ProposalContext,
        default_tool_descriptions,
    )

    scope = runtime.case.effective_scope
    bounds = scope.time_bounds
    authorized_tools = _authorized_supported_tools(runtime)
    return ProposalContext(
        question=runtime.case.question,
        case_id=runtime.case.case_id,
        authorized_tools=authorized_tools,
        authorized_cameras=tuple(scope.camera_ids),
        vms_instance_id=scope.vms_instance_id,
        time_bounds={
            "start_utc": bounds.start_utc.isoformat(),
            "end_utc": None if bounds.end_utc is None else bounds.end_utc.isoformat(),
        },
        prior_tool_name=prior_tool_name,
        prior_tool_status=prior_tool_status,
        prior_match_count=prior_match_count,
        prior_result_summary=prior_result_summary,
        evidence_notes=evidence_notes,
        coverage_notes=coverage_notes,
        tool_descriptions=default_tool_descriptions(authorized_tools),
    )


def _consume_model_call(runtime: InvestigationRuntime) -> bool:
    """Reserve one model consultation against max_model_calls. Returns False if exhausted."""
    if runtime.model_calls >= runtime.scenario.max_model_calls:
        return False
    runtime.model_calls += 1
    return True


def _record_model_proposal(runtime: InvestigationRuntime, proposal: Any) -> None:
    runtime.last_model_proposal = proposal
    runtime.model_proposals.append(proposal)


def _plan_step_from_proposal(
    *,
    step_id: str,
    proposal: Any,
    scope_cameras: tuple[str, ...],
    scope_bounds: Any,
):
    from datetime import datetime

    from nanexus_investigation.contracts.identity import Interval
    from nanexus_investigation.graph.planner import PlanStep

    cameras = proposal.camera_ids or scope_cameras[:1]
    follow_bounds = scope_bounds
    if proposal.time_bounds is not None:
        start_raw = proposal.time_bounds.get("start_utc")
        end_raw = proposal.time_bounds.get("end_utc")
        if not isinstance(start_raw, str):
            raise ValueError("model time_bounds.start_utc missing or invalid")
        start_dt = datetime.fromisoformat(start_raw)
        end_dt = None if end_raw is None else datetime.fromisoformat(str(end_raw))
        follow_bounds = Interval(start_utc=start_dt, end_utc=end_dt)
    tool_args = None
    if hasattr(proposal, "analysis_arguments"):
        tool_args = proposal.analysis_arguments()
    if tool_args is None and hasattr(proposal, "media_arguments"):
        tool_args = proposal.media_arguments()
    if tool_args is None and hasattr(proposal, "observe_arguments"):
        tool_args = proposal.observe_arguments()
    return PlanStep(
        step_id=step_id,
        tool_name=str(proposal.tool_name),
        label=proposal.label,
        camera_ids=tuple(cameras),
        time_bounds=follow_bounds,
        requires_prior_matches=False,
        arguments=tool_args,
    )


def _build_model_initial_plan(
    runtime: InvestigationRuntime,
) -> tuple[list[Any] | None, str | None, dict[str, Any] | None]:
    """Return (steps, error, model_proposal_dict)."""
    from nanexus_investigation.model.provider import ProposalParseError
    from nanexus_investigation.tools.registry import ALLOWED_TOOLS

    if runtime.model_provider is None:
        return None, "model initial plan enabled without a Model Provider", None
    if not _consume_model_call(runtime):
        return None, "model-call budget exhausted before initial plan", None

    context = _proposal_context_base(
        runtime,
        prior_tool_name=None,
        prior_tool_status=None,
        prior_match_count=0,
        prior_result_summary="(none — initial plan)",
    )
    propose = getattr(runtime.model_provider, "propose_initial_plan", None)
    try:
        if callable(propose):
            proposal = propose(context)
        else:
            proposal = runtime.model_provider.propose_follow_up(context)
    except ProposalParseError as exc:
        return None, f"invalid initial model proposal: {exc}", None
    except Exception as exc:  # noqa: BLE001 — fail closed on provider faults
        return None, f"model provider failure during initial plan: {exc}", None

    _record_model_proposal(runtime, proposal)
    runtime.trace.record(
        "model_initial_proposal",
        action=proposal.action,
        tool_name=proposal.tool_name,
        label=proposal.label,
        rationale=proposal.rationale,
        time_bounds=proposal.time_bounds,
        case_id=runtime.case.case_id,
        authorized_tools=list(context.authorized_tools),
        model_calls=runtime.model_calls,
    )

    if proposal.action != "invoke_tool" or not proposal.tool_name:
        return (
            None,
            (
                "initial model plan must invoke an authorized tool; "
                f"got action={proposal.action!r} tool={proposal.tool_name!r}"
            ),
            proposal.to_dict(),
        )

    if (
        proposal.tool_name not in context.authorized_tools
        and proposal.tool_name in ALLOWED_TOOLS
    ):
        return (
            None,
            (
                f"model proposed unsupported or unauthorized tool: "
                f"{proposal.tool_name!r}; authorized={list(context.authorized_tools)}"
            ),
            proposal.to_dict(),
        )

    try:
        step = _plan_step_from_proposal(
            step_id="model-step-1",
            proposal=proposal,
            scope_cameras=runtime.case.effective_scope.camera_ids,
            scope_bounds=runtime.case.effective_scope.time_bounds,
        )
    except ValueError as exc:
        return None, str(exc), proposal.to_dict()

    return [step], None, proposal.to_dict()


def build_plan(runtime: InvestigationRuntime, state: InvestigationState) -> dict[str, Any]:
    _node(runtime, "build_plan")
    model_proposal_dict: dict[str, Any] | None = None
    if runtime.scenario.use_model_initial_plan:
        steps, error, model_proposal_dict = _build_model_initial_plan(runtime)
        if error or steps is None:
            runtime.trace.record(
                "plan_rejected",
                reason=error,
                model_initial_plan=True,
            )
            return {
                "phase": "plan_rejected",
                "plan": [],
                "plan_index": 0,
                "max_plan_steps": runtime.scenario.max_plan_steps,
                "attempt_summary": {"tool_attempts": 0, "plan_steps": 0},
                "last_error": error,
                "route": "terminate",
                "model_proposal": model_proposal_dict,
            }
        runtime.plan_steps = list(steps)
        deterministic = False
    else:
        steps = runtime.build_plan()
        deterministic = True

    max_steps = runtime.scenario.max_plan_steps
    if len(steps) > max_steps:
        runtime.trace.record(
            "plan_truncated",
            requested=len(steps),
            max_plan_steps=max_steps,
        )
        steps = steps[:max_steps]
        runtime.plan_steps = steps
    runtime.trace.record(
        "plan_built",
        steps=[step.step_id for step in steps],
        tools=[step.tool_name for step in steps],
        labels=[step.label for step in steps],
        deterministic=deterministic,
        model_initial_plan=runtime.scenario.use_model_initial_plan,
        model_calls=runtime.model_calls,
    )
    update: dict[str, Any] = {
        "phase": "plan_ready",
        "plan": [step.to_dict() for step in steps],
        "plan_index": 0,
        "max_plan_steps": runtime.scenario.max_plan_steps,
        "attempt_summary": {"tool_attempts": 0, "plan_steps": len(steps)},
    }
    if model_proposal_dict is not None:
        update["model_proposal"] = model_proposal_dict
    return update


def select_validate_tool(
    runtime: InvestigationRuntime, state: InvestigationState
) -> dict[str, Any]:
    _node(runtime, "select_validate_tool")
    if state.get("phase") == "plan_rejected" and state.get("last_error"):
        # Preserve initial-plan failure through select → sufficiency terminate.
        runtime.trace.record(
            "guard_skip",
            reason="plan_rejected",
            detail=state.get("last_error"),
        )
        return {
            "phase": "plan_rejected",
            "pending_tool_request": None,
            "last_error": state.get("last_error"),
            "route": "terminate",
        }

    plan = state.get("plan") or []
    index = int(state.get("plan_index") or 0)
    if index >= len(plan):
        runtime.trace.record("guard_skip", reason="no remaining plan steps")
        return {
            "phase": "no_tool_selected",
            "pending_tool_request": None,
            "last_error": state.get("last_error") or "no remaining plan steps",
            "route": "terminate",
        }

    step = plan[index]
    from nanexus_investigation.graph.request_fingerprint import tool_request_fingerprint

    source_id = runtime.case.effective_scope.vms_instance_id
    step_arguments = dict(step.get("arguments") or {})
    if step.get("label") is not None and "label" not in step_arguments:
        step_arguments["label"] = step.get("label")
    fingerprint = tool_request_fingerprint(
        tool_name=str(step["tool_name"]),
        source_id=source_id,
        camera_ids=step.get("camera_ids") or runtime.case.effective_scope.camera_ids,
        label=step.get("label"),
        time_bounds=step.get("time_bounds") or runtime.case.effective_scope.time_bounds,
        extra_arguments=step_arguments,
    )
    if fingerprint in runtime.executed_request_fingerprints:
        runtime.trace.record(
            "anti_repeat_skip",
            plan_step_id=step.get("step_id"),
            tool_name=step.get("tool_name"),
            label=step.get("label"),
            fingerprint=fingerprint[:16],
        )
        return {
            "phase": "duplicate_tool_skipped",
            "pending_tool_request": None,
            "plan_index": index + 1,
            "last_error": None,
        }

    proposal = {
        # Case-scoped: durable budget reservations share one domain DB across
        # Web Cases; unscoped request_ids collide on consecutive investigations.
        "request_id": f"req-{runtime.case.case_id}-{step['step_id']}-{index}",
        "tool_name": step["tool_name"],
        "label": step.get("label"),
        "camera_ids": step.get("camera_ids"),
        "time_bounds": step.get("time_bounds"),
        "source_id": source_id,
        "arguments": step_arguments,
    }
    runtime.trace.record("tool_proposal", **proposal)
    decision = runtime.guard.validate(proposal)
    runtime.last_guard = decision
    if decision.rejected:
        runtime.trace.record(
            "guard_rejected",
            request_id=decision.request_id,
            reasons=list(decision.reasons),
        )
        return {
            "phase": "tool_rejected",
            "pending_tool_request": None,
            "last_error": "; ".join(decision.reasons),
            "route": "terminate",
            "budget_usage": {
                "reserved": runtime.budget.reserved,
                "consumed": runtime.budget.consumed,
                "max": runtime.budget.max_attempts,
            },
        }

    # Bind durable attempt identity before reservation commit (SQLite path only).
    if isinstance(runtime.budget, DurableBudgetAdapter) and decision.request is not None:
        operation = build_operation_identity(
            case_id=runtime.case.case_id,
            plan_step_id=str(step["step_id"]),
            tool_name=decision.request.tool_name,
            vms_instance_id=decision.request.source_id
            or runtime.case.effective_scope.vms_instance_id,
            request_parameters={
                "arguments": decision.request.arguments,
                "camera_ids": list(decision.request.camera_ids),
            },
        )
        attempt = build_attempt_identity(
            operation_id=operation.operation_id,
            attempt_index=index + 1,
        )
        runtime.budget.bind(
            request_id=decision.request_id,
            operation=operation,
            attempt=attempt,
        )

    reserved = runtime.guard.reserve(decision)
    reservation_meta: dict[str, Any] = {
        "request_id": decision.request_id,
        "reserved": reserved,
        "remaining": runtime.budget.remaining,
    }
    if isinstance(runtime.budget, DurableBudgetAdapter):
        reservation_meta["durable"] = True
        if runtime.budget.last_error:
            reservation_meta["error"] = runtime.budget.last_error
    runtime.trace.record("budget_reservation", **reservation_meta)
    if not reserved:
        reason = "tool-attempt budget exhausted"
        if isinstance(runtime.budget, DurableBudgetAdapter) and runtime.budget.last_error:
            reason = runtime.budget.last_error
        runtime.trace.record(
            "guard_rejected",
            request_id=decision.request_id,
            reasons=[reason],
        )
        phase = (
            "unresolved_attempt_blocked"
            if "unresolved" in reason or "unknown execution outcome" in reason
            else "budget_exhausted"
        )
        return {
            "phase": phase,
            "pending_tool_request": None,
            "last_error": reason,
            "route": "terminate",
            "budget_usage": {
                "reserved": runtime.budget.reserved,
                "consumed": runtime.budget.consumed,
                "max": runtime.budget.max_attempts,
            },
        }

    assert decision.request is not None
    pending = {
        "request_id": decision.request_id,
        "tool_name": decision.request.tool_name,
        "arguments": decision.request.arguments,
        "source_id": decision.request.source_id,
        "camera_ids": list(decision.request.camera_ids),
        "plan_step_id": step["step_id"],
    }
    runtime.trace.record("guard_accepted", request_id=decision.request_id)
    return {
        "phase": "tool_validated",
        "pending_tool_request": pending,
        "last_error": None,
        "budget_usage": {
            "reserved": runtime.budget.reserved,
            "consumed": runtime.budget.consumed,
            "max": runtime.budget.max_attempts,
        },
    }


def execute_tool(runtime: InvestigationRuntime, state: InvestigationState) -> dict[str, Any]:
    _node(runtime, "execute_tool")
    pending = state.get("pending_tool_request")
    if not pending:
        return {
            "phase": "execute_skipped",
            "last_error": state.get("last_error") or "no pending tool request",
            "route": "terminate",
        }

    from nanexus_investigation.contracts.tool import ToolRequest

    request = ToolRequest(
        tool_name=pending["tool_name"],
        arguments=dict(pending.get("arguments") or {}),
        source_id=pending.get("source_id"),
        camera_ids=tuple(pending.get("camera_ids") or ()),
    )
    # No LangGraph node retry policy is attached.
    assert runtime.retry_policy_attached is False

    # N6: never auto-dispatch when durable adapter has blocked the request.
    if isinstance(runtime.budget, DurableBudgetAdapter):
        blocked = runtime.budget.invoke_blocked_reason(pending["request_id"])
        if blocked:
            runtime.trace.record(
                "tool_dispatch_blocked",
                request_id=pending["request_id"],
                reason=blocked,
            )
            return {
                "phase": "unresolved_attempt_blocked",
                "pending_tool_request": None,
                "last_error": blocked,
                "route": "terminate",
            }

    try:
        result = runtime.registry.invoke(request)
    except Exception:
        if isinstance(runtime.budget, DurableBudgetAdapter):
            runtime.budget.mark_failed(pending["request_id"])
        raise

    from nanexus_investigation.graph.request_fingerprint import tool_request_fingerprint

    label_raw = request.arguments.get("label")
    label = label_raw if isinstance(label_raw, str) else None
    runtime.executed_request_fingerprints.add(
        tool_request_fingerprint(
            tool_name=request.tool_name,
            source_id=request.source_id
            or runtime.case.effective_scope.vms_instance_id,
            camera_ids=request.camera_ids,
            label=label,
            time_bounds=request.arguments.get("time_bounds")
            or runtime.case.effective_scope.time_bounds,
            extra_arguments=request.arguments,
        )
    )

    runtime._last_result = result  # noqa: SLF001 — E0 handoff to register node
    runtime.budget.mark_consumed(pending["request_id"])
    if isinstance(runtime.budget, DurableBudgetAdapter):
        # execution_id is required confirmation evidence; not proof by itself
        # that a physical network dispatch occurred (stub path only).
        runtime.budget.mark_confirmed_with_execution(
            pending["request_id"],
            execution_id=result.execution_id,
        )
    runtime.trace.record(
        "tool_executed",
        request_id=pending["request_id"],
        tool_name=request.tool_name,
        execution_id=result.execution_id,
        status=result.status.value,
    )
    attempts = int((state.get("attempt_summary") or {}).get("tool_attempts", 0)) + 1
    match_count = len(result.data) if isinstance(result.data, list) else 0
    return {
        "phase": "tool_executed",
        "tool_execution_ids": [result.execution_id],
        "pending_tool_request": {**pending, "execution_id": result.execution_id},
        "last_tool_status": result.status.value,
        "last_match_count": match_count,
        "attempt_summary": {
            **(state.get("attempt_summary") or {}),
            "tool_attempts": attempts,
        },
        "budget_usage": {
            "reserved": runtime.budget.reserved,
            "consumed": runtime.budget.consumed,
            "max": runtime.budget.max_attempts,
        },
    }


def normalize_register_evidence(
    runtime: InvestigationRuntime, state: InvestigationState
) -> dict[str, Any]:
    _node(runtime, "normalize_register_evidence")
    from nanexus_investigation.contracts.tool import ToolRequest

    pending = state.get("pending_tool_request") or {}
    if not pending.get("tool_name"):
        return {
            "phase": "evidence_skipped",
            "coverage_summary": list(state.get("coverage_summary") or []),
        }

    result = getattr(runtime, "_last_result", None)
    if result is None:
        return {
            "phase": "evidence_failed",
            "last_error": "missing tool result for evidence registration",
        }

    request = ToolRequest(
        tool_name=pending["tool_name"],
        arguments=dict(pending.get("arguments") or {}),
        source_id=pending.get("source_id"),
        camera_ids=tuple(pending.get("camera_ids") or ()),
    )
    try:
        obs_ids, ev_ids, receipt_ids, notes = register_tool_artifacts(
            runtime,
            request=request,
            result=result,
            plan_step_id=pending.get("plan_step_id"),
        )
    except DomainRegistrationError as exc:
        runtime._last_result = None  # noqa: SLF001
        runtime.trace.record(
            "evidence_registration_aborted",
            error=str(exc),
            execution_id=result.execution_id,
        )
        return {
            "phase": "evidence_registration_failed",
            "pending_tool_request": None,
            "last_error": str(exc),
            "route": "terminate",
            # Do not publish uncommitted evidence IDs into graph/checkpoint state.
            "observation_ids": [],
            "evidence_ids": [],
            "search_receipt_ids": [],
        }

    coverage = list(state.get("coverage_summary") or []) + notes
    runtime._last_result = None  # noqa: SLF001
    return {
        "phase": "evidence_registered",
        "observation_ids": obs_ids,
        "evidence_ids": ev_ids,
        "search_receipt_ids": receipt_ids,
        "coverage_summary": coverage,
        "pending_tool_request": None,
        "plan_index": int(state.get("plan_index") or 0) + 1,
    }


def evaluate_temporal_relations(
    runtime: InvestigationRuntime, state: InvestigationState
) -> dict[str, Any]:
    _node(runtime, "evaluate_temporal_relations")
    relation_ids = evaluate_temporal(runtime)
    return {
        "phase": "temporal_evaluated",
        "relation_ids": relation_ids,
    }


def _prior_result_summary(runtime: InvestigationRuntime, state: InvestigationState) -> str:
    result = getattr(runtime, "_last_result", None)
    # _last_result is cleared after evidence registration; summarize from store.
    observations = list(runtime.store.observations.values()) if runtime.store.observations else []
    if observations:
        labels: list[str] = []
        samples: list[str] = []
        for obs in observations[:12]:
            if obs.labels:
                labels.extend(str(label) for label in obs.labels)
            else:
                labels.append(obs.observation_type.value)
            start = obs.start_utc.isoformat()
            end = None if obs.end_utc is None else obs.end_utc.isoformat()
            label = obs.labels[0] if obs.labels else obs.observation_type.value
            samples.append(f"{label}@{start}..{end}")
        unique_labels = sorted(set(labels))
        observation_ids = [obs.observation_id for obs in observations[:12]]
        evidence_ids = list(runtime.store.evidence.keys())[:12]
        from nanexus_investigation.contracts.enums import EvidenceKind

        snapshot_ids = [
            evidence.evidence_id
            for evidence in runtime.store.evidence.values()
            if evidence.evidence_kind is EvidenceKind.SNAPSHOT
        ][:8]
        return (
            f"{len(observations)} registered observation(s); "
            f"labels={unique_labels[:8]}; "
            f"observation_ids={observation_ids}; "
            f"evidence_ids={evidence_ids}; "
            f"snapshot_evidence_ids={snapshot_ids}; "
            f"samples={samples[:6]}; "
            f"last_status={state.get('last_tool_status')}; "
            f"last_match_count={state.get('last_match_count')}"
        )
    if result is not None:
        data = result.data if isinstance(result.data, list) else []
        return (
            f"tool_status={result.status.value}; matches={len(data)}; "
            f"coverage={None if result.coverage is None else result.coverage.value}"
        )
    return (
        f"last_status={state.get('last_tool_status')}; "
        f"last_match_count={state.get('last_match_count')}"
    )


def _apply_model_follow_up(
    runtime: InvestigationRuntime, state: InvestigationState
) -> dict[str, Any] | None:
    """Consult the local Model Provider for budgeted evidence-dependent replanning.

    Returns a sufficiency update dict when the model path handles the decision,
    otherwise ``None`` so the deterministic E0 sufficiency rules apply.
    """
    if not runtime.scenario.use_model_follow_up:
        return None
    attempts = int((state.get("attempt_summary") or {}).get("tool_attempts", 0))
    if attempts < 1:
        return None
    if runtime.model_calls >= runtime.scenario.max_model_calls:
        runtime.trace.record(
            "model_follow_up_budget_exhausted",
            model_calls=runtime.model_calls,
            max_model_calls=runtime.scenario.max_model_calls,
        )
        runtime.note_planning_limitation(
            "partial investigation stop: model-call budget exhausted; "
            "further replanning was not performed"
        )
        return None
    if runtime.model_provider is None:
        runtime.trace.record(
            "model_follow_up_unavailable",
            reason="use_model_follow_up set but no model_provider configured",
        )
        return {
            "phase": "sufficiency_assessed",
            "route": "terminate",
            "last_error": "model follow-up enabled without a Model Provider",
            "model_proposal": None,
        }

    from nanexus_investigation.graph.request_fingerprint import tool_request_fingerprint
    from nanexus_investigation.model.provider import ProposalParseError
    from nanexus_investigation.tools.registry import ALLOWED_TOOLS

    authorized_tools = _authorized_supported_tools(runtime)
    prior_tool = None
    if runtime.registry.invocations:
        prior_tool = runtime.registry.invocations[-1]
    context = _proposal_context_base(
        runtime,
        prior_tool_name=prior_tool,
        prior_tool_status=state.get("last_tool_status"),
        prior_match_count=int(state.get("last_match_count") or 0),
        prior_result_summary=_prior_result_summary(runtime, state),
        evidence_notes=tuple(state.get("evidence_ids") or ())[:12],
        coverage_notes=tuple(state.get("coverage_summary") or ())[:8],
    )

    if not _consume_model_call(runtime):
        return None

    try:
        proposal = runtime.model_provider.propose_follow_up(context)
    except ProposalParseError as exc:
        runtime.trace.record(
            "model_proposal_invalid",
            error=str(exc),
            case_id=runtime.case.case_id,
        )
        # Prefer an evidence-grounded partial report over hard failure when the
        # Case already holds registered observations from successful tools.
        if runtime.store.observations:
            runtime.note_planning_limitation(
                f"partial investigation stop: invalid model follow-up proposal "
                f"({exc}); reporting from evidence already registered"
            )
            return {
                "phase": "sufficiency_assessed",
                "route": "proceed_to_report",
                "last_error": None,
                "model_proposal": None,
            }
        return {
            "phase": "sufficiency_assessed",
            "route": "terminate",
            "last_error": f"invalid model proposal: {exc}",
            "model_proposal": None,
        }
    except Exception as exc:  # noqa: BLE001 — fail closed on provider faults
        runtime.trace.record(
            "model_proposal_failed",
            error=str(exc),
            case_id=runtime.case.case_id,
        )
        return {
            "phase": "sufficiency_assessed",
            "route": "terminate",
            "last_error": f"model provider failure: {exc}",
            "model_proposal": None,
        }

    _record_model_proposal(runtime, proposal)
    runtime.trace.record(
        "model_proposal",
        action=proposal.action,
        tool_name=proposal.tool_name,
        label=proposal.label,
        rationale=proposal.rationale,
        time_bounds=proposal.time_bounds,
        case_id=runtime.case.case_id,
        authorized_tools=list(authorized_tools),
        model_calls=runtime.model_calls,
    )

    if proposal.action == "proceed_to_report":
        return {
            "phase": "sufficiency_assessed",
            "route": "proceed_to_report",
            "model_proposal": proposal.to_dict(),
            "last_error": None,
        }

    # Allowlisted-but-UNSUPPORTED tools (e.g. Frigate search_related_events) are
    # rejected here — never silently stubbed. Unknown tool names still reach
    # Tool Guard so existing unauthorized-tool rejection semantics remain.
    if (
        proposal.tool_name not in authorized_tools
        and proposal.tool_name in ALLOWED_TOOLS
    ):
        runtime.trace.record(
            "model_proposal_unsupported_tool",
            tool_name=proposal.tool_name,
            authorized_tools=list(authorized_tools),
        )
        return {
            "phase": "sufficiency_assessed",
            "route": "terminate",
            "last_error": (
                f"model proposed unsupported or unauthorized tool: "
                f"{proposal.tool_name!r}; authorized={list(authorized_tools)}"
            ),
            "model_proposal": proposal.to_dict(),
        }

    # invoke_tool: append a guarded plan step; Tool Guard still decides execution.
    plan = list(state.get("plan") or [])
    index = int(state.get("plan_index") or 0)
    max_steps = int(state.get("max_plan_steps") or runtime.scenario.max_plan_steps)
    if index >= max_steps or not runtime.budget.can_reserve():
        runtime.note_planning_limitation(
            "partial investigation stop: plan-step or tool-attempt budget exhausted; "
            "model-proposed follow-up was not executed"
        )
        return {
            "phase": "sufficiency_assessed",
            "route": "proceed_to_report",
            "model_proposal": proposal.to_dict(),
            "last_error": None,
        }

    scope = runtime.case.effective_scope
    try:
        step = _plan_step_from_proposal(
            step_id=f"model-step-{index + 1}",
            proposal=proposal,
            scope_cameras=scope.camera_ids,
            scope_bounds=scope.time_bounds,
        )
    except ValueError as exc:
        return {
            "phase": "sufficiency_assessed",
            "route": "terminate",
            "last_error": str(exc),
            "model_proposal": proposal.to_dict(),
        }

    fingerprint = tool_request_fingerprint(
        tool_name=step.tool_name,
        source_id=scope.vms_instance_id,
        camera_ids=step.camera_ids or scope.camera_ids,
        label=step.label,
        time_bounds=step.time_bounds or scope.time_bounds,
        extra_arguments=step.arguments,
    )
    if fingerprint in runtime.executed_request_fingerprints:
        runtime.trace.record(
            "anti_repeat_blocked",
            tool_name=step.tool_name,
            label=step.label,
            fingerprint=fingerprint[:16],
            rationale=proposal.rationale,
        )
        runtime.note_planning_limitation(
            "partial investigation stop: identical tool request already executed; "
            "duplicate retrieval blocked before Tool Guard reservation"
        )
        return {
            "phase": "sufficiency_assessed",
            "route": "proceed_to_report",
            "model_proposal": proposal.to_dict(),
            "last_error": None,
        }

    # Ensure the next select_validate_tool reads this step at current index.
    if index < len(plan):
        plan = plan[:index] + [step.to_dict()] + plan[index:]
    else:
        plan = plan + [step.to_dict()]
    runtime.plan_steps = list(runtime.plan_steps) + [step]
    follow_bounds = step.time_bounds or scope.time_bounds
    runtime.trace.record(
        "model_follow_up_planned",
        step_id=step.step_id,
        tool_name=step.tool_name,
        label=step.label,
        plan_index=index,
        time_bounds={
            "start_utc": follow_bounds.start_utc.isoformat(),
            "end_utc": (
                None
                if follow_bounds.end_utc is None
                else follow_bounds.end_utc.isoformat()
            ),
        },
    )
    return {
        "phase": "sufficiency_assessed",
        "route": "continue_search",
        "plan": plan,
        "model_proposal": proposal.to_dict(),
        "last_error": None,
    }


def assess_sufficiency(
    runtime: InvestigationRuntime, state: InvestigationState
) -> dict[str, Any]:
    _node(runtime, "assess_sufficiency")
    plan = state.get("plan") or []
    index = int(state.get("plan_index") or 0)
    last_status = state.get("last_tool_status")
    last_matches = int(state.get("last_match_count") or 0)
    attempts = int((state.get("attempt_summary") or {}).get("tool_attempts", 0))
    max_steps = int(state.get("max_plan_steps") or runtime.scenario.max_plan_steps)

    if state.get("last_error") and state.get("phase") in {
        "tool_rejected",
        "budget_exhausted",
        "unresolved_attempt_blocked",
        "evidence_registration_failed",
        "intake_failed",
        "plan_rejected",
        "execute_skipped",
    }:
        route: RouteName = "terminate"
        runtime.trace.record(
            "sufficiency",
            route=route,
            plan_index=index,
            last_status=last_status,
            last_matches=last_matches,
            budget_remaining=runtime.budget.remaining,
        )
        return {"phase": "sufficiency_assessed", "route": route}

    model_update = _apply_model_follow_up(runtime, state)
    if model_update is not None:
        runtime.trace.record(
            "sufficiency",
            route=model_update.get("route"),
            plan_index=index,
            last_status=last_status,
            last_matches=last_matches,
            budget_remaining=runtime.budget.remaining,
            model_follow_up=True,
        )
        return model_update

    if (
        index < len(plan)
        and index < max_steps
        and attempts < max_steps
        and runtime.budget.can_reserve()
        and last_status == ToolResultStatus.OK.value
        and last_matches > 0
        and bool(plan[index].get("requires_prior_matches"))
        and state.get("phase") != "evidence_registration_failed"
    ):
        route = "continue_search"
    elif state.get("phase") in {
        "tool_rejected",
        "budget_exhausted",
        "unresolved_attempt_blocked",
        "evidence_registration_failed",
    }:
        route = "terminate"
    else:
        route = "proceed_to_report"

    runtime.trace.record(
        "sufficiency",
        route=route,
        plan_index=index,
        last_status=last_status,
        last_matches=last_matches,
        budget_remaining=runtime.budget.remaining,
    )
    return {"phase": "sufficiency_assessed", "route": route}


def route_after_sufficiency(state: InvestigationState) -> RouteName:
    route = state.get("route") or "terminate"
    if route == "continue_search":
        return "continue_search"
    if route == "proceed_to_report":
        return "proceed_to_report"
    return "terminate"


def route_after_select(state: InvestigationState) -> Literal["execute_tool", "assess_sufficiency"]:
    if state.get("pending_tool_request"):
        return "execute_tool"
    return "assess_sufficiency"


def route_after_register(
    state: InvestigationState,
) -> Literal["evaluate_temporal_relations", "terminate"]:
    """Fail closed: domain registration errors must not continue toward a report."""
    if state.get("phase") == "evidence_registration_failed":
        return "terminate"
    return "evaluate_temporal_relations"


def draft_claims(runtime: InvestigationRuntime, state: InvestigationState) -> dict[str, Any]:
    _node(runtime, "draft_claims")
    claim_ids = draft_candidate_claims(runtime)
    return {
        "phase": "claims_drafted",
        "candidate_claim_ids": claim_ids,
    }


def validate_claims_node(
    runtime: InvestigationRuntime, state: InvestigationState
) -> dict[str, Any]:
    _node(runtime, "validate_claims")
    candidates = list(state.get("candidate_claim_ids") or [])
    validated = validate_claims(runtime, candidates)
    return {
        "phase": "claims_validated",
        "validated_claim_ids": validated,
    }


def generate_report(
    runtime: InvestigationRuntime, state: InvestigationState
) -> dict[str, Any]:
    _node(runtime, "generate_report")
    report = assemble_report(
        runtime,
        validated_claim_ids=list(state.get("validated_claim_ids") or []),
    )
    runtime.store.report = report
    return {
        "phase": "report_generated",
        "report_id": report.report_id,
    }


def output_gate_node(
    runtime: InvestigationRuntime, state: InvestigationState
) -> dict[str, Any]:
    _node(runtime, "output_gate")
    report = runtime.store.report
    if report is None:
        runtime.case = runtime.case.transition_to(CaseStatus.FAILED)
        runtime.store.case = runtime.case
        return {
            "phase": "output_gate_failed",
            "last_error": "missing report",
            "route": "terminate",
        }
    decision = run_output_gate(
        case=runtime.case,
        report=report,
        claims_by_id=runtime.store.claims,
        assessments=runtime.store.assessments,
        evidence_ids=frozenset(runtime.store.evidence),
        observation_ids=frozenset(runtime.store.observations),
        relation_ids=frozenset(runtime.store.relations),
        search_receipt_ids=frozenset(runtime.store.receipts),
        evidence_by_id=runtime.store.evidence,
        observations_by_id=runtime.store.observations,
        relations_by_id=runtime.store.relations,
        receipts_by_id=runtime.store.receipts,
        tool_results_by_receipt_id=runtime.store.tool_results_by_receipt,
        observation_intervals=runtime.store.observation_intervals,
    )
    runtime.last_gate = decision
    runtime.trace.record(
        "output_gate",
        accepted=decision.accepted,
        case_status=decision.case_status.value,
        reasons=list(decision.reasons),
    )
    if decision.accepted and decision.report is not None:
        runtime.store.report = decision.report
        runtime.case = runtime.case.transition_to(decision.case_status)
        runtime.store.case = runtime.case
        return {
            "phase": "completed",
            "report_id": decision.report.report_id,
            "last_error": None,
        }

    if runtime.case.status is CaseStatus.RUNNING:
        runtime.case = runtime.case.transition_to(CaseStatus.FAILED)
        runtime.store.case = runtime.case
    return {
        "phase": "output_gate_rejected",
        "last_error": "; ".join(decision.reasons) or "output gate rejected report",
        "route": "terminate",
    }


def terminate_node(
    runtime: InvestigationRuntime, state: InvestigationState
) -> dict[str, Any]:
    _node(runtime, "terminate")
    if runtime.case.status is CaseStatus.RUNNING:
        runtime.case = runtime.case.transition_to(CaseStatus.FAILED)
        runtime.store.case = runtime.case
    runtime.trace.record(
        "terminated",
        case_status=runtime.case.status.value,
        last_error=state.get("last_error"),
    )
    return {
        "phase": "terminated",
        "pending_tool_request": None,
    }
