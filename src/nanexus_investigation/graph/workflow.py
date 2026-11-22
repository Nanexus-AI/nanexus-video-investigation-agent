"""Compile and run the single LangGraph StateGraph investigation controller."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from nanexus_investigation.graph.nodes import (
    assess_sufficiency,
    build_plan,
    discover_capabilities,
    draft_claims,
    evaluate_temporal_relations,
    execute_tool,
    generate_report,
    normalize_register_evidence,
    output_gate_node,
    route_after_intake,
    route_after_register,
    route_after_select,
    route_after_sufficiency,
    select_validate_tool,
    terminate_node,
    validate_claims_node,
    validate_intake,
)
from nanexus_investigation.graph.runtime import (
    InvestigationRuntime,
    ScenarioConfig,
    finalize_domain_investigation,
    make_case,
)
from nanexus_investigation.graph.state import InvestigationState
from nanexus_investigation.persistence.errors import DomainFinalizationError


def build_investigation_graph(
    runtime: InvestigationRuntime,
    *,
    checkpointer=None,
):
    """Build one StateGraph. Thin Harness remains cross-cutting via ``runtime``.

    ``checkpointer`` defaults to ``InMemorySaver`` (existing E0 path). Callers
    may inject native ``SqliteSaver`` for the durable integration path. No custom
    Checkpointer is used. Node retry remains unset.
    """

    graph = StateGraph(InvestigationState)

    graph.add_node("validate_intake", lambda s: validate_intake(runtime, s))
    graph.add_node("discover_capabilities", lambda s: discover_capabilities(runtime, s))
    graph.add_node("build_plan", lambda s: build_plan(runtime, s))
    graph.add_node("select_validate_tool", lambda s: select_validate_tool(runtime, s))
    graph.add_node("execute_tool", lambda s: execute_tool(runtime, s))
    graph.add_node(
        "normalize_register_evidence",
        lambda s: normalize_register_evidence(runtime, s),
    )
    graph.add_node(
        "evaluate_temporal_relations",
        lambda s: evaluate_temporal_relations(runtime, s),
    )
    graph.add_node("assess_sufficiency", lambda s: assess_sufficiency(runtime, s))
    graph.add_node("draft_claims", lambda s: draft_claims(runtime, s))
    graph.add_node("validate_claims", lambda s: validate_claims_node(runtime, s))
    graph.add_node("generate_report", lambda s: generate_report(runtime, s))
    graph.add_node("output_gate", lambda s: output_gate_node(runtime, s))
    graph.add_node("terminate", lambda s: terminate_node(runtime, s))

    graph.add_edge(START, "validate_intake")
    graph.add_conditional_edges(
        "validate_intake",
        route_after_intake,
        {
            "discover_capabilities": "discover_capabilities",
            "terminate": "terminate",
        },
    )
    graph.add_edge("discover_capabilities", "build_plan")
    graph.add_edge("build_plan", "select_validate_tool")
    graph.add_conditional_edges(
        "select_validate_tool",
        route_after_select,
        {
            "execute_tool": "execute_tool",
            "assess_sufficiency": "assess_sufficiency",
        },
    )
    graph.add_edge("execute_tool", "normalize_register_evidence")
    graph.add_conditional_edges(
        "normalize_register_evidence",
        route_after_register,
        {
            "evaluate_temporal_relations": "evaluate_temporal_relations",
            "terminate": "terminate",
        },
    )
    graph.add_edge("evaluate_temporal_relations", "assess_sufficiency")
    graph.add_conditional_edges(
        "assess_sufficiency",
        route_after_sufficiency,
        {
            "continue_search": "select_validate_tool",
            "proceed_to_report": "draft_claims",
            "terminate": "terminate",
        },
    )
    graph.add_edge("draft_claims", "validate_claims")
    graph.add_edge("validate_claims", "generate_report")
    graph.add_edge("generate_report", "output_gate")
    graph.add_edge("output_gate", END)
    graph.add_edge("terminate", END)

    saver = checkpointer if checkpointer is not None else InMemorySaver()
    return graph.compile(checkpointer=saver)


def initial_state(runtime: InvestigationRuntime) -> InvestigationState:
    return {
        "schema_version": "e0-g4-v1-d",
        "case_id": runtime.case.case_id,
        "phase": "start",
        "effective_scope_ref": "",
        "time_context_ref": "",
        "plan": [],
        "plan_index": 0,
        "pending_tool_request": None,
        "tool_execution_ids": [],
        "observation_ids": [],
        "evidence_ids": [],
        "search_receipt_ids": [],
        "relation_ids": [],
        "candidate_claim_ids": [],
        "validated_claim_ids": [],
        "coverage_summary": [],
        "budget_usage": {
            "reserved": 0,
            "consumed": 0,
            "max": runtime.budget.max_attempts,
        },
        "attempt_summary": {"tool_attempts": 0, "plan_steps": 0},
        "last_error": None,
        "report_id": None,
        "max_plan_steps": runtime.scenario.max_plan_steps,
        "last_tool_status": None,
        "last_match_count": 0,
    }


def run_investigation(
    runtime: InvestigationRuntime,
    *,
    thread_id: str = "e0-thread-1",
    checkpointer=None,
    durability: str | None = None,
) -> dict[str, Any]:
    """Execute the compiled graph once for the given runtime/fixture.

    Pass native ``SqliteSaver`` and ``durability="sync"`` for the durable path.
    Default remains in-memory. Checkpoint and domain DB transactions are
    independent — this runner does not claim a shared atomic commit.

    For the durable path, domain Case admission must already be ``running``
    (see ``from_durable_fixture``). After Graph completion, domain terminal
    finalization is attempted independently. A successful Graph checkpoint
    alone is never reported as ``domain_finalized``.
    """
    if runtime.uses_durable_persistence:
        if runtime.domain_repo is None:
            raise DomainFinalizationError("durable path requires domain_repo")
        domain_case = runtime.domain_repo.get_case(runtime.case.case_id)
        if domain_case is None or domain_case.status.value != "running":
            raise DomainFinalizationError(
                "Graph invocation requires durable Case status=running "
                f"(found {None if domain_case is None else domain_case.status.value})"
            )

    app = build_investigation_graph(runtime, checkpointer=checkpointer)
    invoke_kwargs: dict[str, Any] = {
        "config": {"configurable": {"thread_id": thread_id}},
    }
    if durability is not None:
        invoke_kwargs["durability"] = durability
    final_state = app.invoke(initial_state(runtime), **invoke_kwargs)

    domain_finalized = False
    domain_report = None
    finalization_error = None
    if runtime.uses_durable_persistence:
        try:
            finalized = finalize_domain_investigation(runtime)
            domain_finalized = True
            domain_report = finalized.report
        except Exception as exc:
            # Domain remains consistent (rolled back). Graph candidate results
            # may exist but must not be presented as durable finalization.
            finalization_error = str(exc)
            runtime.trace.record(
                "domain_finalization_failed",
                case_id=runtime.case.case_id,
                error=finalization_error,
                domain_finalized=False,
            )

    return {
        "state": final_state,
        "case": runtime.case,
        "report": runtime.store.report,
        "trace": runtime.trace,
        "runtime": runtime,
        "domain_finalized": domain_finalized if runtime.uses_durable_persistence else None,
        "domain_report": domain_report,
        "finalization_error": finalization_error,
    }


__all__ = [
    "InvestigationRuntime",
    "ScenarioConfig",
    "build_investigation_graph",
    "initial_state",
    "make_case",
    "run_investigation",
]
