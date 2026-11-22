"""G4-E1-A1: local model evidence-dependent follow-up (real Ollama inference).

These tests require a reachable local Ollama endpoint with qwen3:14b.
They fail hard when the model is unavailable — they do not skip.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

import pytest

from nanexus_investigation.contracts.enums import CaseStatus, SearchCoverage
from nanexus_investigation.graph.runtime import (
    ScenarioConfig,
    empty_result,
    make_case,
    ok_event_result,
)
from nanexus_investigation.graph.workflow import InvestigationRuntime, run_investigation
from nanexus_investigation.model.provider import (
    InvestigationProposal,
    OllamaChatProvider,
    StaticModelProvider,
)

UTC = timezone.utc
T0 = datetime(2026, 11, 19, 20, 0, tzinfo=UTC)
T1 = datetime(2026, 11, 19, 23, 0, tzinfo=UTC)
OLLAMA_URL = "http://127.0.0.1:11434"
MODEL_NAME = "qwen3:14b"


def _require_local_model() -> OllamaChatProvider:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        pytest.fail(f"E1-A1 requires reachable local Ollama at {OLLAMA_URL}: {exc}")

    names = {m.get("name") for m in payload.get("models", []) if isinstance(m, dict)}
    if MODEL_NAME not in names:
        pytest.fail(
            f"E1-A1 requires installed model {MODEL_NAME!r}; installed={sorted(names)}"
        )
    return OllamaChatProvider(model=MODEL_NAME, base_url=OLLAMA_URL, timeout_s=180.0)


def _person_ok(execution_id: str = "exec-e1-person") -> object:
    return ok_event_result(
        execution_id=execution_id,
        native_id="evt-e1-person",
        label="person",
        start=T0 + timedelta(minutes=10),
        end=T0 + timedelta(minutes=12),
        camera_id="front_yard",
    )


def _vehicle_ok(execution_id: str = "exec-e1-vehicle") -> object:
    return ok_event_result(
        execution_id=execution_id,
        native_id="evt-e1-vehicle",
        label="vehicle",
        start=T0 + timedelta(minutes=15),
        end=T0 + timedelta(minutes=17),
        camera_id="front_yard",
    )


def test_e1_a1_01_real_inference_produces_structured_proposal():
    provider = _require_local_model()
    from nanexus_investigation.model.provider import ProposalContext

    proposal = provider.propose_follow_up(
        ProposalContext(
            question="Was a person near the driveway, then a related vehicle?",
            case_id="case-e1-smoke",
            authorized_tools=("search_events", "search_related_events"),
            authorized_cameras=("front_yard",),
            vms_instance_id="vms-local-1",
            time_bounds={"start_utc": T0.isoformat(), "end_utc": T1.isoformat()},
            prior_tool_name="search_events",
            prior_tool_status="ok",
            prior_match_count=1,
            prior_result_summary="1 registered observation(s); labels=['person']",
        )
    )
    assert proposal.action in {"invoke_tool", "proceed_to_report"}
    assert proposal.rationale
    if proposal.action == "invoke_tool":
        assert proposal.tool_name in {"search_events", "search_related_events"}


CONTROLLED_QUESTION = (
    "Investigate whether person activity in the specified camera and time window "
    "has related vehicle activity."
)


def _controlled_runtime(
    *,
    case_id: str,
    provider: OllamaChatProvider,
    search_events_result: object,
    related_script: object | None = None,
) -> InvestigationRuntime:
    """Same question/scope/tools/model; only the initial ToolResult differs."""
    case = make_case(
        case_id=case_id,
        question=CONTROLLED_QUESTION,
        cameras=("front_yard",),
        start=T0,
        end=T1,
    )
    scripts: dict = {"search_events": [search_events_result]}
    if related_script is not None:
        scripts["search_related_events"] = [related_script]
    scenario = ScenarioConfig(
        name=f"e1_controlled_{case_id}",
        include_follow_up=False,
        use_model_follow_up=True,
        max_tool_attempts=3,
        tool_scripts=scripts,
        # Preserve E1 related-events follow-up semantics (T3 temporal opt-in).
        enable_temporal_analysis=False,
        enable_media_resolve=False,
    )
    return InvestigationRuntime.from_fixture(
        case=case, scenario=scenario, model_provider=provider
    )


def test_e1_a1_f1_same_question_evidence_dependent_decisions():
    """F1: same question/scope/model; only ToolResult content changes."""
    provider = _require_local_model()

    # Scenario A — person observation makes related-event lookup relevant.
    runtime_a = _controlled_runtime(
        case_id="case-e1-f1-person",
        provider=provider,
        search_events_result=_person_ok("exec-e1-f1-person"),
        related_script=_vehicle_ok("exec-e1-f1-vehicle"),
    )
    result_a = run_investigation(runtime_a, thread_id="thread-e1-f1-person")
    proposal_a = runtime_a.last_model_proposal
    assert proposal_a is not None
    assert isinstance(proposal_a, InvestigationProposal)
    assert proposal_a.raw_text
    assert proposal_a.action == "invoke_tool", (
        f"Scenario A expected related follow-up under person evidence; "
        f"got action={proposal_a.action!r} tool={proposal_a.tool_name!r} "
        f"rationale={proposal_a.rationale!r}"
    )
    # Accept either legacy related-events or a second search_events(label=car).
    assert proposal_a.tool_name in {"search_related_events", "search_events"}, (
        f"Scenario A unexpected tool under person evidence: {proposal_a.to_dict()}"
    )
    if proposal_a.tool_name == "search_events":
        assert proposal_a.label in {"car", "vehicle", None}
    assert runtime_a.registry.invocations[0] == "search_events"
    assert len(runtime_a.registry.invocations) >= 2
    assert runtime_a.registry.invocations[1] == proposal_a.tool_name
    assert runtime_a.last_guard is not None and runtime_a.last_guard.accepted
    assert runtime_a.store.observations  # person (+ related) evidence registered
    assert runtime_a.store.report is not None
    assert runtime_a.last_gate is not None and runtime_a.last_gate.accepted
    assert runtime_a.case.status in {CaseStatus.COMPLETED, CaseStatus.PARTIAL}
    assert result_a["state"].get("model_proposal", {}).get("tool_name") == (
        proposal_a.tool_name
    )

    # Scenario B — identical question/scope; EMPTY exhausted, no person observation.
    runtime_b = _controlled_runtime(
        case_id="case-e1-f1-empty",
        provider=provider,
        search_events_result=empty_result(
            execution_id="exec-e1-f1-empty", coverage=SearchCoverage.EXHAUSTED
        ),
        related_script=_vehicle_ok("exec-e1-f1-should-not-run"),
    )
    result_b = run_investigation(runtime_b, thread_id="thread-e1-f1-empty")
    proposal_b = runtime_b.last_model_proposal
    assert proposal_b is not None
    assert proposal_b.raw_text
    assert proposal_b.action == "proceed_to_report", (
        f"Scenario B expected proceed_to_report under EMPTY evidence; "
        f"got action={proposal_b.action!r} tool={proposal_b.tool_name!r} "
        f"rationale={proposal_b.rationale!r}"
    )
    assert runtime_b.registry.invocations == ["search_events"]
    assert "search_related_events" not in runtime_b.registry.invocations
    assert not runtime_b.store.observations  # EMPTY must not invent observations
    assert runtime_b.store.report is not None
    assert runtime_b.last_gate is not None and runtime_b.last_gate.accepted
    limitations = " ".join(runtime_b.store.report.limitations).lower()
    assert "empty" in limitations
    assert "absence" in limitations
    assert "never happened" not in limitations
    assert result_b["state"].get("model_proposal", {}).get("action") == (
        "proceed_to_report"
    )

    # Evidence-dependent contrast (not question-dependent).
    assert runtime_a.case.question == runtime_b.case.question == CONTROLLED_QUESTION
    assert (
        runtime_a.case.effective_scope.camera_ids
        == runtime_b.case.effective_scope.camera_ids
    )
    assert proposal_a.action != proposal_b.action


def test_e1_a1_03_guard_accepts_model_tool_before_execution():
    """Valid model tool proposals still pass through Tool Guard (covered in 02)."""
    # Reinforced with an explicit static valid proposal path for isolation.
    provider = StaticModelProvider(
        proposal=InvestigationProposal(
            action="invoke_tool",
            tool_name="search_related_events",
            label="vehicle",
            rationale="static valid follow-up for guard path",
        )
    )
    case = make_case(case_id="case-e1-guard-ok", start=T0, end=T1)
    scenario = ScenarioConfig(
        name="e1_guard_ok",
        use_model_follow_up=True,
        enable_temporal_analysis=False,
        enable_media_resolve=False,
        tool_scripts={
            "search_events": [_person_ok("exec-guard-person")],
            "search_related_events": [_vehicle_ok("exec-guard-vehicle")],
        },
    )
    runtime = InvestigationRuntime.from_fixture(
        case=case, scenario=scenario, model_provider=provider
    )
    run_investigation(runtime, thread_id="thread-e1-guard-ok")
    assert runtime.registry.invocations == ["search_events", "search_related_events"]
    assert runtime.last_guard is not None
    assert runtime.last_guard.accepted
    assert runtime.last_guard.request is not None
    assert runtime.last_guard.request.tool_name == "search_related_events"


def test_e1_a1_04_invalid_proposals_rejected_without_unauthorized_tools():
    # Unknown tool from model/static provider.
    bad_tool = StaticModelProvider(
        proposal=InvestigationProposal(
            action="invoke_tool",
            tool_name="shell_exec",
            label=None,
            rationale="attempt unauthorized tool",
        )
    )
    case = make_case(case_id="case-e1-bad-tool", start=T0, end=T1)
    runtime = InvestigationRuntime.from_fixture(
        case=case,
        scenario=ScenarioConfig(
            name="e1_bad_tool",
            use_model_follow_up=True,
            enable_temporal_analysis=False,
            enable_media_resolve=False,
            tool_scripts={"search_events": [_person_ok("exec-bad-tool")]},
        ),
        model_provider=bad_tool,
    )
    result = run_investigation(runtime, thread_id="thread-e1-bad-tool")
    assert runtime.registry.invocations == ["search_events"]
    assert "shell_exec" not in runtime.registry.invocations
    assert runtime.last_guard is not None and runtime.last_guard.rejected
    assert result["state"].get("phase") in {"terminated", "sufficiency_assessed", "tool_rejected"}
    assert runtime.store.report is None or runtime.last_gate is None or not (
        runtime.last_gate and runtime.last_gate.accepted and "shell_exec" in str(runtime.trace.events)
    )

    # Unauthorized camera in proposal.
    bad_cam = StaticModelProvider(
        proposal=InvestigationProposal(
            action="invoke_tool",
            tool_name="search_related_events",
            label="vehicle",
            camera_ids=("back_alley",),
            rationale="attempt unauthorized camera",
        )
    )
    runtime2 = InvestigationRuntime.from_fixture(
        case=make_case(case_id="case-e1-bad-cam", start=T0, end=T1),
        scenario=ScenarioConfig(
            name="e1_bad_cam",
            use_model_follow_up=True,
            enable_temporal_analysis=False,
            enable_media_resolve=False,
            tool_scripts={
                "search_events": [_person_ok("exec-bad-cam")],
                "search_related_events": [_vehicle_ok("exec-bad-cam-v")],
            },
        ),
        model_provider=bad_cam,
    )
    run_investigation(runtime2, thread_id="thread-e1-bad-cam")
    assert runtime2.registry.invocations == ["search_events"]
    assert runtime2.last_guard is not None and runtime2.last_guard.rejected

    # Malformed follow-up JSON: with registered observations, prefer a partial
    # evidence-grounded report over hard termination (no further tools run).
    malformed = StaticModelProvider(raw_text="not-a-proposal")
    runtime3 = InvestigationRuntime.from_fixture(
        case=make_case(case_id="case-e1-malformed", start=T0, end=T1),
        scenario=ScenarioConfig(
            name="e1_malformed",
            use_model_follow_up=True,
            enable_temporal_analysis=False,
            enable_media_resolve=False,
            tool_scripts={"search_events": [_person_ok("exec-malformed")]},
        ),
        model_provider=malformed,
    )
    out = run_investigation(runtime3, thread_id="thread-e1-malformed")
    assert runtime3.registry.invocations == ["search_events"]
    # Partial is expected: evidence was registered, but replanning stopped.
    assert runtime3.case.status.value in {"completed", "partial"}
    assert runtime3.last_gate is not None and runtime3.last_gate.accepted
    assert out["report"] is not None
    assert any(
        "invalid model follow-up proposal" in item
        for item in runtime3.planning_limitations
    )


def test_e1_a1_05_deterministic_e0_path_unchanged_by_default():
    case = make_case(case_id="case-e1-e0-default", start=T0, end=T1)
    scenario = ScenarioConfig(
        name="e1_default_deterministic",
        include_follow_up=False,
        use_model_follow_up=False,
        tool_scripts={"search_events": [_person_ok("exec-e0-default")]},
    )
    runtime = InvestigationRuntime.from_fixture(case=case, scenario=scenario)
    run_investigation(runtime, thread_id="thread-e1-e0-default")
    assert runtime.model_follow_up_applied is False
    assert runtime.last_model_proposal is None
    assert runtime.registry.invocations == ["search_events"]
    assert not any(e["kind"] == "model_proposal" for e in runtime.trace.events)
    assert runtime.store.report is not None
