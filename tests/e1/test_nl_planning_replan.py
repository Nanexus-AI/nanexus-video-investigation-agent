"""V0.1 T2: NL initial planning and budgeted evidence-dependent replanning.

Deterministic StaticModelProvider tests always run.
Genuine Ollama × recorded Frigate tests require reachable Ollama and fail hard
when unavailable — they never skip as PASS.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests.e1.live_scope import require_live_cameras, require_live_time_window

from nanexus_investigation.contracts.enums import (
    CaseStatus,
    RecordType,
    SearchCoverage,
    ToolResultStatus,
)
from nanexus_investigation.contracts.identity import Interval
from nanexus_investigation.contracts.tool import ToolResult
from nanexus_investigation.graph.request_fingerprint import tool_request_fingerprint
from nanexus_investigation.graph.runtime import ScenarioConfig, make_case
from nanexus_investigation.graph.workflow import InvestigationRuntime, run_investigation
from nanexus_investigation.model.provider import (
    InvestigationProposal,
    OllamaChatProvider,
    ProposalContext,
    StaticModelProvider,
    build_initial_plan_prompt,
)
from nanexus_investigation.tools.frigate_backend import FrigateToolRegistry

FIXTURES = Path(__file__).parent / "fixtures" / "frigate"
PRIVATE_RECORDING = Path(".local/frigate_recordings/events_list_live.json")
UTC = timezone.utc
VMS = "frigate-fixture-0.17.1"
OLLAMA_URL = "http://127.0.0.1:11434"
MODEL_NAME = "qwen3:14b"

PERSON_QUESTION = "Find person detections in this camera and time window."
VEHICLE_QUESTION = "Find vehicle detections in this camera and time window."
FOLLOW_UP_QUESTION = (
    "Investigate person activity and check whether vehicle detections "
    "occurred in the same authorized period. "
    "Do not invent physical interaction from timestamps alone."
)


def _require_local_model() -> OllamaChatProvider:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        pytest.fail(f"T2 requires reachable local Ollama at {OLLAMA_URL}: {exc}")
    names = {m.get("name") for m in payload.get("models", []) if isinstance(m, dict)}
    if MODEL_NAME not in names:
        pytest.fail(
            f"T2 requires installed model {MODEL_NAME!r}; installed={sorted(names)}"
        )
    return OllamaChatProvider(model=MODEL_NAME, base_url=OLLAMA_URL, timeout_s=180.0)


def _fixture_case(*, case_id: str, question: str, vms: str = "frigate-fixture"):
    return make_case(
        case_id=case_id,
        question=question,
        cameras=("front_yard",),
        start=datetime(2023, 11, 14, 22, 0, tzinfo=UTC),
        end=datetime(2023, 11, 14, 23, 30, tzinfo=UTC),
        vms_instance_id=vms,
    )


def _nl_scenario(**overrides) -> ScenarioConfig:
    base = dict(
        name="t2_nl",
        tool_backend="frigate_recorded",
        frigate_recording_path=str(FIXTURES / "events_list_sanitized.json"),
        use_model_initial_plan=True,
        use_model_follow_up=True,
        max_model_calls=4,
        max_tool_attempts=4,
        max_plan_steps=4,
        include_follow_up=False,
        # Deterministic search_label must not drive the model path.
        search_label="person",
    )
    base.update(overrides)
    return ScenarioConfig(**base)


def test_t2_fingerprint_equivalence_ignores_request_id():
    a = tool_request_fingerprint(
        tool_name="search_events",
        source_id="vms-1",
        camera_ids=("front_yard",),
        label="person",
        time_bounds={"start_utc": "2023-11-14T22:00:00+00:00", "end_utc": "2023-11-14T23:30:00+00:00"},
    )
    b = tool_request_fingerprint(
        tool_name="search_events",
        source_id="vms-1",
        camera_ids=["front_yard"],
        label="person",
        time_bounds=Interval(
            start_utc=datetime(2023, 11, 14, 22, 0, tzinfo=UTC),
            end_utc=datetime(2023, 11, 14, 23, 30, tzinfo=UTC),
        ),
    )
    assert a == b
    c = tool_request_fingerprint(
        tool_name="search_events",
        source_id="vms-1",
        camera_ids=("front_yard",),
        label="car",
        time_bounds={"start_utc": "2023-11-14T22:00:00+00:00", "end_utc": "2023-11-14T23:30:00+00:00"},
    )
    assert a != c


def test_t6_temporal_fingerprint_ignores_decorative_label():
    args = {
        "predicate": "near",
        "window_seconds": 180.0,
        "anchor_label": "person",
        "candidate_label": "car",
    }
    a = tool_request_fingerprint(
        tool_name="analyze_temporal_relations",
        source_id="vms-1",
        camera_ids=("front_yard",),
        label=None,
        time_bounds=None,
        extra_arguments=args,
    )
    b = tool_request_fingerprint(
        tool_name="analyze_temporal_relations",
        source_id="vms-1",
        camera_ids=("front_yard",),
        label="person|car",
        time_bounds=None,
        extra_arguments=args,
    )
    assert a == b


def _multi_label_ok_result(*, execution_id: str, t0: datetime) -> ToolResult:
    from datetime import timedelta

    return ToolResult(
        status=ToolResultStatus.OK,
        execution_id=execution_id,
        data=[
            {
                "native_record_id": "evt-t7c-ml-person",
                "record_type": RecordType.EVENT,
                "camera_id": "front_yard",
                "label": "person",
                "start_utc": t0,
                "end_utc": t0 + timedelta(seconds=20),
                "source_revision": "rev-1",
            },
            {
                "native_record_id": "evt-t7c-ml-car",
                "record_type": RecordType.EVENT,
                "camera_id": "front_yard",
                "label": "car",
                "start_utc": t0 + timedelta(seconds=40),
                "end_utc": t0 + timedelta(seconds=60),
                "source_revision": "rev-1",
            },
        ],
        source_id="vms-local-1",
        retrieved_at=datetime(2026, 11, 20, 12, 5, tzinfo=UTC),
        coverage=SearchCoverage.EXHAUSTED,
        provenance="stub:search/multi-label",
        usage={"tool_attempts": 1},
    )


def test_t6f_multi_label_question_stays_partial_when_one_label_searched():
    """Person+vehicle question with only car search must not complete as fully answered."""
    from datetime import timedelta

    from nanexus_investigation.graph.runtime import ok_event_result

    t0 = datetime(2023, 11, 14, 22, 10, tzinfo=UTC)
    provider = StaticModelProvider(
        initial_proposal=InvestigationProposal(
            action="invoke_tool",
            tool_name="search_events",
            label="car",
            rationale="search vehicles only",
            camera_ids=("front_yard",),
        ),
        proposals=[
            InvestigationProposal(
                action="proceed_to_report",
                tool_name=None,
                label=None,
                rationale="stop after car search",
            )
        ],
    )
    runtime = InvestigationRuntime.from_fixture(
        case=make_case(
            case_id="case-t6f-multi-obj",
            question="Find person or vehicle detections in this camera and time window.",
            start=datetime(2023, 11, 14, 22, 0, tzinfo=UTC),
            end=datetime(2023, 11, 14, 23, 30, tzinfo=UTC),
        ),
        scenario=ScenarioConfig(
            name="t6f_multi_obj",
            use_model_initial_plan=True,
            use_model_follow_up=True,
            enable_temporal_analysis=False,
            enable_media_resolve=False,
            max_model_calls=3,
            max_tool_attempts=3,
            tool_scripts={
                "search_events": [
                    ok_event_result(
                        execution_id="exec-t6f-multi",
                        native_id="evt-t6f-multi",
                        label="car",
                        start=t0,
                        end=t0 + timedelta(seconds=30),
                    )
                ]
            },
        ),
        model_provider=provider,
    )
    result = run_investigation(runtime, thread_id="thread-t6f-multi-obj")
    report = result["report"]
    assert report is not None
    assert runtime.last_gate is not None and runtime.last_gate.accepted
    assert runtime.case.status is CaseStatus.PARTIAL
    assert report.execution_status is CaseStatus.PARTIAL
    assert any("partial investigation coverage" in item for item in report.limitations)
    assert any("person" in item.lower() for item in report.unresolved_questions)


def test_t7c_multi_label_person_car_filter_covers_both_objectives():
    """Comma multi-label search_events(person,car) covers person+vehicle questions."""
    from datetime import timedelta

    from nanexus_investigation.graph.runtime import ok_event_result

    t0 = datetime(2023, 11, 14, 22, 10, tzinfo=UTC)
    provider = StaticModelProvider(
        initial_proposal=InvestigationProposal(
            action="invoke_tool",
            tool_name="search_events",
            label="person,car",
            rationale="search person and car together",
            camera_ids=("front_yard",),
        ),
        proposals=[
            InvestigationProposal(
                action="proceed_to_report",
                tool_name=None,
                label=None,
                rationale="both families covered",
            )
        ],
    )
    runtime = InvestigationRuntime.from_fixture(
        case=make_case(
            case_id="case-t7c-multi-label",
            question=(
                "Find all person and vehicle detections and identify nearby activity."
            ),
            start=datetime(2023, 11, 14, 22, 0, tzinfo=UTC),
            end=datetime(2023, 11, 14, 23, 30, tzinfo=UTC),
        ),
        scenario=ScenarioConfig(
            name="t7c_multi_label",
            use_model_initial_plan=True,
            use_model_follow_up=True,
            enable_temporal_analysis=False,
            enable_media_resolve=False,
            max_model_calls=3,
            max_tool_attempts=3,
            tool_scripts={
                "search_events": [
                    _multi_label_ok_result(
                        execution_id="exec-t7c-ml",
                        t0=t0,
                    )
                ]
            },
        ),
        model_provider=provider,
    )
    result = run_investigation(runtime, thread_id="thread-t7c-multi-label")
    report = result["report"]
    assert report is not None
    assert runtime.case.status is CaseStatus.COMPLETED
    assert report.execution_status is CaseStatus.COMPLETED
    assert not any(
        "partial investigation coverage" in item for item in report.limitations
    )
    assert not report.unresolved_questions
    # Receipt retained the multi-label filter form.
    search_receipts = [
        r for r in runtime.store.receipts.values() if r.tool_name == "search_events"
    ]
    assert search_receipts
    assert search_receipts[0].filters.get("label") == "person,car"


def test_t7c_unfiltered_search_covers_person_and_vehicle_objectives():
    """Unfiltered search_events covers both families without label-specific calls."""
    from datetime import timedelta

    from nanexus_investigation.graph.runtime import ok_event_result

    t0 = datetime(2023, 11, 14, 22, 10, tzinfo=UTC)
    provider = StaticModelProvider(
        initial_proposal=InvestigationProposal(
            action="invoke_tool",
            tool_name="search_events",
            label=None,
            rationale="unfiltered search",
            camera_ids=("front_yard",),
        ),
        proposals=[
            InvestigationProposal(
                action="proceed_to_report",
                tool_name=None,
                label=None,
                rationale="scope complete",
            )
        ],
    )
    runtime = InvestigationRuntime.from_fixture(
        case=make_case(
            case_id="case-t7c-unfiltered",
            question="Find person and vehicle detections in this window.",
            start=datetime(2023, 11, 14, 22, 0, tzinfo=UTC),
            end=datetime(2023, 11, 14, 23, 30, tzinfo=UTC),
        ),
        scenario=ScenarioConfig(
            name="t7c_unfiltered",
            use_model_initial_plan=True,
            use_model_follow_up=True,
            enable_temporal_analysis=False,
            enable_media_resolve=False,
            max_model_calls=3,
            max_tool_attempts=3,
            tool_scripts={
                "search_events": [
                    ok_event_result(
                        execution_id="exec-t7c-uf",
                        native_id="evt-t7c-uf",
                        label="person",
                        start=t0,
                        end=t0 + timedelta(seconds=20),
                    )
                ]
            },
        ),
        model_provider=provider,
    )
    result = run_investigation(runtime, thread_id="thread-t7c-unfiltered")
    report = result["report"]
    assert report is not None
    assert runtime.case.status is CaseStatus.COMPLETED
    assert not any(
        "partial investigation coverage" in item for item in report.limitations
    )


def test_t2_initial_prompt_maps_vehicle_to_car_and_lists_tools():
    ctx = ProposalContext(
        question=VEHICLE_QUESTION,
        case_id="case-prompt",
        authorized_tools=("search_events",),
        authorized_cameras=("front_yard",),
        vms_instance_id=VMS,
        time_bounds={"start_utc": "t0", "end_utc": "t1"},
        prior_tool_name=None,
        prior_tool_status=None,
        prior_match_count=0,
        prior_result_summary="(none)",
        tool_descriptions=(
            "search_events: bounded read-only detection search by optional label.",
        ),
    )
    system, user = build_initial_plan_prompt(ctx)
    assert "initial planner" in system
    assert "label=car" in user
    assert "search_events" in user
    assert "analyze_temporal_relations" in user or "visual/media tools" in user


def test_t2_01_static_person_initial_plan_no_extra_tools():
    """T2-01: model selects search_events(label=person); no temporal/visual tools."""
    case = _fixture_case(case_id="case-t2-01-static", question=PERSON_QUESTION)
    provider = StaticModelProvider(
        initial_proposal=InvestigationProposal(
            action="invoke_tool",
            tool_name="search_events",
            label="person",
            rationale="User asked for person detections in the authorized window.",
            camera_ids=("front_yard",),
        ),
        proposals=[
            InvestigationProposal(
                action="proceed_to_report",
                tool_name=None,
                label=None,
                rationale="Person search completed; no further retrieval needed.",
            )
        ],
    )
    runtime = InvestigationRuntime.from_fixture(
        case=case, scenario=_nl_scenario(name="t2_01_static"), model_provider=provider
    )
    run_investigation(runtime, thread_id="thread-t2-01-static")

    assert provider.initial_calls, "initial planner must be consulted"
    assert runtime.registry.invocations == ["search_events"]
    assert "search_related_events" not in runtime.registry.invocations
    assert runtime.plan_steps[0].label == "person"
    assert runtime.last_gate is not None and runtime.last_gate.accepted
    assert runtime.case.status in {CaseStatus.COMPLETED, CaseStatus.PARTIAL}
    assert len(runtime.store.receipts) == 1
    assert runtime.model_calls >= 1


def test_t2_02_static_vehicle_maps_to_car_label():
    """T2-02: vehicle question → search_events(label=car) through Guard."""
    case = _fixture_case(case_id="case-t2-02-static", question=VEHICLE_QUESTION)
    provider = StaticModelProvider(
        initial_proposal=InvestigationProposal(
            action="invoke_tool",
            tool_name="search_events",
            label="car",
            rationale="Map vehicle request to Frigate label=car.",
            camera_ids=("front_yard",),
        ),
        proposals=[
            InvestigationProposal(
                action="proceed_to_report",
                tool_name=None,
                label=None,
                rationale="Vehicle/car search complete.",
            )
        ],
    )
    # search_label remains person to prove deterministic label is not used.
    runtime = InvestigationRuntime.from_fixture(
        case=case,
        scenario=_nl_scenario(name="t2_02_static", search_label="person"),
        model_provider=provider,
    )
    run_investigation(runtime, thread_id="thread-t2-02-static")

    assert runtime.plan_steps[0].label == "car"
    assert runtime.registry.invocations == ["search_events"]
    assert runtime.last_guard is not None and runtime.last_guard.accepted
    labels = {
        label
        for obs in runtime.store.observations.values()
        for label in obs.labels
    }
    assert "car" in labels
    assert runtime.last_gate is not None and runtime.last_gate.accepted


def test_t2_03_static_evidence_dependent_person_then_car():
    """T2-03: person evidence motivates a subsequent car search; then stop."""
    case = _fixture_case(case_id="case-t2-03-static", question=FOLLOW_UP_QUESTION)
    provider = StaticModelProvider(
        initial_proposal=InvestigationProposal(
            action="invoke_tool",
            tool_name="search_events",
            label="person",
            rationale="Start with person detections from the question.",
            camera_ids=("front_yard",),
        ),
        proposals=[
            InvestigationProposal(
                action="invoke_tool",
                tool_name="search_events",
                label="car",
                rationale="Person evidence found; check vehicle/car detections next.",
                camera_ids=("front_yard",),
            ),
            InvestigationProposal(
                action="proceed_to_report",
                tool_name=None,
                label=None,
                rationale="Both searches complete; no fabricated interaction.",
            ),
        ],
    )
    runtime = InvestigationRuntime.from_fixture(
        case=case, scenario=_nl_scenario(name="t2_03_static"), model_provider=provider
    )
    run_investigation(runtime, thread_id="thread-t2-03-static")

    assert runtime.registry.invocations == ["search_events", "search_events"]
    assert len(runtime.store.receipts) == 2
    assert len(provider.calls) >= 1
    assert any(p.label == "car" for p in runtime.model_proposals if p.action == "invoke_tool")
    labels = sorted(
        {
            label
            for obs in runtime.store.observations.values()
            for label in obs.labels
        }
    )
    assert "person" in labels
    assert "car" in labels
    # No temporal/visual tools and no invented related-events.
    assert "search_related_events" not in runtime.registry.invocations
    report = runtime.store.report
    assert report is not None
    joined = " ".join(report.limitations).lower() + " ".join(
        c.statement.lower() for c in runtime.store.claims.values()
    )
    assert "entered" not in joined
    assert "interaction" not in joined or "not" in joined
    assert runtime.last_gate is not None and runtime.last_gate.accepted


def test_t2_anti_repeat_blocks_identical_model_proposal():
    case = _fixture_case(case_id="case-t2-anti-repeat", question=PERSON_QUESTION)
    # Initial person search, then model keeps proposing the same person search.
    provider = StaticModelProvider(
        initial_proposal=InvestigationProposal(
            action="invoke_tool",
            tool_name="search_events",
            label="person",
            rationale="person search",
            camera_ids=("front_yard",),
        ),
        proposal=InvestigationProposal(
            action="invoke_tool",
            tool_name="search_events",
            label="person",
            rationale="repeat identical search",
            camera_ids=("front_yard",),
        ),
    )
    runtime = InvestigationRuntime.from_fixture(
        case=case,
        scenario=_nl_scenario(name="t2_anti_repeat", max_model_calls=3),
        model_provider=provider,
    )
    run_investigation(runtime, thread_id="thread-t2-anti-repeat")

    assert runtime.registry.invocations == ["search_events"]
    assert len(runtime.store.receipts) == 1
    assert runtime.budget.consumed == 1
    assert len(runtime.executed_request_fingerprints) == 1
    assert any(e.get("kind") == "anti_repeat_blocked" for e in runtime.trace.events)
    assert runtime.last_gate is not None and runtime.last_gate.accepted
    # Gate acceptance is not treated as full objective completion.
    assert runtime.case.status is CaseStatus.PARTIAL
    assert any("identical tool request" in item for item in runtime.store.report.limitations)


def test_t2_model_call_budget_accounting_and_legacy_oneshot():
    """shared model-call budget for initial + follow-up; legacy one-shot."""
    # Combined mode: initial + one follow-up within max_model_calls=2.
    case = _fixture_case(case_id="case-t2-budget-combined", question=FOLLOW_UP_QUESTION)
    provider = StaticModelProvider(
        initial_proposal=InvestigationProposal(
            action="invoke_tool",
            tool_name="search_events",
            label="person",
            rationale="initial person",
            camera_ids=("front_yard",),
        ),
        proposals=[
            InvestigationProposal(
                action="invoke_tool",
                tool_name="search_events",
                label="car",
                rationale="then car",
                camera_ids=("front_yard",),
            ),
            InvestigationProposal(
                action="proceed_to_report",
                tool_name=None,
                label=None,
                rationale="should not be reached when max_model_calls=2",
            ),
        ],
    )
    runtime = InvestigationRuntime.from_fixture(
        case=case,
        scenario=_nl_scenario(name="t2_budget_combined", max_model_calls=2),
        model_provider=provider,
    )
    run_investigation(runtime, thread_id="thread-t2-budget-combined")
    assert runtime.model_calls == 2
    assert len(provider.initial_calls) == 1
    assert len(provider.calls) == 1
    assert runtime.registry.invocations == ["search_events", "search_events"]
    assert any(
        e.get("kind") == "model_follow_up_budget_exhausted" for e in runtime.trace.events
    )
    assert runtime.case.status is CaseStatus.PARTIAL
    assert any("model-call budget exhausted" in item for item in runtime.store.report.limitations)

    # Legacy E1-style: deterministic initial plan + exactly one follow-up consultation.
    legacy_case = _fixture_case(case_id="case-t2-budget-legacy", question=PERSON_QUESTION)
    legacy_provider = StaticModelProvider(
        proposal=InvestigationProposal(
            action="proceed_to_report",
            tool_name=None,
            label=None,
            rationale="one-shot stop after first tool result",
        )
    )
    legacy = InvestigationRuntime.from_fixture(
        case=legacy_case,
        scenario=ScenarioConfig(
            name="t2_budget_legacy",
            tool_backend="frigate_recorded",
            frigate_recording_path=str(FIXTURES / "events_list_sanitized.json"),
            search_label="person",
            use_model_initial_plan=False,
            use_model_follow_up=True,
            max_model_calls=1,
            max_tool_attempts=3,
            include_follow_up=False,
        ),
        model_provider=legacy_provider,
    )
    run_investigation(legacy, thread_id="thread-t2-budget-legacy")
    assert legacy.model_calls == 1
    assert legacy_provider.initial_calls == []
    assert len(legacy_provider.calls) == 1
    assert legacy.registry.invocations == ["search_events"]
    assert legacy.plan_steps[0].label == "person"


def test_t2_empty_initial_then_proceed():
    case = _fixture_case(case_id="case-t2-empty", question=PERSON_QUESTION)
    provider = StaticModelProvider(
        initial_proposal=InvestigationProposal(
            action="invoke_tool",
            tool_name="search_events",
            label="person",
            rationale="search persons",
            camera_ids=("front_yard",),
        ),
        proposals=[
            InvestigationProposal(
                action="proceed_to_report",
                tool_name=None,
                label=None,
                rationale="EMPTY result; stop without fabricating detections.",
            )
        ],
    )
    runtime = InvestigationRuntime.from_fixture(
        case=case,
        scenario=_nl_scenario(
            name="t2_empty",
            frigate_recording_path=str(FIXTURES / "events_empty.json"),
        ),
        model_provider=provider,
    )
    run_investigation(runtime, thread_id="thread-t2-empty")
    assert runtime.registry.invocations == ["search_events"]
    assert len(runtime.store.observations) == 0
    assert runtime.last_model_proposal is not None
    assert runtime.last_model_proposal.action == "proceed_to_report"
    assert runtime.last_gate is not None


def test_t2_e0_default_still_deterministic_without_model_flags():
    """E0 path: no model flags → deterministic planner; model never called."""
    case = _fixture_case(case_id="case-t2-e0-default", question=PERSON_QUESTION)
    provider = StaticModelProvider(
        initial_proposal=InvestigationProposal(
            action="invoke_tool",
            tool_name="search_events",
            label="car",
            rationale="should not be consulted",
        )
    )
    scenario = ScenarioConfig(
        name="t2_e0_default",
        tool_backend="frigate_recorded",
        frigate_recording_path=str(FIXTURES / "events_list_sanitized.json"),
        search_label="person",
        use_model_initial_plan=False,
        use_model_follow_up=False,
    )
    runtime = InvestigationRuntime.from_fixture(
        case=case, scenario=scenario, model_provider=provider
    )
    run_investigation(runtime, thread_id="thread-t2-e0-default")
    assert provider.initial_calls == []
    assert provider.calls == []
    assert runtime.model_calls == 0
    assert runtime.plan_steps[0].label == "person"


def test_t2_real_ollama_contrasting_initial_plans():
    """Different NL questions → different initial labels from real qwen3:14b."""
    provider = _require_local_model()

    person_case = _fixture_case(case_id="case-t2-ollama-person", question=PERSON_QUESTION)
    vehicle_case = _fixture_case(
        case_id="case-t2-ollama-vehicle", question=VEHICLE_QUESTION
    )

    person_rt = InvestigationRuntime.from_fixture(
        case=person_case,
        scenario=_nl_scenario(
            name="t2_ollama_person",
            use_model_follow_up=True,
            max_model_calls=2,
        ),
        model_provider=provider,
    )
    vehicle_rt = InvestigationRuntime.from_fixture(
        case=vehicle_case,
        scenario=_nl_scenario(
            name="t2_ollama_vehicle",
            use_model_follow_up=True,
            max_model_calls=2,
        ),
        model_provider=provider,
    )
    run_investigation(person_rt, thread_id="thread-t2-ollama-person")
    run_investigation(vehicle_rt, thread_id="thread-t2-ollama-vehicle")

    p0 = person_rt.model_proposals[0]
    v0 = vehicle_rt.model_proposals[0]
    print(
        json.dumps(
            {
                "person_initial": p0.to_dict(),
                "vehicle_initial": v0.to_dict(),
                "person_invocations": list(person_rt.registry.invocations),
                "vehicle_invocations": list(vehicle_rt.registry.invocations),
            }
        )
    )
    assert p0.action == "invoke_tool"
    assert p0.tool_name == "search_events"
    assert p0.label == "person"
    assert v0.action == "invoke_tool"
    assert v0.tool_name == "search_events"
    assert v0.label == "car"
    assert person_rt.last_guard is not None and person_rt.last_guard.accepted
    assert vehicle_rt.last_guard is not None and vehicle_rt.last_guard.accepted
    assert person_rt.last_gate is not None and person_rt.last_gate.accepted
    assert vehicle_rt.last_gate is not None and vehicle_rt.last_gate.accepted


def test_t2_real_ollama_evidence_dependent_replan_recorded():
    """Real model: person hits can motivate car follow-up; record proposals."""
    provider = _require_local_model()
    recording = (
        PRIVATE_RECORDING
        if PRIVATE_RECORDING.is_file()
        else FIXTURES / "events_list_sanitized.json"
    )
    if recording == PRIVATE_RECORDING:
        case = make_case(
            case_id="case-t2-ollama-replan",
            question=FOLLOW_UP_QUESTION,
            cameras=require_live_cameras()[:1],
            start=require_live_time_window()[0].astimezone(UTC),
            end=require_live_time_window()[1].astimezone(UTC),
            vms_instance_id=VMS,
        )
        force_bounds = Interval(
            start_utc=require_live_time_window()[0].astimezone(UTC),
            end_utc=require_live_time_window()[1].astimezone(UTC),
        )
    else:
        case = _fixture_case(case_id="case-t2-ollama-replan", question=FOLLOW_UP_QUESTION)
        force_bounds = None

    scenario = _nl_scenario(
        name="t2_ollama_replan",
        frigate_recording_path=str(recording),
        force_time_bounds=force_bounds,
        max_model_calls=4,
        max_tool_attempts=4,
    )
    runtime = InvestigationRuntime.from_fixture(
        case=case, scenario=scenario, model_provider=provider
    )
    run_investigation(runtime, thread_id="thread-t2-ollama-replan")

    assert isinstance(runtime.registry, FrigateToolRegistry)
    assert runtime.registry.invocations[0] == "search_events"
    assert "search_related_events" not in runtime.registry.invocations
    assert runtime.model_proposals
    initial = runtime.model_proposals[0]
    assert initial.action == "invoke_tool"
    assert initial.tool_name == "search_events"
    # Label may be omitted by the local model; unfiltered search_events is still valid.
    assert initial.label in {"person", "car", None}

    follow_ups = runtime.model_proposals[1:]
    car_follow = any(
        p.action == "invoke_tool"
        and p.tool_name == "search_events"
        and p.label == "car"
        for p in follow_ups
    )
    temporal_follow = any(
        p.action == "invoke_tool" and p.tool_name == "analyze_temporal_relations"
        for p in follow_ups
    )
    proceeded = any(p.action == "proceed_to_report" for p in follow_ups) or (
        len(runtime.registry.invocations) == 1
        and runtime.last_model_proposal is not None
        and runtime.last_model_proposal.action == "proceed_to_report"
    )
    print(
        json.dumps(
            {
                "scenario": "t2_evidence_replan",
                "recording": str(recording),
                "invocations": list(runtime.registry.invocations),
                "proposals": [p.to_dict() for p in runtime.model_proposals],
                "car_follow_up": car_follow,
                "temporal_follow_up": temporal_follow,
                "receipts": len(runtime.store.receipts),
                "observations": len(runtime.store.observations),
                "gate_accepted": None
                if runtime.last_gate is None
                else runtime.last_gate.accepted,
                "fingerprints": len(runtime.executed_request_fingerprints),
            }
        )
    )
    # Prefer person→car or temporal when person hits exist; allow stop if model declines.
    if (initial.label in {"person", None}) and len(runtime.store.observations) > 0:
        assert car_follow or temporal_follow or proceeded
        if car_follow or temporal_follow:
            assert len(runtime.registry.invocations) >= 2
            assert len(runtime.store.receipts) >= 2
            assert runtime.last_guard is not None and runtime.last_guard.accepted
    assert runtime.last_gate is not None and runtime.last_gate.accepted
    # Anti-repeat: no identical fingerprint executed twice.
    assert len(runtime.executed_request_fingerprints) == len(
        runtime.registry.invocations
    )


def test_t2_real_ollama_empty_contrast_recorded():
    provider = _require_local_model()
    case = _fixture_case(case_id="case-t2-ollama-empty", question=FOLLOW_UP_QUESTION)
    scenario = _nl_scenario(
        name="t2_ollama_empty",
        frigate_recording_path=str(FIXTURES / "events_empty.json"),
        max_model_calls=3,
    )
    runtime = InvestigationRuntime.from_fixture(
        case=case, scenario=scenario, model_provider=provider
    )
    run_investigation(runtime, thread_id="thread-t2-ollama-empty")
    print(
        json.dumps(
            {
                "scenario": "t2_empty_control",
                "invocations": list(runtime.registry.invocations),
                "proposals": [p.to_dict() for p in runtime.model_proposals],
                "observations": len(runtime.store.observations),
            }
        )
    )
    assert runtime.registry.invocations[0] == "search_events"
    assert len(runtime.store.observations) == 0
    assert "search_related_events" not in runtime.registry.invocations
    # After EMPTY, model should stop or only propose guarded search_events.
    for proposal in runtime.model_proposals[1:]:
        if proposal.action == "invoke_tool":
            assert proposal.tool_name == "search_events"
            assert runtime.last_guard is not None and runtime.last_guard.accepted
        else:
            assert proposal.action == "proceed_to_report"
