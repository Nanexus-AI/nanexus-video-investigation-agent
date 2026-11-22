"""V0.1 T3: temporal analysis tool + shared Agent orchestration.

Deterministic StaticModelProvider / unit tests always run.
Genuine Ollama tests require reachable qwen3:14b and fail hard when unavailable.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from tests.e1.live_scope import require_live_cameras, require_live_time_window

from nanexus_investigation.contracts.enums import (
    CaseStatus,
    TemporalRelationType,
    ToolResultStatus,
)
from nanexus_investigation.contracts.identity import Interval
from nanexus_investigation.contracts.scope import InvestigationScope
from nanexus_investigation.contracts.tool import ToolRequest
from nanexus_investigation.graph.runtime import (
    ScenarioConfig,
    make_case,
    ok_event_result,
)
from nanexus_investigation.graph.workflow import InvestigationRuntime, run_investigation
from nanexus_investigation.model.provider import (
    InvestigationProposal,
    OllamaChatProvider,
    StaticModelProvider,
)
from nanexus_investigation.tools.temporal_analysis import analyze_temporal_relations

FIXTURES = Path(__file__).parent / "fixtures" / "frigate"
PRIVATE_RECORDING = Path(".local/frigate_recordings/events_list_live.json")
UTC = timezone.utc
VMS = "frigate-fixture-0.17.1"
OLLAMA_URL = "http://127.0.0.1:11434"
MODEL_NAME = "qwen3:14b"

PERSON_Q = "Find person detections in this camera and time window."
NEARBY_Q = (
    "Find vehicle activity within three minutes before or after person "
    "detections in this time window, and build a timeline. "
    "Do not invent physical interaction from timestamps alone."
)


def _require_local_model() -> OllamaChatProvider:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        pytest.fail(f"T3 requires reachable local Ollama at {OLLAMA_URL}: {exc}")
    names = {m.get("name") for m in payload.get("models", []) if isinstance(m, dict)}
    if MODEL_NAME not in names:
        pytest.fail(
            f"T3 requires installed model {MODEL_NAME!r}; installed={sorted(names)}"
        )
    return OllamaChatProvider(model=MODEL_NAME, base_url=OLLAMA_URL, timeout_s=180.0)


def _fixture_case(*, case_id: str, question: str):
    return make_case(
        case_id=case_id,
        question=question,
        cameras=("front_yard",),
        start=datetime(2023, 11, 14, 22, 0, tzinfo=UTC),
        end=datetime(2023, 11, 14, 23, 30, tzinfo=UTC),
        vms_instance_id="frigate-fixture",
    )


def _nl_scenario(**overrides) -> ScenarioConfig:
    base = dict(
        name="t3_nl",
        tool_backend="frigate_recorded",
        frigate_recording_path=str(FIXTURES / "events_list_sanitized.json"),
        use_model_initial_plan=True,
        use_model_follow_up=True,
        max_model_calls=5,
        max_tool_attempts=5,
        max_plan_steps=6,
        include_follow_up=False,
        search_label="person",
    )
    base.update(overrides)
    return ScenarioConfig(**base)


def test_t3_unit_near_match_and_nonmatch():
    scope = InvestigationScope(
        vms_instance_id="vms-1",
        camera_ids=("front_yard",),
        time_bounds=Interval(
            start_utc=datetime(2023, 11, 14, 22, 0, tzinfo=UTC),
            end_utc=datetime(2023, 11, 14, 23, 30, tzinfo=UTC),
        ),
    )
    t0 = datetime(2023, 11, 14, 22, 10, tzinfo=UTC)
    from nanexus_investigation.contracts.enums import ObservationType, RecordType
    from nanexus_investigation.contracts.evidence import Observation
    from nanexus_investigation.contracts.identity import SourceLocator

    def _obs(oid: str, label: str, start: datetime, end: datetime) -> Observation:
        return Observation(
            observation_id=oid,
            source_locator=SourceLocator(
                vms_instance_id="vms-1",
                record_type=RecordType.EVENT,
                native_record_id=f"native-{oid}",
            ),
            source_revision="rev-1",
            observation_type=ObservationType.DETECTION_EVENT,
            camera_id="front_yard",
            start_utc=start,
            end_utc=end,
            labels=(label,),
            retrieved_at=t0,
            provenance="test",
        )

    near_obs = {
        "obs-p": _obs("obs-p", "person", t0, t0 + timedelta(seconds=20)),
        "obs-c": _obs(
            "obs-c",
            "car",
            t0 + timedelta(seconds=40),
            t0 + timedelta(seconds=60),
        ),
    }
    ok = analyze_temporal_relations(
        ToolRequest(
            tool_name="analyze_temporal_relations",
            arguments={
                "predicate": "near",
                "window_seconds": 180,
                "anchor_label": "person",
                "candidate_label": "car",
            },
            source_id="vms-1",
            camera_ids=("front_yard",),
        ),
        observations=near_obs,
        scope=scope,
    )
    assert ok.status is ToolResultStatus.OK
    assert ok.data and ok.data[0]["holds"] is True
    assert ok.data[0]["relation_type"] == "near"

    far_obs = {
        "obs-p": _obs("obs-p", "person", t0, t0 + timedelta(seconds=10)),
        "obs-c": _obs(
            "obs-c",
            "car",
            t0 + timedelta(minutes=30),
            t0 + timedelta(minutes=31),
        ),
    }
    empty = analyze_temporal_relations(
        ToolRequest(
            tool_name="analyze_temporal_relations",
            arguments={
                "predicate": "near",
                "window_seconds": 180,
                "anchor_label": "person",
                "candidate_label": "car",
            },
            source_id="vms-1",
            camera_ids=("front_yard",),
        ),
        observations=far_obs,
        scope=scope,
    )
    assert empty.status is ToolResultStatus.EMPTY
    assert empty.data in (None, [])


def test_t3_unit_rejects_foreign_ids_and_excessive_window():
    scope = InvestigationScope(
        vms_instance_id="vms-1",
        camera_ids=("front_yard",),
        time_bounds=Interval(
            start_utc=datetime(2023, 11, 14, 22, 0, tzinfo=UTC),
            end_utc=datetime(2023, 11, 14, 23, 30, tzinfo=UTC),
        ),
    )
    bad = analyze_temporal_relations(
        ToolRequest(
            tool_name="analyze_temporal_relations",
            arguments={
                "predicate": "near",
                "window_seconds": 180,
                "anchor_observation_ids": ["obs-missing"],
                "candidate_observation_ids": ["obs-missing-2"],
            },
            source_id="vms-1",
            camera_ids=("front_yard",),
        ),
        observations={},
        scope=scope,
    )
    assert bad.status is ToolResultStatus.ERROR
    assert "unknown or foreign" in bad.error.message

    huge = analyze_temporal_relations(
        ToolRequest(
            tool_name="analyze_temporal_relations",
            arguments={"predicate": "near", "window_seconds": 99999},
            source_id="vms-1",
            camera_ids=("front_yard",),
        ),
        observations={},
        scope=scope,
        max_window_seconds=3600,
    )
    assert huge.status is ToolResultStatus.ERROR
    assert "exceeds max_window_seconds" in huge.error.message


def test_t3_01_static_simple_retrieval_skips_temporal():
    case = _fixture_case(case_id="case-t3-01", question=PERSON_Q)
    provider = StaticModelProvider(
        initial_proposal=InvestigationProposal(
            action="invoke_tool",
            tool_name="search_events",
            label="person",
            rationale="person search only",
            camera_ids=("front_yard",),
        ),
        proposals=[
            InvestigationProposal(
                action="proceed_to_report",
                tool_name=None,
                label=None,
                rationale="retrieval complete; temporal not needed",
            )
        ],
    )
    runtime = InvestigationRuntime.from_fixture(
        case=case, scenario=_nl_scenario(name="t3_01"), model_provider=provider
    )
    run_investigation(runtime, thread_id="thread-t3-01")
    assert runtime.registry.invocations == ["search_events"]
    assert "analyze_temporal_relations" not in runtime.registry.invocations
    assert runtime.last_gate is not None and runtime.last_gate.accepted


def test_t3_02_static_nearby_person_car_temporal_timeline():
    """Person + car retrieval then model-selected temporal NEAR analysis."""
    case = _fixture_case(case_id="case-t3-02", question=NEARBY_Q)
    # Build stub path with controllable near intervals via Frigate recording
    # is hard; use stub backend with scripted events then temporal engine.
    t0 = datetime(2023, 11, 14, 22, 15, tzinfo=UTC)
    person = ok_event_result(
        execution_id="exec-t3-person",
        native_id="evt-person-1",
        label="person",
        start=t0,
        end=t0 + timedelta(seconds=15),
        camera_id="front_yard",
    )
    car = ok_event_result(
        execution_id="exec-t3-car",
        native_id="evt-car-1",
        label="car",
        start=t0 + timedelta(seconds=45),
        end=t0 + timedelta(seconds=70),
        camera_id="front_yard",
    )
    provider = StaticModelProvider(
        initial_proposal=InvestigationProposal(
            action="invoke_tool",
            tool_name="search_events",
            label="person",
            rationale="retrieve persons first",
            camera_ids=("front_yard",),
        ),
        proposals=[
            InvestigationProposal(
                action="invoke_tool",
                tool_name="search_events",
                label="car",
                rationale="then vehicles/cars",
                camera_ids=("front_yard",),
            ),
            InvestigationProposal(
                action="invoke_tool",
                tool_name="analyze_temporal_relations",
                label=None,
                rationale="analyze NEAR within 3 minutes",
                camera_ids=("front_yard",),
                predicate="near",
                window_seconds=180,
                anchor_label="person",
                candidate_label="car",
            ),
            InvestigationProposal(
                action="proceed_to_report",
                tool_name=None,
                label=None,
                rationale="timeline ready",
            ),
        ],
    )
    scenario = ScenarioConfig(
        name="t3_02_stub",
        tool_backend="stub",
        use_model_initial_plan=True,
        use_model_follow_up=True,
        max_model_calls=5,
        max_tool_attempts=5,
        max_plan_steps=6,
        tool_scripts={
            "search_events": [person, car],
        },
    )
    runtime = InvestigationRuntime.from_fixture(
        case=case, scenario=scenario, model_provider=provider
    )
    run_investigation(runtime, thread_id="thread-t3-02")

    assert runtime.registry.invocations == [
        "search_events",
        "search_events",
        "analyze_temporal_relations",
    ]
    assert any(
        r.relation_type is TemporalRelationType.NEAR for r in runtime.store.relations.values()
    )
    assert runtime.store.report is not None
    assert any(rid.startswith("rel-") for rid in runtime.store.report.timeline)
    joined = " ".join(c.statement.lower() for c in runtime.store.claims.values())
    assert "interaction" in joined  # disclaimer present
    assert "does not prove physical interaction" in joined
    assert runtime.last_gate is not None and runtime.last_gate.accepted
    assert runtime.case.status in {CaseStatus.COMPLETED, CaseStatus.PARTIAL}


def test_t3_03_static_no_temporal_match():
    t0 = datetime(2023, 11, 14, 22, 15, tzinfo=UTC)
    person = ok_event_result(
        execution_id="exec-t3-nm-p",
        native_id="evt-p-far",
        label="person",
        start=t0,
        end=t0 + timedelta(seconds=10),
        camera_id="front_yard",
    )
    car = ok_event_result(
        execution_id="exec-t3-nm-c",
        native_id="evt-c-far",
        label="car",
        start=t0 + timedelta(minutes=20),
        end=t0 + timedelta(minutes=21),
        camera_id="front_yard",
    )
    provider = StaticModelProvider(
        initial_proposal=InvestigationProposal(
            action="invoke_tool",
            tool_name="search_events",
            label="person",
            rationale="person",
            camera_ids=("front_yard",),
        ),
        proposals=[
            InvestigationProposal(
                action="invoke_tool",
                tool_name="search_events",
                label="car",
                rationale="car",
                camera_ids=("front_yard",),
            ),
            InvestigationProposal(
                action="invoke_tool",
                tool_name="analyze_temporal_relations",
                label=None,
                rationale="near check",
                camera_ids=("front_yard",),
                predicate="near",
                window_seconds=180,
                anchor_label="person",
                candidate_label="car",
            ),
            InvestigationProposal(
                action="proceed_to_report",
                tool_name=None,
                label=None,
                rationale="no near matches",
            ),
        ],
    )
    case = _fixture_case(case_id="case-t3-03", question=NEARBY_Q)
    runtime = InvestigationRuntime.from_fixture(
        case=case,
        scenario=ScenarioConfig(
            name="t3_03",
            tool_backend="stub",
            use_model_initial_plan=True,
            use_model_follow_up=True,
            max_model_calls=5,
            max_tool_attempts=5,
            tool_scripts={"search_events": [person, car]},
        ),
        model_provider=provider,
    )
    run_investigation(runtime, thread_id="thread-t3-03")
    assert "analyze_temporal_relations" in runtime.registry.invocations
    assert not any(
        r.relation_type is TemporalRelationType.NEAR
        and r.evaluation_status.value == "established"
        for r in runtime.store.relations.values()
    )
    # EMPTY temporal receipt present; no fabricated near claims.
    temporal_receipts = [
        r
        for r in runtime.store.receipts.values()
        if r.tool_name == "analyze_temporal_relations"
    ]
    assert temporal_receipts
    tool = runtime.store.tool_results_by_receipt[temporal_receipts[0].search_execution_id]
    assert tool.status is ToolResultStatus.EMPTY
    assert runtime.last_gate is not None and runtime.last_gate.accepted


def test_t3_04_authorization_boundaries():
    case = _fixture_case(case_id="case-t3-04", question=NEARBY_Q)
    # Unsupported related-events still fails closed under Frigate.
    provider = StaticModelProvider(
        initial_proposal=InvestigationProposal(
            action="invoke_tool",
            tool_name="search_events",
            label="person",
            rationale="person",
            camera_ids=("front_yard",),
        ),
        proposals=[
            InvestigationProposal(
                action="invoke_tool",
                tool_name="search_related_events",
                label="vehicle",
                rationale="unsupported",
                camera_ids=("front_yard",),
            )
        ],
    )
    runtime = InvestigationRuntime.from_fixture(
        case=case,
        scenario=_nl_scenario(name="t3_04_unsupported", max_model_calls=3),
        model_provider=provider,
    )
    run_investigation(runtime, thread_id="thread-t3-04")
    assert runtime.case.status is CaseStatus.FAILED or runtime.last_error
    assert "search_related_events" not in [
        i for i in runtime.registry.invocations if i == "search_related_events"
    ] or runtime.registry.invocations.count("search_related_events") == 0

    # Excessive window rejected by temporal engine after Guard (scope time OK).
    t0 = datetime(2023, 11, 14, 22, 15, tzinfo=UTC)
    person = ok_event_result(
        execution_id="exec-t3-04-p",
        native_id="evt-p",
        label="person",
        start=t0,
        end=t0 + timedelta(seconds=5),
        camera_id="front_yard",
    )
    provider2 = StaticModelProvider(
        initial_proposal=InvestigationProposal(
            action="invoke_tool",
            tool_name="search_events",
            label="person",
            rationale="person",
            camera_ids=("front_yard",),
        ),
        proposals=[
            InvestigationProposal(
                action="invoke_tool",
                tool_name="analyze_temporal_relations",
                label=None,
                rationale="too wide",
                camera_ids=("front_yard",),
                predicate="near",
                window_seconds=99999,
                anchor_label="person",
                candidate_label="car",
            )
        ],
    )
    runtime2 = InvestigationRuntime.from_fixture(
        case=_fixture_case(case_id="case-t3-04b", question=NEARBY_Q),
        scenario=ScenarioConfig(
            name="t3_04_window",
            tool_backend="stub",
            use_model_initial_plan=True,
            use_model_follow_up=True,
            max_model_calls=3,
            max_tool_attempts=3,
            tool_scripts={"search_events": [person]},
        ),
        model_provider=provider2,
    )
    run_investigation(runtime2, thread_id="thread-t3-04b")
    assert "analyze_temporal_relations" in runtime2.registry.invocations
    temporal_receipts = [
        r
        for r in runtime2.store.receipts.values()
        if r.tool_name == "analyze_temporal_relations"
    ]
    assert temporal_receipts
    tool = runtime2.store.tool_results_by_receipt[
        temporal_receipts[0].search_execution_id
    ]
    assert tool.status is ToolResultStatus.ERROR


def test_t3_capability_advertises_temporal_for_frigate():
    case = _fixture_case(case_id="case-t3-caps", question=PERSON_Q)
    runtime = InvestigationRuntime.from_fixture(
        case=case,
        scenario=_nl_scenario(
            name="t3_caps",
            use_model_initial_plan=False,
            use_model_follow_up=False,
        ),
    )
    cap = runtime.capabilities.for_operation("analyze_temporal_relations")
    assert cap is not None
    assert cap.availability.value == "supported"


def test_t3_real_ollama_nearby_orchestration_recorded():
    provider = _require_local_model()
    recording = (
        PRIVATE_RECORDING
        if PRIVATE_RECORDING.is_file()
        else FIXTURES / "events_list_sanitized.json"
    )
    if recording == PRIVATE_RECORDING:
        case = make_case(
            case_id="case-t3-ollama-near",
            question=NEARBY_Q,
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
        case = _fixture_case(case_id="case-t3-ollama-near", question=NEARBY_Q)
        force_bounds = None

    runtime = InvestigationRuntime.from_fixture(
        case=case,
        scenario=_nl_scenario(
            name="t3_ollama_near",
            frigate_recording_path=str(recording),
            force_time_bounds=force_bounds,
            max_model_calls=6,
            max_tool_attempts=6,
        ),
        model_provider=provider,
    )
    run_investigation(runtime, thread_id="thread-t3-ollama-near")

    print(
        json.dumps(
            {
                "invocations": list(runtime.registry.invocations),
                "proposals": [p.to_dict() for p in runtime.model_proposals],
                "relations": len(runtime.store.relations),
                "observations": len(runtime.store.observations),
                "receipts": len(runtime.store.receipts),
                "timeline": list(runtime.store.report.timeline)
                if runtime.store.report
                else [],
                "gate": None
                if runtime.last_gate is None
                else runtime.last_gate.accepted,
                "status": runtime.case.status.value,
            }
        )
    )
    assert runtime.registry.invocations
    assert runtime.registry.invocations[0] == "search_events"
    assert "search_related_events" not in runtime.registry.invocations
    # Model should eventually use temporal analysis when both label classes exist,
    # or stop safely after retrievals.
    used_temporal = "analyze_temporal_relations" in runtime.registry.invocations
    if used_temporal:
        assert runtime.last_guard is not None and runtime.last_guard.accepted
        # Relations may be zero if no pairs are within 3 minutes — that is OK.
        assert runtime.store.report is not None
        claim_text = " ".join(
            f"{c.statement} {' '.join(c.limitations)}"
            for c in runtime.store.claims.values()
        ).lower()
        report_text = " ".join(runtime.store.report.limitations).lower()
        assert (
            "proximity" in report_text
            or "interaction" in report_text
            or "proximity" in claim_text
            or "interaction" in claim_text
            or "near relation" in claim_text
            or "temporal analysis" in claim_text
        )
    assert runtime.last_gate is not None and runtime.last_gate.accepted


def test_t3f_unit_pair_cap_truncation_semantics():
    """max_pairs stops evaluation; coverage is TRUNCATED, not exhaustive."""
    from nanexus_investigation.contracts.enums import (
        ObservationType,
        RecordType,
        SearchCoverage,
    )
    from nanexus_investigation.contracts.evidence import Observation
    from nanexus_investigation.contracts.identity import SourceLocator

    scope = InvestigationScope(
        vms_instance_id="vms-1",
        camera_ids=("front_yard",),
        time_bounds=Interval(
            start_utc=datetime(2023, 11, 14, 22, 0, tzinfo=UTC),
            end_utc=datetime(2023, 11, 14, 23, 30, tzinfo=UTC),
        ),
    )
    t0 = datetime(2023, 11, 14, 22, 10, tzinfo=UTC)

    def _obs(oid: str, label: str, start: datetime, end: datetime) -> Observation:
        return Observation(
            observation_id=oid,
            source_locator=SourceLocator(
                vms_instance_id="vms-1",
                record_type=RecordType.EVENT,
                native_record_id=f"native-{oid}",
            ),
            source_revision="rev-1",
            observation_type=ObservationType.DETECTION_EVENT,
            camera_id="front_yard",
            start_utc=start,
            end_utc=end,
            labels=(label,),
            retrieved_at=t0,
            provenance="test",
        )

    observations = {
        "obs-p": _obs("obs-p", "person", t0, t0 + timedelta(seconds=10)),
        "obs-c1": _obs(
            "obs-c1", "car", t0 + timedelta(seconds=20), t0 + timedelta(seconds=30)
        ),
        "obs-c2": _obs(
            "obs-c2", "car", t0 + timedelta(seconds=40), t0 + timedelta(seconds=50)
        ),
        "obs-c3": _obs(
            "obs-c3", "car", t0 + timedelta(seconds=60), t0 + timedelta(seconds=70)
        ),
    }
    result = analyze_temporal_relations(
        ToolRequest(
            tool_name="analyze_temporal_relations",
            arguments={
                "predicate": "near",
                "window_seconds": 180,
                "anchor_label": "person",
                "candidate_label": "car",
                "max_pairs": 1,
            },
            source_id="vms-1",
            camera_ids=("front_yard",),
        ),
        observations=observations,
        scope=scope,
        max_pairs=100,
    )
    assert result.coverage is SearchCoverage.TRUNCATED
    assert result.usage["pairs_evaluated"] == 1
    assert result.usage["truncated"] == 1
    assert result.status in {ToolResultStatus.OK, ToolResultStatus.EMPTY}


def test_t3f_static_pair_cap_report_limitation():
    """Truncated temporal analysis must not claim exhaustive absence of matches."""
    from nanexus_investigation.contracts.enums import (
        RecordType,
        SearchCoverage,
        ToolResultStatus as TRS,
    )
    from nanexus_investigation.contracts.tool import ToolResult

    t0 = datetime(2023, 11, 14, 22, 15, tzinfo=UTC)
    person = ok_event_result(
        execution_id="exec-t3f-cap-p",
        native_id="evt-cap-p",
        label="person",
        start=t0,
        end=t0 + timedelta(seconds=10),
        camera_id="front_yard",
    )
    cars = ToolResult(
        status=TRS.OK,
        execution_id="exec-t3f-cap-cars",
        data=[
            {
                "native_record_id": "evt-cap-ca",
                "record_type": RecordType.EVENT,
                "camera_id": "front_yard",
                "label": "car",
                "start_utc": t0 + timedelta(minutes=20),
                "end_utc": t0 + timedelta(minutes=21),
                "source_revision": "rev-1",
            },
            {
                "native_record_id": "evt-cap-cb",
                "record_type": RecordType.EVENT,
                "camera_id": "front_yard",
                "label": "car",
                "start_utc": t0 + timedelta(minutes=25),
                "end_utc": t0 + timedelta(minutes=26),
                "source_revision": "rev-1",
            },
            {
                "native_record_id": "evt-cap-cc",
                "record_type": RecordType.EVENT,
                "camera_id": "front_yard",
                "label": "car",
                "start_utc": t0 + timedelta(minutes=30),
                "end_utc": t0 + timedelta(minutes=31),
                "source_revision": "rev-1",
            },
        ],
        source_id="vms-local-1",
        retrieved_at=datetime(2026, 11, 20, 12, 5, tzinfo=UTC),
        coverage=SearchCoverage.EXHAUSTED,
        provenance="stub:search/cars",
        usage={"tool_attempts": 1},
    )
    provider = StaticModelProvider(
        initial_proposal=InvestigationProposal(
            action="invoke_tool",
            tool_name="search_events",
            label="person",
            rationale="person",
            camera_ids=("front_yard",),
        ),
        proposals=[
            InvestigationProposal(
                action="invoke_tool",
                tool_name="search_events",
                label="car",
                rationale="cars",
                camera_ids=("front_yard",),
            ),
            InvestigationProposal(
                action="invoke_tool",
                tool_name="analyze_temporal_relations",
                label=None,
                rationale="bounded near",
                camera_ids=("front_yard",),
                predicate="near",
                window_seconds=180,
                anchor_label="person",
                candidate_label="car",
                max_pairs=1,
            ),
            InvestigationProposal(
                action="proceed_to_report",
                tool_name=None,
                label=None,
                rationale="done",
            ),
        ],
    )
    case = _fixture_case(case_id="case-t3f-cap", question=NEARBY_Q)
    runtime = InvestigationRuntime.from_fixture(
        case=case,
        scenario=ScenarioConfig(
            name="t3f_cap",
            tool_backend="stub",
            use_model_initial_plan=True,
            use_model_follow_up=True,
            max_model_calls=5,
            max_tool_attempts=5,
            tool_scripts={"search_events": [person, cars]},
        ),
        model_provider=provider,
    )
    run_investigation(runtime, thread_id="thread-t3f-cap")
    temporal_receipts = [
        r
        for r in runtime.store.receipts.values()
        if r.tool_name == "analyze_temporal_relations"
    ]
    assert temporal_receipts
    assert temporal_receipts[0].coverage is SearchCoverage.TRUNCATED
    receipt_limit = (temporal_receipts[0].consistency_limitations or "").lower()
    assert "truncated" in receipt_limit or "max_pairs" in receipt_limit
    assert "not proven" in receipt_limit or "not exhaustive" in receipt_limit
    assert runtime.store.report is not None
    report_limits = " ".join(runtime.store.report.limitations).lower()
    assert "truncated" in report_limits or "max_pairs" in report_limits
    claim_text = " ".join(c.statement.lower() for c in runtime.store.claims.values())
    assert "unevaluated" in claim_text or "truncated" in claim_text
    assert runtime.last_gate is not None and runtime.last_gate.accepted


def test_t3f_durable_relation_survives_runtime_reinit(tmp_path: Path):
    """Cited TemporalRelation IDs resolve after discarding the in-memory runtime."""
    from nanexus_investigation.contracts.enums import TemporalEvaluationStatus
    from nanexus_investigation.persistence.db import InvestigationDatabase
    from nanexus_investigation.persistence.repository import InvestigationRepository

    t0 = datetime(2023, 11, 14, 22, 15, tzinfo=UTC)
    person = ok_event_result(
        execution_id="exec-t3f-dur-p",
        native_id="evt-dur-p",
        label="person",
        start=t0,
        end=t0 + timedelta(seconds=15),
        camera_id="front_yard",
    )
    car = ok_event_result(
        execution_id="exec-t3f-dur-c",
        native_id="evt-dur-c",
        label="car",
        start=t0 + timedelta(seconds=45),
        end=t0 + timedelta(seconds=70),
        camera_id="front_yard",
    )
    provider = StaticModelProvider(
        initial_proposal=InvestigationProposal(
            action="invoke_tool",
            tool_name="search_events",
            label="person",
            rationale="person",
            camera_ids=("front_yard",),
        ),
        proposals=[
            InvestigationProposal(
                action="invoke_tool",
                tool_name="search_events",
                label="car",
                rationale="car",
                camera_ids=("front_yard",),
            ),
            InvestigationProposal(
                action="invoke_tool",
                tool_name="analyze_temporal_relations",
                label=None,
                rationale="near",
                camera_ids=("front_yard",),
                predicate="near",
                window_seconds=180,
                anchor_label="person",
                candidate_label="car",
            ),
            InvestigationProposal(
                action="proceed_to_report",
                tool_name=None,
                label=None,
                rationale="done",
            ),
        ],
    )
    domain_db = InvestigationDatabase.open(tmp_path / "t3f-domain.sqlite")
    case = _fixture_case(case_id="case-t3f-durable", question=NEARBY_Q)
    runtime = InvestigationRuntime.from_durable_fixture(
        case=case,
        scenario=ScenarioConfig(
            name="t3f_durable",
            tool_backend="stub",
            use_model_initial_plan=True,
            use_model_follow_up=True,
            max_model_calls=5,
            max_tool_attempts=5,
            tool_scripts={"search_events": [person, car]},
        ),
        domain_db=domain_db,
        model_provider=provider,
    )
    run_investigation(runtime, thread_id="thread-t3f-durable")
    assert runtime.last_gate is not None and runtime.last_gate.accepted
    assert runtime.store.report is not None
    cited_relations = [
        entry for entry in runtime.store.report.timeline if entry.startswith("rel-")
    ]
    assert cited_relations
    # Automatic BEFORE must not remain citeable alongside durable NEAR.
    assert not any(rid.endswith("-before") for rid in cited_relations)
    relation_snapshot = {
        rid: runtime.store.relations[rid] for rid in cited_relations
    }
    assert all(
        rel.relation_type is TemporalRelationType.NEAR
        for rel in relation_snapshot.values()
    )
    receipt_snapshot = {
        rid: r
        for rid, r in runtime.store.receipts.items()
        if r.tool_name == "analyze_temporal_relations"
    }
    assert receipt_snapshot

    del runtime
    repo = InvestigationRepository(domain_db)
    for relation_id, expected in relation_snapshot.items():
        loaded = repo.get_temporal_relation(relation_id)
        assert loaded is not None
        assert loaded.relation_id == expected.relation_id
        assert loaded.left_observation_id == expected.left_observation_id
        assert loaded.right_observation_id == expected.right_observation_id
        assert loaded.relation_type is TemporalRelationType.NEAR
        assert loaded.evaluation_status is TemporalEvaluationStatus.ESTABLISHED
        assert loaded.window_seconds == expected.window_seconds
        assert repo.get_observation(loaded.left_observation_id) is not None
        assert repo.get_observation(loaded.right_observation_id) is not None

    for receipt in receipt_snapshot.values():
        durable_receipt = repo.get_search_receipt(receipt.search_execution_id)
        assert durable_receipt is not None
        assert durable_receipt.filters.get("predicate") == "near"
        assert durable_receipt.filters.get("window_seconds") == 180
        assert durable_receipt.filters.get("anchor_label") == "person"
        assert durable_receipt.filters.get("candidate_label") == "car"
    domain_db.close()


def test_t3f_atomic_idempotent_relation_registration(tmp_path: Path):
    """Relations cannot commit without durable sources; replay reuses the group."""
    from nanexus_investigation.contracts.enums import (
        CaseStatus,
        EvidenceAvailability,
        EvidenceKind,
        ObservationType,
        RecordType,
        SearchCoverage,
        TemporalEvaluationStatus,
    )
    from nanexus_investigation.contracts.evidence import EvidenceRef, Observation
    from nanexus_investigation.contracts.identity import SourceLocator
    from nanexus_investigation.contracts.search import SearchExecutionReceipt
    from nanexus_investigation.contracts.temporal import TemporalRelation
    from nanexus_investigation.graph.runtime import make_case
    from nanexus_investigation.persistence.db import InvestigationDatabase
    from nanexus_investigation.persistence.errors import IdempotencyConflictError
    from nanexus_investigation.persistence.operation_id import build_operation_identity
    from nanexus_investigation.persistence.repository import InvestigationRepository

    db = InvestigationDatabase.open(tmp_path / "t3f-atomic.sqlite")
    case = make_case(case_id="case-t3f-atomic")
    if case.status is CaseStatus.CREATED:
        case = case.transition_to(CaseStatus.RUNNING)
    repo = InvestigationRepository(db)
    repo.upsert_case(case)

    t0 = datetime(2023, 11, 14, 22, 15, tzinfo=UTC)
    left = Observation(
        observation_id="obs-atomic-l",
        source_locator=SourceLocator(
            vms_instance_id="vms-local-1",
            record_type=RecordType.EVENT,
            native_record_id="evt-atomic-l",
        ),
        source_revision="rev-1",
        observation_type=ObservationType.DETECTION_EVENT,
        camera_id="front_yard",
        start_utc=t0,
        end_utc=t0 + timedelta(seconds=10),
        labels=("person",),
        retrieved_at=t0,
        provenance="stub",
    )
    right = Observation(
        observation_id="obs-atomic-r",
        source_locator=SourceLocator(
            vms_instance_id="vms-local-1",
            record_type=RecordType.EVENT,
            native_record_id="evt-atomic-r",
        ),
        source_revision="rev-1",
        observation_type=ObservationType.DETECTION_EVENT,
        camera_id="front_yard",
        start_utc=t0 + timedelta(seconds=30),
        end_utc=t0 + timedelta(seconds=40),
        labels=("car",),
        retrieved_at=t0,
        provenance="stub",
    )

    def _ev(obs: Observation) -> EvidenceRef:
        return EvidenceRef(
            evidence_id=f"ev-{obs.observation_id}",
            source_locator=obs.source_locator,
            evidence_kind=EvidenceKind.EVENT_METADATA,
            camera_id=obs.camera_id,
            interval=Interval(start_utc=obs.start_utc, end_utc=obs.end_utc),
            retrieval_spec={"operation": "search_events"},
            availability=EvidenceAvailability.AVAILABLE,
            provenance="stub",
        )

    source_op = build_operation_identity(
        case_id=case.case_id,
        plan_step_id="src",
        tool_name="search_events",
        vms_instance_id="vms-local-1",
        request_parameters={"label": "mixed"},
    )
    source_receipt = SearchExecutionReceipt(
        search_execution_id="receipt-src",
        tool_name="search_events",
        source_id="vms-local-1",
        camera_ids=("front_yard",),
        time_bounds=Interval(start_utc=t0, end_utc=t0 + timedelta(hours=1)),
        filters={"label": "mixed"},
        retrieved_at=t0,
        pages_fetched=1,
        match_count=2,
        coverage=SearchCoverage.EXHAUSTED,
    )
    repo.register_artifact_group(
        operation=source_op,
        receipt=source_receipt,
        observations=(left, right),
        evidences=(_ev(left), _ev(right)),
    )

    relation = TemporalRelation(
        relation_id="rel-obs-atomic-l-obs-atomic-r-near-180",
        left_observation_id=left.observation_id,
        right_observation_id=right.observation_id,
        relation_type=TemporalRelationType.NEAR,
        evaluation_status=TemporalEvaluationStatus.ESTABLISHED,
        window_seconds=180.0,
        gap_seconds=20.0,
        camera_relation="same_camera",
        time_basis="completed_half_open_utc_micros",
        calculation_provenance="test:near",
    )
    orphan = TemporalRelation(
        relation_id="rel-orphan",
        left_observation_id="obs-missing-l",
        right_observation_id="obs-missing-r",
        relation_type=TemporalRelationType.NEAR,
        evaluation_status=TemporalEvaluationStatus.ESTABLISHED,
        window_seconds=180.0,
        gap_seconds=1.0,
        camera_relation="same_camera",
        time_basis="completed_half_open_utc_micros",
        calculation_provenance="test:orphan",
    )
    temporal_receipt = SearchExecutionReceipt(
        search_execution_id="receipt-temporal",
        tool_name="analyze_temporal_relations",
        source_id="vms-local-1",
        camera_ids=("front_yard",),
        time_bounds=Interval(start_utc=t0, end_utc=t0 + timedelta(hours=1)),
        filters={"predicate": "near", "window_seconds": 180},
        retrieved_at=t0,
        pages_fetched=1,
        match_count=1,
        coverage=SearchCoverage.EXHAUSTED,
    )
    bad_op = build_operation_identity(
        case_id=case.case_id,
        plan_step_id="temporal-bad",
        tool_name="analyze_temporal_relations",
        vms_instance_id="vms-local-1",
        request_parameters={"predicate": "near", "window_seconds": 180},
    )
    with pytest.raises(ValueError, match="not durable"):
        repo.register_artifact_group(
            operation=bad_op,
            receipt=temporal_receipt,
            relations=(orphan,),
        )
    assert repo.get_temporal_relation("rel-orphan") is None
    assert repo.get_registration_group(bad_op.operation_id) is None

    good_op = build_operation_identity(
        case_id=case.case_id,
        plan_step_id="temporal-good",
        tool_name="analyze_temporal_relations",
        vms_instance_id="vms-local-1",
        request_parameters={"predicate": "near", "window_seconds": 180},
    )
    first = repo.register_artifact_group(
        operation=good_op,
        receipt=temporal_receipt,
        relations=(relation,),
    )
    assert first.reused is False
    assert len(first.relations) == 1
    second = repo.register_artifact_group(
        operation=good_op,
        receipt=temporal_receipt,
        relations=(relation,),
    )
    assert second.reused is True
    assert second.relations[0].relation_id == relation.relation_id

    conflict_receipt = temporal_receipt.model_copy(update={"match_count": 2})
    with pytest.raises(IdempotencyConflictError):
        repo.register_artifact_group(
            operation=good_op,
            receipt=conflict_receipt,
            relations=(relation,),
        )
    db.close()
