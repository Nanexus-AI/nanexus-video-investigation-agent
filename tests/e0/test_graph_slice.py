"""E0 LangGraph investigation slice. Deterministic stubs only — no VMS/LLM."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from nanexus_investigation.contracts.capability import (
    CapabilitySnapshot,
    OperationCapability,
)
from nanexus_investigation.contracts.enums import (
    CapabilityAvailability,
    CaseStatus,
    ClaimValidationStatus,
    ObservationType,
    RecordType,
    SearchCoverage,
    TemporalEvaluationStatus,
    ToolErrorCode,
    ToolResultStatus,
)
from nanexus_investigation.contracts.evidence import Observation
from nanexus_investigation.contracts.identity import Interval, SourceLocator
from nanexus_investigation.contracts.tool import ToolError, ToolResult
from nanexus_investigation.evidence.claims import observation_interval
from nanexus_investigation.graph.runtime import (
    ScenarioConfig,
    empty_result,
    evaluate_temporal,
    make_case,
    ok_event_result,
)
from nanexus_investigation.graph.workflow import InvestigationRuntime, run_investigation
from nanexus_investigation.harness.tool_guard import ToolGuard
from nanexus_investigation.tools.registry import StubToolRegistry

UTC = timezone.utc
T0 = datetime(2026, 11, 19, 20, 0, tzinfo=UTC)
T1 = datetime(2026, 11, 19, 23, 0, tzinfo=UTC)


def _person_ok(execution_id: str = "exec-person-1") -> ToolResult:
    return ok_event_result(
        execution_id=execution_id,
        native_id="evt-person-1",
        label="person",
        start=T0 + timedelta(minutes=10),
        end=T0 + timedelta(minutes=12),
    )


def _vehicle_ok(execution_id: str = "exec-vehicle-1") -> ToolResult:
    return ok_event_result(
        execution_id=execution_id,
        native_id="evt-vehicle-1",
        label="vehicle",
        start=T0 + timedelta(minutes=15),
        end=T0 + timedelta(minutes=17),
    )


def _error_result(execution_id: str = "exec-error-1") -> ToolResult:
    return ToolResult(
        status=ToolResultStatus.ERROR,
        execution_id=execution_id,
        data=None,
        error=ToolError(
            code=ToolErrorCode.SOURCE_UNAVAILABLE,
            message="stub source unavailable",
        ),
        source_id="vms-local-1",
        retrieved_at=datetime(2026, 11, 20, 12, 5, tzinfo=UTC),
        coverage=SearchCoverage.UNKNOWN,
        provenance="stub:error",
        usage={"tool_attempts": 1},
    )


def _unsupported_result(execution_id: str = "exec-unsup-1") -> ToolResult:
    return ToolResult(
        status=ToolResultStatus.UNSUPPORTED,
        execution_id=execution_id,
        data=None,
        error=ToolError(
            code=ToolErrorCode.DISABLED_CAPABILITY,
            message="stub capability disabled",
        ),
        source_id="vms-local-1",
        retrieved_at=datetime(2026, 11, 20, 12, 5, tzinfo=UTC),
        coverage=SearchCoverage.UNKNOWN,
        provenance="stub:unsupported",
        usage={"tool_attempts": 1},
    )


def _run(scenario: ScenarioConfig, **case_kwargs):
    case = make_case(**case_kwargs)
    runtime = InvestigationRuntime.from_fixture(case=case, scenario=scenario)
    result = run_investigation(runtime, thread_id=f"thread-{scenario.name}")
    return result


def test_happy_path_produces_report_with_resolvable_citations():
    scenario = ScenarioConfig(
        name="happy_path",
        include_follow_up=False,
        max_tool_attempts=2,
        tool_scripts={"search_events": [_person_ok()]},
    )
    result = _run(scenario)
    state = result["state"]
    report = result["report"]
    runtime = result["runtime"]

    assert runtime.registry.invocations == ["search_events"]
    assert state["evidence_ids"]
    assert state["observation_ids"]
    assert state["search_receipt_ids"]
    assert report is not None
    assert report.case_id == runtime.case.case_id
    for evidence_id in report.evidence_index:
        assert evidence_id in runtime.store.evidence
    for claim_id in report.claims:
        claim = runtime.store.claims[claim_id]
        for evidence_id in claim.evidence_ids:
            assert evidence_id in runtime.store.evidence
        assessment = runtime.store.assessments[claim_id]
        assert assessment.status in {
            ClaimValidationStatus.SUPPORTED,
            ClaimValidationStatus.QUALIFIED,
        }
    assert runtime.case.status in {CaseStatus.COMPLETED, CaseStatus.PARTIAL}
    assert runtime.last_gate is not None and runtime.last_gate.accepted
    # Budget reserved before invoke.
    assert runtime.trace.of_kind("budget_reservation")
    assert runtime.trace.of_kind("budget_reservation")[0]["reserved"] is True
    assert runtime.retry_policy_attached is False


def test_conditional_follow_up_second_tool_only_when_justified():
    scenario = ScenarioConfig(
        name="conditional_follow_up",
        include_follow_up=True,
        max_tool_attempts=3,
        tool_scripts={
            "search_events": [_person_ok()],
            "search_related_events": [_vehicle_ok()],
        },
    )
    result = _run(scenario)
    runtime = result["runtime"]
    trace = runtime.trace

    assert runtime.registry.invocations == [
        "search_events",
        "search_related_events",
    ]
    sufficiency = trace.of_kind("sufficiency")
    assert any(item["route"] == "continue_search" for item in sufficiency)
    assert any(item["route"] == "proceed_to_report" for item in sufficiency)
    assert len(result["state"]["evidence_ids"]) == 2
    assert runtime.case.status in {CaseStatus.COMPLETED, CaseStatus.PARTIAL}
    nodes = trace.node_sequence()
    # Loop returns to select_validate_tool after sufficiency continue_search.
    assert nodes.count("select_validate_tool") == 2
    assert nodes.count("execute_tool") == 2


def test_no_unnecessary_follow_up_when_first_result_empty():
    scenario = ScenarioConfig(
        name="no_follow_up",
        include_follow_up=True,
        max_tool_attempts=3,
        tool_scripts={
            "search_events": [empty_result()],
            "search_related_events": [_vehicle_ok()],
        },
    )
    result = _run(scenario)
    runtime = result["runtime"]

    assert runtime.registry.invocations == ["search_events"]
    assert "search_related_events" not in runtime.registry.invocations
    routes = [item["route"] for item in runtime.trace.of_kind("sufficiency")]
    assert "continue_search" not in routes
    assert "proceed_to_report" in routes
    assert result["state"]["evidence_ids"] == []
    assert result["report"] is not None
    assert any(
        "real-world absence" in item.lower() or "EMPTY" in item
        for item in result["report"].limitations
    )


def test_unknown_tool_never_invokes_body():
    scenario = ScenarioConfig(
        name="unknown_tool",
        force_tool_name="drop_all_cameras",
        tool_scripts={},
    )
    result = _run(scenario)
    runtime = result["runtime"]
    assert runtime.registry.invocations == []
    assert runtime.trace.of_kind("guard_rejected")
    assert "unknown or unregistered tool" in (result["state"].get("last_error") or "")
    assert runtime.case.status is CaseStatus.FAILED


def test_unauthorized_camera_never_invokes_body():
    scenario = ScenarioConfig(
        name="unauthorized_camera",
        force_camera_ids=("back_alley",),
        tool_scripts={"search_events": [_person_ok()]},
    )
    result = _run(scenario)
    runtime = result["runtime"]
    assert runtime.registry.invocations == []
    reasons = runtime.trace.of_kind("guard_rejected")[0]["reasons"]
    assert any("unauthorized camera" in item for item in reasons)


def test_out_of_scope_time_never_invokes_body():
    scenario = ScenarioConfig(
        name="out_of_scope_time",
        force_time_bounds=Interval(
            start_utc=T0 - timedelta(days=1),
            end_utc=T0 - timedelta(hours=20),
        ),
        tool_scripts={"search_events": [_person_ok()]},
    )
    result = _run(scenario)
    runtime = result["runtime"]
    assert runtime.registry.invocations == []
    reasons = runtime.trace.of_kind("guard_rejected")[0]["reasons"]
    assert any("outside effective scope" in item for item in reasons)


def test_budget_exhausted_blocks_further_invocation():
    scenario = ScenarioConfig(
        name="budget_exhausted",
        include_follow_up=True,
        max_tool_attempts=1,
        tool_scripts={
            "search_events": [_person_ok()],
            "search_related_events": [_vehicle_ok()],
        },
    )
    result = _run(scenario)
    runtime = result["runtime"]
    assert runtime.registry.invocations == ["search_events"]
    assert runtime.budget.reserved == 1
    assert runtime.budget.remaining == 0
    assert "search_related_events" not in runtime.registry.invocations


def test_empty_search_not_real_world_absence():
    scenario = ScenarioConfig(
        name="empty_search",
        tool_scripts={"search_events": [empty_result(coverage=SearchCoverage.EXHAUSTED)]},
        inject_real_world_absence_claim=True,
    )
    result = _run(scenario)
    runtime = result["runtime"]
    assert runtime.registry.invocations == ["search_events"]
    assert result["state"]["evidence_ids"] == []
    assert runtime.case.status is CaseStatus.FAILED
    assert runtime.last_gate is not None
    assert runtime.last_gate.accepted is False
    assert any(
        "real-world absence" in item for item in runtime.last_gate.reasons
    )


def test_error_and_unsupported_distinct_from_empty_no_fabricated_evidence():
    for status_name, script in (
        ("error", _error_result()),
        ("unsupported", _unsupported_result()),
    ):
        scenario = ScenarioConfig(
            name=f"status_{status_name}",
            tool_scripts={"search_events": [script]},
        )
        result = _run(scenario)
        runtime = result["runtime"]
        assert runtime.registry.invocations == ["search_events"]
        assert result["state"]["evidence_ids"] == []
        assert result["state"]["observation_ids"] == []
        interpretations = runtime.trace.of_kind("search_interpretation")
        assert interpretations[0]["status"] == script.status.value
        assert interpretations[0]["allows_real_world_absence"] is False
        assert result["report"] is not None
        assert any(
            script.status.value in item for item in result["report"].limitations
        )
        assert any("distinct from EMPTY" in item for item in result["report"].limitations)


def test_evidence_integrity_blocks_unsupported_report_claims():
    scenario = ScenarioConfig(
        name="evidence_integrity",
        tool_scripts={"search_events": [_person_ok()]},
        inject_fabricated_evidence_claim=True,
    )
    result = _run(scenario)
    runtime = result["runtime"]
    assert runtime.last_gate is not None
    assert runtime.last_gate.accepted is False
    assert runtime.case.status is CaseStatus.FAILED
    assert any(
        "missing evidence" in item or "unregistered evidence" in item
        for item in runtime.last_gate.reasons
    )


def test_temporal_contradiction_not_supported_conclusion():
    scenario = ScenarioConfig(
        name="temporal_contradiction",
        include_follow_up=True,
        tool_scripts={
            "search_events": [_person_ok()],
            "search_related_events": [_vehicle_ok()],
        },
        inject_false_temporal_claim=True,
    )
    result = _run(scenario)
    runtime = result["runtime"]
    assessment = runtime.store.assessments.get("claim-false-temporal")
    assert assessment is not None
    assert assessment.status is ClaimValidationStatus.UNSUPPORTED
    assert any("contradicted by recalculation" in item for item in assessment.reasons)
    assert runtime.last_gate is not None
    assert runtime.last_gate.accepted is False
    assert runtime.case.status is CaseStatus.FAILED


def test_unknown_temporal_relation_not_false_contradiction():
    scenario = ScenarioConfig(
        name="unknown_temporal",
        tool_scripts={"search_events": [_person_ok()]},
        inject_unknown_temporal_claim=True,
    )
    result = _run(scenario)
    runtime = result["runtime"]
    assessment = runtime.store.assessments.get("claim-unknown-temporal")
    assert assessment is not None
    assert assessment.status is ClaimValidationStatus.UNVERIFIABLE
    assert not any("contradicted by recalculation" in item for item in assessment.reasons)


def test_bounded_execution_cannot_unbounded_loop():
    scenario = ScenarioConfig(
        name="bounded_execution",
        include_follow_up=True,
        max_plan_steps=1,
        max_tool_attempts=5,
        tool_scripts={
            "search_events": [_person_ok()],
            "search_related_events": [_vehicle_ok()],
        },
    )
    result = _run(scenario)
    runtime = result["runtime"]
    assert runtime.registry.invocations == ["search_events"]
    assert result["state"]["attempt_summary"]["tool_attempts"] == 1
    assert runtime.trace.node_sequence().count("execute_tool") == 1


def test_no_hidden_retries_each_execution_accounted():
    scenario = ScenarioConfig(
        name="no_hidden_retries",
        include_follow_up=True,
        max_tool_attempts=3,
        tool_scripts={
            "search_events": [_person_ok()],
            "search_related_events": [_vehicle_ok()],
        },
    )
    result = _run(scenario)
    runtime = result["runtime"]
    executed = runtime.trace.of_kind("tool_executed")
    reservations = runtime.trace.of_kind("budget_reservation")
    assert len(executed) == len(runtime.registry.invocations) == 2
    assert len(reservations) == 2
    assert all(item["reserved"] is True for item in reservations)
    assert runtime.budget.reserved == 2
    assert runtime.budget.consumed == 2
    assert runtime.retry_policy_attached is False
    # compile path does not attach retry policy; guard is not a retry loop
    assert not isinstance(runtime.guard, type)  # sanity
    assert isinstance(runtime.guard, ToolGuard)
    assert isinstance(runtime.registry, StubToolRegistry)


def test_unavailable_capability_never_invokes_body():
    case = make_case()
    caps = CapabilitySnapshot(
        vms_instance_id=case.effective_scope.vms_instance_id,
        operations=(
            OperationCapability(
                operation="search_events",
                availability=CapabilityAvailability.TEMPORARILY_UNAVAILABLE,
                authorized_camera_ids=frozenset(case.effective_scope.camera_ids),
            ),
            OperationCapability(
                operation="search_related_events",
                availability=CapabilityAvailability.SUPPORTED,
                authorized_camera_ids=frozenset(case.effective_scope.camera_ids),
            ),
        ),
    )
    scenario = ScenarioConfig(
        name="unavailable_capability",
        tool_scripts={"search_events": [_person_ok()]},
    )
    runtime = InvestigationRuntime.from_fixture(
        case=case, scenario=scenario, capabilities=caps
    )
    result = run_investigation(runtime, thread_id="thread-unavailable-cap")
    assert runtime.registry.invocations == []
    reasons = runtime.trace.of_kind("guard_rejected")[0]["reasons"]
    assert any("capability not supported" in item for item in reasons)
    assert result["state"].get("last_error")


def _seed_observation(
    runtime: InvestigationRuntime,
    *,
    obs_id: str,
    native_id: str,
    start: datetime,
    end: datetime | None,
) -> None:
    observation = Observation(
        observation_id=obs_id,
        source_locator=SourceLocator(
            vms_instance_id=runtime.case.effective_scope.vms_instance_id,
            record_type=RecordType.EVENT,
            native_record_id=native_id,
        ),
        source_revision="rev-1",
        observation_type=ObservationType.DETECTION_EVENT,
        camera_id=runtime.case.effective_scope.camera_ids[0],
        start_utc=start,
        end_utc=end,
        retrieved_at=T0,
        provenance="test:temporal-seed",
    )
    runtime.store.register_observation(observation)
    runtime.store.observation_intervals[obs_id] = observation_interval(observation)
    # Observations themselves are unchanged by temporal evaluation.


def test_temporal_genuine_before_materializes_established():
    runtime = InvestigationRuntime.from_fixture(
        case=make_case(), scenario=ScenarioConfig(name="temporal_before")
    )
    _seed_observation(
        runtime,
        obs_id="obs-a",
        native_id="evt-a",
        start=T0,
        end=T0 + timedelta(minutes=2),
    )
    _seed_observation(
        runtime,
        obs_id="obs-b",
        native_id="evt-b",
        start=T0 + timedelta(minutes=5),
        end=T0 + timedelta(minutes=7),
    )
    before_starts = {
        obs_id: runtime.store.observations[obs_id].start_utc
        for obs_id in ("obs-a", "obs-b")
    }
    relation_ids = evaluate_temporal(runtime)
    assert relation_ids == ["rel-obs-a-obs-b-before"]
    relation = runtime.store.relations[relation_ids[0]]
    assert relation.evaluation_status is TemporalEvaluationStatus.ESTABLISHED
    event = runtime.trace.of_kind("temporal_eval")[0]
    assert event["holds"] is True
    assert event["materialized"] is True
    # Source observations must not gain fabricated temporal fields.
    assert runtime.store.observations["obs-a"].start_utc == before_starts["obs-a"]
    assert runtime.store.observations["obs-b"].start_utc == before_starts["obs-b"]
    assert runtime.store.observations["obs-a"].end_utc == T0 + timedelta(minutes=2)


def test_temporal_overlap_does_not_create_before():
    runtime = InvestigationRuntime.from_fixture(
        case=make_case(), scenario=ScenarioConfig(name="temporal_overlap")
    )
    _seed_observation(
        runtime,
        obs_id="obs-a",
        native_id="evt-a",
        start=T0,
        end=T0 + timedelta(minutes=10),
    )
    _seed_observation(
        runtime,
        obs_id="obs-b",
        native_id="evt-b",
        start=T0 + timedelta(minutes=5),
        end=T0 + timedelta(minutes=15),
    )
    assert evaluate_temporal(runtime) == []
    assert runtime.store.relations == {}
    event = runtime.trace.of_kind("temporal_eval")[0]
    assert event["holds"] is False
    assert event["materialized"] is False
    assert "before predicate does not hold" in event["reason"]


def test_temporal_reversed_order_does_not_create_before():
    runtime = InvestigationRuntime.from_fixture(
        case=make_case(), scenario=ScenarioConfig(name="temporal_reversed")
    )
    # Registration order is later-first; BEFORE(left,right) must be false.
    _seed_observation(
        runtime,
        obs_id="obs-later",
        native_id="evt-later",
        start=T0 + timedelta(minutes=20),
        end=T0 + timedelta(minutes=22),
    )
    _seed_observation(
        runtime,
        obs_id="obs-earlier",
        native_id="evt-earlier",
        start=T0,
        end=T0 + timedelta(minutes=2),
    )
    assert evaluate_temporal(runtime) == []
    assert runtime.store.relations == {}
    event = runtime.trace.of_kind("temporal_eval")[0]
    assert event["holds"] is False
    assert event["materialized"] is False


def test_temporal_insufficient_time_not_established_and_not_false():
    runtime = InvestigationRuntime.from_fixture(
        case=make_case(), scenario=ScenarioConfig(name="temporal_unknown")
    )
    _seed_observation(
        runtime,
        obs_id="obs-closed",
        native_id="evt-closed",
        start=T0,
        end=T0 + timedelta(minutes=2),
    )
    _seed_observation(
        runtime,
        obs_id="obs-open",
        native_id="evt-open",
        start=T0 + timedelta(minutes=5),
        end=None,
    )
    relation_ids = evaluate_temporal(runtime)
    assert relation_ids == ["rel-obs-closed-obs-open-before"]
    relation = runtime.store.relations[relation_ids[0]]
    assert relation.evaluation_status is TemporalEvaluationStatus.INDETERMINATE
    event = runtime.trace.of_kind("temporal_eval")[0]
    assert event["holds"] is None
    assert event["materialized"] is True
    assert event["evaluation_status"] == "indeterminate"
    # False remains distinguishable: this path is unknown, not holds=False.
    assert event["holds"] is not False


def test_execution_trace_multi_step_investigation():
    """Actual executed graph trace for the completion report."""
    scenario = ScenarioConfig(
        name="trace_demo",
        include_follow_up=True,
        max_tool_attempts=3,
        tool_scripts={
            "search_events": [_person_ok("exec-trace-person")],
            "search_related_events": [_vehicle_ok("exec-trace-vehicle")],
        },
    )
    result = _run(scenario)
    runtime = result["runtime"]
    trace = runtime.trace

    print("\n=== G4-V1-D2 EXECUTION TRACE ===")
    print("nodes:", trace.node_sequence())
    print("proposals:", trace.of_kind("tool_proposal"))
    print("guard_accepted:", trace.of_kind("guard_accepted"))
    print("guard_rejected:", trace.of_kind("guard_rejected"))
    print("budget:", trace.of_kind("budget_reservation"))
    print("executed:", trace.of_kind("tool_executed"))
    print("evidence:", trace.of_kind("evidence_registered"))
    print("temporal:", trace.of_kind("temporal_eval"))
    print("sufficiency:", trace.of_kind("sufficiency"))
    print("claims:", trace.of_kind("claim_assessment"))
    print("output_gate:", trace.of_kind("output_gate"))
    print("case_status:", runtime.case.status.value)
    print("report_id:", result["state"].get("report_id"))
    print("=== END TRACE ===\n")

    assert "continue_search" in [s["route"] for s in trace.of_kind("sufficiency")]
    temporal_events = [e for e in trace.of_kind("temporal_eval") if e.get("materialized")]
    assert temporal_events
    assert temporal_events[-1]["holds"] is True
    assert temporal_events[-1]["evaluation_status"] == "established"
    assert runtime.case.status in {CaseStatus.COMPLETED, CaseStatus.PARTIAL}
    assert result["report"] is not None
