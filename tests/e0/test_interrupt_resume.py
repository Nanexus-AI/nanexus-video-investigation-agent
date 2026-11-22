"""G4-V1-F: native LangGraph clarification interrupt / resume."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from nanexus_investigation.contracts.enums import CaseStatus
from nanexus_investigation.contracts.identity import Interval
from nanexus_investigation.graph.clarification import authorized_time_envelope
from nanexus_investigation.graph.runtime import (
    ScenarioConfig,
    make_case,
    ok_event_result,
)
from nanexus_investigation.graph.workflow import (
    InvestigationRuntime,
    build_investigation_graph,
    initial_state,
    run_investigation,
)

UTC = timezone.utc
T0 = datetime(2026, 11, 19, 20, 0, tzinfo=UTC)
T1 = datetime(2026, 11, 19, 23, 0, tzinfo=UTC)
T_NARROW_END = datetime(2026, 11, 19, 21, 0, tzinfo=UTC)
T_LATE_START = datetime(2026, 11, 19, 21, 0, tzinfo=UTC)


def _person(execution_id: str = "exec-f"):
    return ok_event_result(
        execution_id=execution_id,
        native_id=f"evt-{execution_id}",
        label="person",
        start=T0 + timedelta(minutes=10),
        end=T0 + timedelta(minutes=12),
    )


def _clarification_runtime(*, case_id: str, unresolved_end: bool = False):
    case = make_case(case_id=case_id, start=T0, end=T1)
    if unresolved_end:
        open_bounds = Interval(start_utc=T0, end_utc=None)
        case = case.model_copy(
            update={
                "requested_scope": case.requested_scope.model_copy(
                    update={"time_bounds": open_bounds}
                ),
                "effective_scope": case.effective_scope.model_copy(
                    update={"time_bounds": open_bounds}
                ),
            }
        )
    scenario = ScenarioConfig(
        name="interrupt_time",
        require_time_clarification=not unresolved_end,
        tool_scripts={"search_events": [_person(case_id)]},
    )
    return InvestigationRuntime.from_fixture(case=case, scenario=scenario)


def test_f01_unresolved_time_triggers_native_interrupt():
    runtime = _clarification_runtime(case_id="case-f01", unresolved_end=True)
    saver = InMemorySaver()
    app = build_investigation_graph(runtime, checkpointer=saver)
    thread_id = "thread-f01"
    config = {"configurable": {"thread_id": thread_id}}
    result = app.invoke(initial_state(runtime), config=config)

    assert "__interrupt__" in result
    interrupts = result["__interrupt__"]
    assert interrupts
    payload = interrupts[0].value
    assert payload["case_id"] == "case-f01"
    assert payload["clarification_id"] == "time_bounds"
    assert "start_utc" in payload["required_fields"]

    state = app.get_state(config)
    assert state.next == ("validate_intake",)
    assert runtime.case.status is CaseStatus.AWAITING_INPUT
    assert runtime.registry.invocations == []
    assert runtime.budget.reserved == 0
    assert runtime.budget.consumed == 0
    assert runtime.store.report is None
    assert not runtime.store.evidence
    assert any(e["kind"] == "clarification_interrupt" for e in runtime.trace.events)


def test_f02_valid_resume_continues_same_thread():
    runtime = _clarification_runtime(case_id="case-f02")
    saver = InMemorySaver()
    app = build_investigation_graph(runtime, checkpointer=saver)
    thread_id = "thread-f02"
    config = {"configurable": {"thread_id": thread_id}}
    paused = app.invoke(initial_state(runtime), config=config)
    assert "__interrupt__" in paused
    assert runtime.registry.invocations == []

    resume_payload = {
        "start_utc": T0.isoformat(),
        "end_utc": T_NARROW_END.isoformat(),
    }
    final = app.invoke(Command(resume=resume_payload), config=config)
    assert "__interrupt__" not in final
    assert final.get("case_id") == "case-f02"
    assert final.get("time_clarification_applied") is True
    assert runtime.case.status in {CaseStatus.COMPLETED, CaseStatus.PARTIAL}
    assert runtime.case.effective_scope.time_bounds.end_utc == T_NARROW_END
    assert runtime.registry.invocations == ["search_events"]
    assert runtime.budget.reserved == 1
    assert runtime.store.report is not None
    assert runtime.last_gate is not None and runtime.last_gate.accepted

    # Same thread preserved; no duplicate tool invocation after completion.
    finished = app.get_state(config)
    assert finished.next == ()
    assert runtime.registry.invocations == ["search_events"]


def test_f03_unauthorized_resume_rejected_without_tools():
    runtime = _clarification_runtime(case_id="case-f03")
    saver = InMemorySaver()
    app = build_investigation_graph(runtime, checkpointer=saver)
    thread_id = "thread-f03"
    config = {"configurable": {"thread_id": thread_id}}
    paused = app.invoke(initial_state(runtime), config=config)
    assert "__interrupt__" in paused
    original_end = runtime.case.effective_scope.time_bounds.end_utc
    original_cameras = list(runtime.case.effective_scope.camera_ids)

    # Expand time beyond authorized envelope and request unauthorized camera.
    bad = app.invoke(
        Command(
            resume={
                "start_utc": T0.isoformat(),
                "end_utc": (T1 + timedelta(hours=5)).isoformat(),
                "camera_ids": ["back_alley"],
                "vms_instance_id": "vms-other",
            }
        ),
        config=config,
    )
    # Remain interruptible on the same thread; do not strand as terminated.
    assert "__interrupt__" in bad
    assert bad["__interrupt__"][0].value.get("last_rejection")
    assert app.get_state(config).next == ("validate_intake",)
    assert runtime.registry.invocations == []
    assert runtime.budget.reserved == 0
    assert runtime.store.report is None
    # Scope not widened to unauthorized values.
    assert "back_alley" not in runtime.case.effective_scope.camera_ids
    assert runtime.case.effective_scope.camera_ids == tuple(original_cameras)
    assert runtime.case.effective_scope.vms_instance_id == "vms-local-1"
    assert runtime.case.effective_scope.time_bounds.end_utc == original_end
    assert runtime.case.status is CaseStatus.AWAITING_INPUT
    assert any(e["kind"] == "clarification_rejected" for e in runtime.trace.events)


def test_f04_complete_request_skips_interrupt_and_existing_paths_ok():
    # Complete request: no clarification interrupt.
    runtime = InvestigationRuntime.from_fixture(
        case=make_case(case_id="case-f04"),
        scenario=ScenarioConfig(
            name="f04_complete",
            require_time_clarification=False,
            tool_scripts={"search_events": [_person("f04")]},
        ),
    )
    result = run_investigation(runtime, thread_id="thread-f04")
    assert result["state"].get("__interrupt__") is None
    assert result["report"] is not None
    assert runtime.registry.invocations == ["search_events"]
    assert not any(e["kind"] == "clarification_interrupt" for e in runtime.trace.events)

    # Inverted interval rejected; thread remains awaiting corrected clarification.
    runtime2 = _clarification_runtime(case_id="case-f04-bad-interval")
    app = build_investigation_graph(runtime2, checkpointer=InMemorySaver())
    config = {"configurable": {"thread_id": "thread-f04-bad"}}
    app.invoke(initial_state(runtime2), config=config)
    rejected = app.invoke(
        Command(
            resume={
                "start_utc": T1.isoformat(),
                "end_utc": T0.isoformat(),
            }
        ),
        config=config,
    )
    assert "__interrupt__" in rejected
    assert runtime2.registry.invocations == []
    assert runtime2.case.status is CaseStatus.AWAITING_INPUT
    assert any(e["kind"] == "clarification_rejected" for e in runtime2.trace.events)


def test_f05_invalid_then_valid_clarification_same_thread():
    runtime = _clarification_runtime(case_id="case-f05")
    saver = InMemorySaver()
    app = build_investigation_graph(runtime, checkpointer=saver)
    config = {"configurable": {"thread_id": "thread-f05"}}
    paused = app.invoke(initial_state(runtime), config=config)
    assert "__interrupt__" in paused

    rejected = app.invoke(
        Command(
            resume={
                "start_utc": T0.isoformat(),
                "end_utc": (T1 + timedelta(hours=2)).isoformat(),
                "camera_ids": ["back_alley"],
            }
        ),
        config=config,
    )
    assert "__interrupt__" in rejected
    assert runtime.registry.invocations == []
    assert runtime.budget.reserved == 0
    assert runtime.case.status is CaseStatus.AWAITING_INPUT
    assert "back_alley" not in runtime.case.effective_scope.camera_ids

    final = app.invoke(
        Command(
            resume={
                "start_utc": T0.isoformat(),
                "end_utc": T_NARROW_END.isoformat(),
            }
        ),
        config=config,
    )
    assert "__interrupt__" not in final
    assert final.get("time_clarification_applied") is True
    assert runtime.case.status in {CaseStatus.COMPLETED, CaseStatus.PARTIAL}
    assert runtime.case.effective_scope.time_bounds.end_utc == T_NARROW_END
    assert runtime.registry.invocations == ["search_events"]
    assert runtime.budget.reserved == 1
    assert runtime.store.report is not None
    assert runtime.last_gate is not None and runtime.last_gate.accepted


def test_f02_envelope_does_not_widen_narrower_effective_start():
    case = make_case(case_id="case-env", start=T0, end=T1)
    open_late = Interval(start_utc=T_LATE_START, end_utc=None)
    case = case.model_copy(
        update={
            "effective_scope": case.effective_scope.model_copy(
                update={"time_bounds": open_late}
            ),
            # Requested/time_context remain the wider [T0, T1) from make_case.
        }
    )
    runtime = InvestigationRuntime.from_fixture(
        case=case,
        scenario=ScenarioConfig(name="envelope", require_time_clarification=True),
    )
    envelope = authorized_time_envelope(runtime)
    assert envelope is not None
    assert envelope.start_utc == T_LATE_START
    assert envelope.end_utc == T1
