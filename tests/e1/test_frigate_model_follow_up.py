"""G4-E1-C: local-model follow-up over Frigate search_events (no related-events).

Deterministic StaticModelProvider + recorded Frigate always run.
Genuine Ollama × Frigate recorded/live tests require reachable Ollama and fail
hard when unavailable — they never skip as PASS.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tests.e1.live_scope import require_live_cameras, require_live_time_window

from nanexus_investigation.contracts.enums import CaseStatus, SearchCoverage
from nanexus_investigation.contracts.identity import Interval
from nanexus_investigation.graph.runtime import ScenarioConfig, make_case
from nanexus_investigation.graph.workflow import InvestigationRuntime, run_investigation
from nanexus_investigation.model.provider import (
    InvestigationProposal,
    OllamaChatProvider,
    StaticModelProvider,
    build_follow_up_prompt,
)
from nanexus_investigation.model.provider import ProposalContext
from nanexus_investigation.persistence.db import InvestigationDatabase
from nanexus_investigation.persistence.repository import InvestigationRepository
from nanexus_investigation.tools.frigate_backend import FrigateToolRegistry

FIXTURES = Path(__file__).parent / "fixtures" / "frigate"
PRIVATE_RECORDING = Path(".local/frigate_recordings/events_list_live.json")
UTC = timezone.utc
VMS = "frigate-fixture-0.17.1"
OLLAMA_URL = "http://127.0.0.1:11434"
MODEL_NAME = "qwen3:14b"

QUESTION = (
    "Investigate person detections in the authorized time window and determine "
    "whether vehicle detections were recorded near those times. "
    "Do not invent physical interaction from timestamps alone."
)


def _require_local_model() -> OllamaChatProvider:
    try:
        with urllib.request.urlopen(f"{OLLAMA_URL}/api/tags", timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        pytest.fail(f"E1-C requires reachable local Ollama at {OLLAMA_URL}: {exc}")
    names = {m.get("name") for m in payload.get("models", []) if isinstance(m, dict)}
    if MODEL_NAME not in names:
        pytest.fail(
            f"E1-C requires installed model {MODEL_NAME!r}; installed={sorted(names)}"
        )
    return OllamaChatProvider(model=MODEL_NAME, base_url=OLLAMA_URL, timeout_s=180.0)


def _fixture_window_case(*, case_id: str, vms: str = "frigate-fixture"):
    return make_case(
        case_id=case_id,
        question=QUESTION,
        cameras=("front_yard",),
        start=datetime(2023, 11, 14, 22, 0, tzinfo=UTC),
        end=datetime(2023, 11, 14, 23, 30, tzinfo=UTC),
        vms_instance_id=vms,
    )


def test_e1_c_prompt_does_not_advertise_unsupported_related_events():
    ctx = ProposalContext(
        question=QUESTION,
        case_id="case-prompt",
        authorized_tools=("search_events",),
        authorized_cameras=("front_yard",),
        vms_instance_id=VMS,
        time_bounds={"start_utc": "t0", "end_utc": "t1"},
        prior_tool_name="search_events",
        prior_tool_status="OK",
        prior_match_count=2,
        prior_result_summary="2 registered observation(s); labels=['person']",
    )
    _system, user = build_follow_up_prompt(ctx)
    assert "search_events" in user
    assert "Do not invent search_related_events" in user
    assert "typically search_related_events" not in user


def test_e1_c_static_follow_up_second_search_events_recorded():
    """Pipeline proof: StaticModel selects second search_events over Frigate."""
    case = _fixture_window_case(case_id="case-e1-c-static-follow")
    provider = StaticModelProvider(
        proposal=InvestigationProposal(
            action="invoke_tool",
            tool_name="search_events",
            label="car",
            rationale="Person detections found; query vehicle/car detections next.",
            camera_ids=("front_yard",),
        )
    )
    scenario = ScenarioConfig(
        name="e1_c_static_follow",
        tool_backend="frigate_recorded",
        frigate_recording_path=str(FIXTURES / "events_list_sanitized.json"),
        search_label="person",
        include_follow_up=False,
        use_model_follow_up=True,
        enable_temporal_analysis=False,
        enable_media_resolve=False,
        max_tool_attempts=3,
        max_plan_steps=4,
    )
    runtime = InvestigationRuntime.from_fixture(
        case=case, scenario=scenario, model_provider=provider
    )
    result = run_investigation(runtime, thread_id="thread-e1-c-static-follow")

    assert isinstance(runtime.registry, FrigateToolRegistry)
    assert runtime.registry.invocations == ["search_events", "search_events"]
    assert runtime.last_model_proposal is not None
    assert runtime.last_model_proposal.action == "invoke_tool"
    assert runtime.last_model_proposal.tool_name == "search_events"
    assert runtime.last_model_proposal.label == "car"
    assert "search_related_events" not in runtime.registry.invocations
    assert provider.calls
    assert provider.calls[0].authorized_tools == ("search_events",)
    assert len(runtime.store.receipts) == 2
    # First person + follow-up car from sanitized fixture.
    labels = sorted(
        {
            label
            for obs in runtime.store.observations.values()
            for label in obs.labels
        }
    )
    assert "person" in labels
    assert "car" in labels
    assert runtime.last_gate is not None and runtime.last_gate.accepted
    assert runtime.case.status in {CaseStatus.COMPLETED, CaseStatus.PARTIAL}
    assert result["state"].get("model_proposal", {}).get("label") == "car"


def test_e1_c_static_proceed_when_empty_recorded():
    """Contrasting control: EMPTY initial search → model proceeds to report."""
    case = _fixture_window_case(case_id="case-e1-c-static-empty")
    provider = StaticModelProvider(
        proposal=InvestigationProposal(
            action="proceed_to_report",
            tool_name=None,
            label=None,
            rationale="No person detections; further vehicle search not warranted.",
        )
    )
    scenario = ScenarioConfig(
        name="e1_c_static_empty",
        tool_backend="frigate_recorded",
        frigate_recording_path=str(FIXTURES / "events_empty.json"),
        search_label="person",
        include_follow_up=False,
        use_model_follow_up=True,
        enable_temporal_analysis=False,
        enable_media_resolve=False,
        max_tool_attempts=3,
    )
    runtime = InvestigationRuntime.from_fixture(
        case=case, scenario=scenario, model_provider=provider
    )
    run_investigation(runtime, thread_id="thread-e1-c-static-empty")
    assert runtime.registry.invocations == ["search_events"]
    assert runtime.last_model_proposal is not None
    assert runtime.last_model_proposal.action == "proceed_to_report"
    assert len(runtime.store.observations) == 0
    receipt = next(iter(runtime.store.receipts.values()))
    tool = runtime.store.tool_results_by_receipt[receipt.search_execution_id]
    assert tool.status.value == "EMPTY"
    assert runtime.last_gate is not None


def test_e1_c_static_rejects_related_events_without_stub_fallback():
    case = _fixture_window_case(case_id="case-e1-c-reject-related")
    provider = StaticModelProvider(
        proposal=InvestigationProposal(
            action="invoke_tool",
            tool_name="search_related_events",
            label="vehicle",
            rationale="attempt unsupported related search",
        )
    )
    scenario = ScenarioConfig(
        name="e1_c_reject_related",
        tool_backend="frigate_recorded",
        frigate_recording_path=str(FIXTURES / "events_list_sanitized.json"),
        search_label="person",
        use_model_follow_up=True,
        enable_temporal_analysis=False,
        enable_media_resolve=False,
        max_tool_attempts=3,
    )
    runtime = InvestigationRuntime.from_fixture(
        case=case, scenario=scenario, model_provider=provider
    )
    run_investigation(runtime, thread_id="thread-e1-c-reject-related")
    assert runtime.registry.invocations == ["search_events"]
    assert runtime.last_model_proposal is not None
    assert runtime.last_model_proposal.tool_name == "search_related_events"
    assert runtime.case.status is CaseStatus.FAILED
    unsupported_events = runtime.trace.of_kind("model_proposal_unsupported_tool")
    assert unsupported_events, runtime.trace.events[-5:]
    assert unsupported_events[0]["tool_name"] == "search_related_events"


def test_e1_c_static_durable_multi_receipt_integrity(tmp_path: Path):
    domain_db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    case = _fixture_window_case(case_id="case-e1-c-durable")
    provider = StaticModelProvider(
        proposal=InvestigationProposal(
            action="invoke_tool",
            tool_name="search_events",
            label="car",
            rationale="follow up for car detections",
        )
    )
    scenario = ScenarioConfig(
        name="e1_c_durable",
        tool_backend="frigate_recorded",
        frigate_recording_path=str(FIXTURES / "events_list_sanitized.json"),
        search_label="person",
        use_model_follow_up=True,
        enable_temporal_analysis=False,
        enable_media_resolve=False,
        max_tool_attempts=3,
    )
    runtime = InvestigationRuntime.from_durable_fixture(
        case=case,
        scenario=scenario,
        domain_db=domain_db,
        model_provider=provider,
    )
    run_investigation(runtime, thread_id="thread-e1-c-durable")
    assert runtime.registry.invocations == ["search_events", "search_events"]
    sql_obs = domain_db.conn.execute(
        "SELECT COUNT(*) AS c FROM nx_artifacts WHERE artifact_kind='observation'"
    ).fetchone()["c"]
    sql_ev = domain_db.conn.execute(
        "SELECT COUNT(*) AS c FROM nx_artifacts WHERE artifact_kind='evidence_ref'"
    ).fetchone()["c"]
    sql_receipts = domain_db.conn.execute(
        "SELECT COUNT(*) AS c FROM nx_artifacts WHERE artifact_kind='search_receipt'"
    ).fetchone()["c"]
    assert sql_obs == len(runtime.store.observations) == 2
    assert sql_ev == len(runtime.store.evidence) == 2
    assert sql_receipts == 2
    assert runtime.store.report is not None
    repo = InvestigationRepository(domain_db)
    for evidence_id in runtime.store.report.evidence_index:
        assert repo.get_evidence(evidence_id) is not None
    domain_db.close()


def test_e1_c_real_ollama_frigate_recorded_follow_up_or_report():
    """Genuine Ollama over recorded Frigate person hits; capture actual proposal."""
    provider = _require_local_model()
    recording = (
        PRIVATE_RECORDING
        if PRIVATE_RECORDING.is_file()
        else FIXTURES / "events_list_sanitized.json"
    )
    if recording == PRIVATE_RECORDING:
        case = make_case(
            case_id="case-e1-c-ollama-person",
            question=QUESTION,
            cameras=require_live_cameras()[:1],
            start=require_live_time_window()[0].astimezone(UTC),
            end=require_live_time_window()[1].astimezone(UTC),
            vms_instance_id=VMS,
        )
        force_bounds = Interval(
            start_utc=require_live_time_window()[0].astimezone(UTC),
            end_utc=require_live_time_window()[1].astimezone(UTC),
        )
        vms = VMS
    else:
        case = _fixture_window_case(case_id="case-e1-c-ollama-person")
        force_bounds = None
        vms = "frigate-fixture"

    scenario = ScenarioConfig(
        name="e1_c_ollama_person",
        tool_backend="frigate_recorded",
        frigate_recording_path=str(recording),
        search_label="person",
        include_follow_up=False,
        use_model_follow_up=True,
        enable_temporal_analysis=False,
        enable_media_resolve=False,
        max_tool_attempts=3,
        max_plan_steps=4,
        force_time_bounds=force_bounds,
    )
    runtime = InvestigationRuntime.from_fixture(
        case=case, scenario=scenario, model_provider=provider
    )
    run_investigation(runtime, thread_id="thread-e1-c-ollama-person")

    proposal = runtime.last_model_proposal
    assert proposal is not None
    assert isinstance(proposal, InvestigationProposal)
    assert proposal.raw_text
    assert proposal.rationale
    assert runtime.registry.invocations[0] == "search_events"
    assert "search_related_events" not in runtime.registry.invocations

    follow_up = (
        proposal.action == "invoke_tool"
        and proposal.tool_name == "search_events"
        and len(runtime.registry.invocations) >= 2
    )
    proceeded = proposal.action == "proceed_to_report"

    print(
        json.dumps(
            {
                "scenario": "person_hits",
                "vms": vms,
                "recording": str(recording),
                "initial_invocations_prefix": runtime.registry.invocations[:1],
                "all_invocations": list(runtime.registry.invocations),
                "proposal": proposal.to_dict(),
                "follow_up_executed": follow_up,
                "proceeded": proceeded,
                "observations": len(runtime.store.observations),
                "evidence": len(runtime.store.evidence),
                "receipts": len(runtime.store.receipts),
                "gate_accepted": None
                if runtime.last_gate is None
                else runtime.last_gate.accepted,
                "case_status": runtime.case.status.value,
            }
        )
    )
    # OllamaChatProvider does not record calls; assert structural outcomes.
    assert follow_up or proceeded, (
        f"unexpected proposal action={proposal.action!r} tool={proposal.tool_name!r}"
    )
    if follow_up:
        assert runtime.last_guard is not None and runtime.last_guard.accepted
        assert len(runtime.store.receipts) >= 2
        assert runtime.store.report is not None
        assert runtime.last_gate is not None and runtime.last_gate.accepted
    else:
        # Model chose stop — still a valid guarded outcome; note for report.
        assert runtime.registry.invocations == ["search_events"]


def test_e1_c_real_ollama_empty_contrast_recorded():
    """Same question; EMPTY Frigate-derived first search (explicit empty fixture)."""
    provider = _require_local_model()
    case = _fixture_window_case(case_id="case-e1-c-ollama-empty")
    scenario = ScenarioConfig(
        name="e1_c_ollama_empty",
        tool_backend="frigate_recorded",
        frigate_recording_path=str(FIXTURES / "events_empty.json"),
        search_label="person",
        use_model_follow_up=True,
        enable_temporal_analysis=False,
        enable_media_resolve=False,
        max_tool_attempts=3,
    )
    runtime = InvestigationRuntime.from_fixture(
        case=case, scenario=scenario, model_provider=provider
    )
    run_investigation(runtime, thread_id="thread-e1-c-ollama-empty")
    proposal = runtime.last_model_proposal
    assert proposal is not None
    assert proposal.raw_text
    assert "search_related_events" not in runtime.registry.invocations
    print(
        json.dumps(
            {
                "scenario": "empty_control",
                "invocations": list(runtime.registry.invocations),
                "proposal": proposal.to_dict(),
                "observations": len(runtime.store.observations),
                "coverage": next(
                    iter(runtime.store.receipts.values())
                ).coverage.value
                if runtime.store.receipts
                else None,
            }
        )
    )
    # Prefer stop after EMPTY; if model still invokes search_events, require guard OK.
    if proposal.action == "invoke_tool":
        assert proposal.tool_name == "search_events"
        assert runtime.last_guard is not None and runtime.last_guard.accepted
    else:
        assert proposal.action == "proceed_to_report"
        assert runtime.registry.invocations == ["search_events"]


@pytest.mark.live_frigate
def test_e1_c_real_ollama_live_frigate_follow_up(tmp_path: Path):
    """Live Frigate + real Ollama; report actual proposal and durable counts."""
    from nanexus_investigation.graph.runtime import _client_config_for_case
    from nanexus_investigation.vms.frigate.client import FrigateReadOnlyClient
    from nanexus_investigation.vms.frigate.record import (
        config_from_env,
        live_client_ready,
        load_frigate_env,
    )

    provider = _require_local_model()
    env = load_frigate_env()
    ready, reason = live_client_ready(env)
    if not ready:
        pytest.fail(f"Live Frigate unavailable: {reason}")

    domain_db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    case = make_case(
        case_id="case-e1-c-live",
        question=QUESTION,
        cameras=require_live_cameras()[:1],
        start=require_live_time_window()[0].astimezone(UTC),
        end=require_live_time_window()[1].astimezone(UTC),
        vms_instance_id=VMS,
    )
    scenario = ScenarioConfig(
        name="e1_c_live",
        tool_backend="frigate_live",
        search_label="person",
        include_follow_up=False,
        use_model_follow_up=True,
        enable_temporal_analysis=False,
        enable_media_resolve=False,
        max_tool_attempts=3,
        max_plan_steps=4,
        frigate_max_limit=50,
        force_time_bounds=Interval(
            start_utc=require_live_time_window()[0].astimezone(UTC),
            end_utc=require_live_time_window()[1].astimezone(UTC),
        ),
    )
    cfg = _client_config_for_case(config_from_env(env), case, scenario)
    client = FrigateReadOnlyClient(cfg)
    runtime = InvestigationRuntime.from_durable_fixture(
        case=case,
        scenario=scenario,
        domain_db=domain_db,
        frigate_client=client,
        model_provider=provider,
    )
    run_investigation(runtime, thread_id="thread-e1-c-live")

    proposal = runtime.last_model_proposal
    assert proposal is not None
    assert isinstance(runtime.registry, FrigateToolRegistry)
    assert runtime.registry.mode == "live"
    assert "search_related_events" not in runtime.registry.invocations

    sql_obs = domain_db.conn.execute(
        "SELECT COUNT(*) AS c FROM nx_artifacts WHERE artifact_kind='observation'"
    ).fetchone()["c"]
    sql_ev = domain_db.conn.execute(
        "SELECT COUNT(*) AS c FROM nx_artifacts WHERE artifact_kind='evidence_ref'"
    ).fetchone()["c"]
    sql_receipts = domain_db.conn.execute(
        "SELECT COUNT(*) AS c FROM nx_artifacts WHERE artifact_kind='search_receipt'"
    ).fetchone()["c"]

    follow_up = (
        proposal.action == "invoke_tool"
        and proposal.tool_name == "search_events"
        and len(runtime.registry.invocations) >= 2
    )
    print(
        json.dumps(
            {
                "scenario": "live_person_then_model",
                "invocations": list(runtime.registry.invocations),
                "proposal": proposal.to_dict(),
                "follow_up_executed": follow_up,
                "memory_obs": len(runtime.store.observations),
                "memory_ev": len(runtime.store.evidence),
                "memory_receipts": len(runtime.store.receipts),
                "sqlite_obs": sql_obs,
                "sqlite_ev": sql_ev,
                "sqlite_receipts": sql_receipts,
                "gate": None
                if runtime.last_gate is None
                else runtime.last_gate.case_status.value,
                "case_status": runtime.case.status.value,
                "limitations": list(runtime.store.report.limitations)
                if runtime.store.report
                else [],
            }
        )
    )
    assert sql_obs == len(runtime.store.observations)
    assert sql_ev == len(runtime.store.evidence)
    assert sql_receipts == len(runtime.store.receipts)
    if runtime.store.report is not None:
        repo = InvestigationRepository(domain_db)
        for evidence_id in runtime.store.report.evidence_index:
            assert repo.get_evidence(evidence_id) is not None
    domain_db.close()
