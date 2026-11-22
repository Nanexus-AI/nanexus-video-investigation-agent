"""V0.1 T5: local visual verification over retained media.

Deterministic StaticVisualProvider tests always run. Real moondream inference
is opt-in via reachable Ollama with moondream installed.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from nanexus_investigation.contracts.enums import (
    ClaimType,
    ClaimValidationStatus,
    EvidenceAvailability,
    EvidenceKind,
    RecordType,
    SearchCoverage,
    ToolErrorCode,
    ToolResultStatus,
)
from nanexus_investigation.contracts.evidence import EvidenceRef
from nanexus_investigation.contracts.identity import Interval, SourceLocator
from nanexus_investigation.contracts.scope import InvestigationScope
from nanexus_investigation.contracts.tool import ToolRequest
from nanexus_investigation.evidence.media_store import LocalMediaStore
from nanexus_investigation.graph.runtime import (
    ScenarioConfig,
    make_case,
    ok_event_result,
)
from nanexus_investigation.graph.workflow import InvestigationRuntime, run_investigation
from nanexus_investigation.model.provider import (
    InvestigationProposal,
    StaticModelProvider,
)
from nanexus_investigation.model.visual_provider import (
    OllamaVisualProvider,
    StaticVisualProvider,
)
from nanexus_investigation.persistence.db import InvestigationDatabase
from nanexus_investigation.persistence.repository import InvestigationRepository
from nanexus_investigation.tools.media_observe import observe_media
from nanexus_investigation.tools.media_resolve import resolve_media
from nanexus_investigation.vms.frigate.client import FrigateHttpResponse

UTC = timezone.utc
PRIVATE_MEDIA = Path(".local/frigate_recordings/media")
PERSON_Q = "Find person detections in this camera and time window."
VISUAL_Q = (
    "Retrieve available media for a driveway detection and examine the image "
    "for relevant visible content. Do not invent identity."
)


class _FakeTransport:
    def __init__(self, *, snapshot: bytes) -> None:
        self.snapshot = snapshot

    def get_event_snapshot(self, event_id: str, *, max_bytes: int = 0):
        return FrigateHttpResponse(
            status_code=200,
            path=f"/api/events/{event_id}/snapshot.jpg",
            query={},
            body=self.snapshot,
            headers={"Content-Type": "image/jpeg"},
            retrieved_at=datetime.now(tz=UTC),
        )

    def get_event_clip(self, event_id: str, *, max_bytes: int = 0):
        raise AssertionError("clip not used in snapshot observe tests")


def _scope() -> InvestigationScope:
    return InvestigationScope(
        vms_instance_id="vms-1",
        camera_ids=("front_yard",),
        time_bounds=Interval(
            start_utc=datetime(2023, 11, 14, 22, 0, tzinfo=UTC),
            end_utc=datetime(2023, 11, 14, 23, 30, tzinfo=UTC),
        ),
    )


def _media_evidence(
    tmp_path: Path, *, payload: bytes, case_id: str = "case-t5"
) -> tuple[EvidenceRef, LocalMediaStore]:
    store = LocalMediaStore(tmp_path / "media")
    digest = hashlib.sha256(payload).hexdigest()
    retained_id = f"media-{digest[:24]}"
    store.put(
        case_id=case_id,
        retained_media_id=retained_id,
        media_kind="snapshot",
        content_type="image/jpeg",
        payload=payload,
    )
    t0 = datetime(2023, 11, 14, 22, 15, tzinfo=UTC)
    evidence = EvidenceRef(
        evidence_id="ev-media-src-1",
        source_locator=SourceLocator(
            vms_instance_id="vms-1",
            record_type=RecordType.EVENT,
            native_record_id="evt-t5-1",
        ),
        evidence_kind=EvidenceKind.SNAPSHOT,
        camera_id="front_yard",
        interval=Interval(start_utc=t0, end_utc=t0 + timedelta(seconds=10)),
        retrieval_spec={
            "operation": "resolve_media",
            "media_kind": "snapshot",
            "retained_media_id": retained_id,
            "source_evidence_id": "ev-meta-1",
            "bytes_retained": True,
            "visual_interpretation": False,
        },
        availability=EvidenceAvailability.AVAILABLE,
        content_sha256=digest,
        content_type="image/jpeg",
        provenance="adapter:frigate-media",
    )
    return evidence, store


def test_t5_unit_observe_snapshot_static_provider(tmp_path: Path):
    payload = b"\xff\xd8\xff" + b"t5snap" * 40
    evidence, store = _media_evidence(tmp_path, payload=payload)
    provider = StaticVisualProvider(
        observation_text="The image appears to contain a possible vehicle."
    )
    result = observe_media(
        ToolRequest(
            tool_name="observe_media",
            arguments={
                "evidence_id": evidence.evidence_id,
                "objective": "Does the image appear to contain a vehicle?",
            },
            source_id="vms-1",
            camera_ids=("front_yard",),
        ),
        evidence_by_id={evidence.evidence_id: evidence},
        scope=_scope(),
        case_id="case-t5",
        media_store=store,
        visual_provider=provider,
    )
    assert result.status is ToolResultStatus.OK
    record = result.data[0]
    assert record["record_kind"] == "visual_observation"
    assert record["physical_world_fact"] is False
    assert record["content_sha256"] == hashlib.sha256(payload).hexdigest()
    assert record["source_content_sha256"] == record["content_sha256"]
    assert record["inference_input_sha256"]
    assert record["preprocessing"] in {"none", "ffmpeg_reencode_q5_max_edge"}
    assert "vehicle" in record["observation_text"].lower()


def test_t5_04_rejects_foreign_url_hash_mismatch_and_unavailable(tmp_path: Path):
    payload = b"\xff\xd8\xff" + b"bad" * 20
    evidence, store = _media_evidence(tmp_path, payload=payload)
    provider = StaticVisualProvider()

    banned = observe_media(
        ToolRequest(
            tool_name="observe_media",
            arguments={
                "evidence_id": evidence.evidence_id,
                "objective": "look",
                "url": "https://evil.example/x.jpg",
            },
            source_id="vms-1",
            camera_ids=("front_yard",),
        ),
        evidence_by_id={evidence.evidence_id: evidence},
        scope=_scope(),
        case_id="case-t5",
        media_store=store,
        visual_provider=provider,
    )
    assert banned.status is ToolResultStatus.ERROR
    assert "rejects model-supplied url" in banned.error.message

    foreign = observe_media(
        ToolRequest(
            tool_name="observe_media",
            arguments={"evidence_id": "ev-foreign", "objective": "look"},
            source_id="vms-1",
            camera_ids=("front_yard",),
        ),
        evidence_by_id={evidence.evidence_id: evidence},
        scope=_scope(),
        case_id="case-t5",
        media_store=store,
        visual_provider=provider,
    )
    assert foreign.status is ToolResultStatus.ERROR
    assert "unknown or foreign" in foreign.error.message

    # Hash mismatch on reopen.
    bin_path = tmp_path / "media" / "case-t5" / f"{evidence.retrieval_spec['retained_media_id']}.bin"
    bin_path.write_bytes(b"tampered-bytes")
    mismatch = observe_media(
        ToolRequest(
            tool_name="observe_media",
            arguments={
                "evidence_id": evidence.evidence_id,
                "objective": "look",
            },
            source_id="vms-1",
            camera_ids=("front_yard",),
        ),
        evidence_by_id={evidence.evidence_id: evidence},
        scope=_scope(),
        case_id="case-t5",
        media_store=store,
        visual_provider=provider,
    )
    assert mismatch.status in {ToolResultStatus.EMPTY, ToolResultStatus.ERROR}

    unavailable = observe_media(
        ToolRequest(
            tool_name="observe_media",
            arguments={
                "evidence_id": evidence.evidence_id,
                "objective": "look",
            },
            source_id="vms-1",
            camera_ids=("front_yard",),
        ),
        evidence_by_id={evidence.evidence_id: evidence},
        scope=_scope(),
        case_id="case-t5",
        media_store=store,
        visual_provider=StaticVisualProvider(available=False),
    )
    assert unavailable.status is ToolResultStatus.UNSUPPORTED
    assert unavailable.error.code is ToolErrorCode.DISABLED_CAPABILITY

    fresh_evidence, fresh_store = _media_evidence(
        tmp_path / "fresh", payload=payload, case_id="case-t5-empty"
    )
    empty = observe_media(
        ToolRequest(
            tool_name="observe_media",
            arguments={
                "evidence_id": fresh_evidence.evidence_id,
                "objective": "look",
            },
            source_id="vms-1",
            camera_ids=("front_yard",),
        ),
        evidence_by_id={fresh_evidence.evidence_id: fresh_evidence},
        scope=_scope(),
        case_id="case-t5-empty",
        media_store=fresh_store,
        visual_provider=StaticVisualProvider(empty_response=True),
    )
    assert empty.status is ToolResultStatus.ERROR
    assert empty.error.code is ToolErrorCode.INVALID_RESPONSE


def test_t5_01_simple_retrieval_skips_observe():
    t0 = datetime(2023, 11, 14, 22, 10, tzinfo=UTC)
    provider = StaticModelProvider(
        initial_proposal=InvestigationProposal(
            action="invoke_tool",
            tool_name="search_events",
            label="person",
            rationale="person only",
            camera_ids=("front_yard",),
        ),
        proposals=[
            InvestigationProposal(
                action="proceed_to_report",
                tool_name=None,
                label=None,
                rationale="retrieval complete",
            )
        ],
    )
    runtime = InvestigationRuntime.from_fixture(
        case=make_case(
            case_id="case-t5-01",
            question=PERSON_Q,
            start=datetime(2023, 11, 14, 22, 0, tzinfo=UTC),
            end=datetime(2023, 11, 14, 23, 30, tzinfo=UTC),
        ),
        scenario=ScenarioConfig(
            name="t5_01",
            tool_backend="stub",
            use_model_initial_plan=True,
            use_model_follow_up=True,
            max_model_calls=3,
            max_tool_attempts=3,
            enable_observe_media=True,
            tool_scripts={
                "search_events": [
                    ok_event_result(
                        execution_id="exec-t5-01",
                        native_id="evt-t5-01",
                        label="person",
                        start=t0,
                        end=t0 + timedelta(seconds=60),
                    )
                ]
            },
        ),
        model_provider=provider,
        visual_provider=StaticVisualProvider(),
    )
    run_investigation(runtime, thread_id="thread-t5-01")
    assert runtime.registry.invocations == ["search_events"]
    assert "observe_media" not in runtime.registry.invocations
    assert "resolve_media" not in runtime.registry.invocations
    assert runtime.last_gate is not None and runtime.last_gate.accepted


def test_t5_02_snapshot_visual_orchestration_and_durable_reopen(tmp_path: Path):
    payload = b"\xff\xd8\xff" + b"orch" * 80
    t0 = datetime(2023, 11, 14, 22, 15, tzinfo=UTC)
    person = ok_event_result(
        execution_id="exec-t5-02",
        native_id="evt-t5-02",
        label="car",
        start=t0,
        end=t0 + timedelta(seconds=15),
        camera_id="front_yard",
    )
    provider = StaticModelProvider(
        initial_proposal=InvestigationProposal(
            action="invoke_tool",
            tool_name="search_events",
            label="car",
            rationale="retrieve detection first",
            camera_ids=("front_yard",),
        ),
        proposals=[],
    )
    domain_db = InvestigationDatabase.open(tmp_path / "t5-02.sqlite")
    runtime = InvestigationRuntime.from_durable_fixture(
        case=make_case(
            case_id="case-t5-02",
            question=VISUAL_Q,
            cameras=("front_yard",),
            start=datetime(2023, 11, 14, 22, 0, tzinfo=UTC),
            end=datetime(2023, 11, 14, 23, 30, tzinfo=UTC),
            vms_instance_id="vms-local-1",
        ),
        scenario=ScenarioConfig(
            name="t5_02",
            tool_backend="stub",
            use_model_initial_plan=True,
            use_model_follow_up=True,
            max_model_calls=5,
            max_tool_attempts=5,
            enable_media_resolve=True,
            enable_observe_media=True,
            tool_scripts={"search_events": [person]},
            media_store_root=str(tmp_path / "media_root"),
        ),
        domain_db=domain_db,
        model_provider=provider,
        visual_provider=StaticVisualProvider(
            observation_text="The image appears to show a possible vehicle."
        ),
    )

    original_invoke = runtime.registry.invoke

    def _invoke(request: ToolRequest):
        if request.tool_name == "resolve_media":
            runtime.registry.invocations.append("resolve_media")
            return resolve_media(
                request,
                evidence_by_id=runtime.store.evidence,
                observations_by_id=runtime.store.observations,
                scope=runtime.case.effective_scope,
                case_id=runtime.case.case_id,
                media_store=runtime.media_store,
                transport=_FakeTransport(snapshot=payload),
            )
        return original_invoke(request)

    runtime.registry.invoke = _invoke  # type: ignore[method-assign]

    def _follow_up(context):
        if "resolve_media" not in runtime.registry.invocations:
            return InvestigationProposal(
                action="invoke_tool",
                tool_name="resolve_media",
                label=None,
                rationale="acquire snapshot",
                camera_ids=("front_yard",),
                media_kind="snapshot",
                observation_id="obs-exec-t5-02-0",
            )
        snap = next(
            (
                e
                for e in runtime.store.evidence.values()
                if e.evidence_kind is EvidenceKind.SNAPSHOT
            ),
            None,
        )
        if snap is not None and "observe_media" not in runtime.registry.invocations:
            return InvestigationProposal(
                action="invoke_tool",
                tool_name="observe_media",
                label=None,
                rationale="examine retained snapshot",
                camera_ids=("front_yard",),
                evidence_id=snap.evidence_id,
                objective="Does the image appear to contain a vehicle?",
            )
        return InvestigationProposal(
            action="proceed_to_report",
            tool_name=None,
            label=None,
            rationale="visual interpretation registered; not a verified fact",
        )

    provider.propose_follow_up = _follow_up  # type: ignore[method-assign]
    run_investigation(runtime, thread_id="thread-t5-02")

    assert runtime.registry.invocations == [
        "search_events",
        "resolve_media",
        "observe_media",
    ]
    observe_claims = [
        c
        for c in runtime.store.claims.values()
        if c.claim_type is ClaimType.MODEL_INTERPRETATION
    ]
    assert observe_claims
    assert all("local visual model" in c.statement.lower() for c in observe_claims)
    assert all(
        "not an independently verified" in " ".join(c.limitations).lower()
        for c in observe_claims
    )
    assert runtime.last_gate is not None and runtime.last_gate.accepted
    assert runtime.last_gate.report is not None
    joined_limits = " ".join(runtime.last_gate.report.limitations).lower()
    assert "model inferences" in joined_limits or "local vlm" in joined_limits

    receipt_ids = [
        rid
        for rid, r in runtime.store.receipts.items()
        if r.tool_name == "observe_media"
    ]
    assert receipt_ids
    observe_receipt = runtime.store.receipts[receipt_ids[0]]
    assert observe_receipt.filters.get("physical_world_fact") is False
    assert observe_receipt.filters.get("observation_text")
    assert observe_receipt.filters.get("source_content_sha256")
    assert observe_receipt.filters.get("inference_input_sha256")
    assert observe_receipt.filters.get("preprocessing") is not None
    media_id = str(observe_receipt.filters.get("media_evidence_id"))
    digest = str(observe_receipt.filters.get("content_sha256"))
    # Retained source hash remains authoritative and distinct from optional normalize.
    assert digest == str(observe_receipt.filters.get("source_content_sha256"))

    del runtime
    repo = InvestigationRepository(domain_db)
    loaded_media = repo.get_evidence(media_id)
    assert loaded_media is not None
    assert loaded_media.content_sha256 == digest
    loaded_receipt = repo.get_search_receipt(receipt_ids[0])
    assert loaded_receipt is not None
    assert loaded_receipt.tool_name == "observe_media"
    assert loaded_receipt.filters.get("observation_text")
    store = LocalMediaStore(tmp_path / "media_root")
    retained = store.get(
        case_id="case-t5-02",
        retained_media_id=str(loaded_media.retrieval_spec["retained_media_id"]),
    )
    assert retained.content_sha256 == digest
    domain_db.close()


def test_t5_03_temporal_plus_visual_remains_qualified(tmp_path: Path):
    payload = b"\xff\xd8\xff" + b"near" * 60
    t0 = datetime(2023, 11, 14, 22, 10, tzinfo=UTC)
    person = ok_event_result(
        execution_id="exec-t5-03p",
        native_id="evt-t5-03p",
        label="person",
        start=t0,
        end=t0 + timedelta(seconds=20),
    )
    car = ok_event_result(
        execution_id="exec-t5-03c",
        native_id="evt-t5-03c",
        label="car",
        start=t0 + timedelta(seconds=30),
        end=t0 + timedelta(seconds=50),
    )
    provider = StaticModelProvider(
        initial_proposal=InvestigationProposal(
            action="invoke_tool",
            tool_name="search_events",
            label="person",
            rationale="person first",
            camera_ids=("front_yard",),
        ),
        proposals=[
            InvestigationProposal(
                action="invoke_tool",
                tool_name="search_events",
                label="car",
                rationale="vehicle next",
                camera_ids=("front_yard",),
            ),
            InvestigationProposal(
                action="invoke_tool",
                tool_name="analyze_temporal_relations",
                label=None,
                rationale="check proximity",
                camera_ids=("front_yard",),
                predicate="near",
                window_seconds=180.0,
                anchor_label="person",
                candidate_label="car",
            ),
            InvestigationProposal(
                action="proceed_to_report",
                tool_name=None,
                label=None,
                rationale="insufficient visual proof of interaction",
            ),
        ],
    )
    runtime = InvestigationRuntime.from_fixture(
        case=make_case(
            case_id="case-t5-03",
            question=(
                "Investigate person and nearby vehicle detections and assess whether "
                "available visual evidence supports a proposed relationship."
            ),
            start=datetime(2023, 11, 14, 22, 0, tzinfo=UTC),
            end=datetime(2023, 11, 14, 23, 30, tzinfo=UTC),
        ),
        scenario=ScenarioConfig(
            name="t5_03",
            tool_backend="stub",
            use_model_initial_plan=True,
            use_model_follow_up=True,
            max_model_calls=6,
            max_tool_attempts=6,
            enable_observe_media=True,
            tool_scripts={"search_events": [person, car]},
            media_store_root=str(tmp_path / "media_root"),
        ),
        model_provider=provider,
        visual_provider=StaticVisualProvider(),
    )
    run_investigation(runtime, thread_id="thread-t5-03")
    assert "analyze_temporal_relations" in runtime.registry.invocations
    # Visual tool may be skipped when no media was resolved — that is correct.
    joined = " ".join(c.statement.lower() for c in runtime.store.claims.values())
    assert "entered a vehicle" not in joined
    assert "same person" not in joined
    assert runtime.last_gate is not None and runtime.last_gate.accepted
    _ = payload  # reserved for optional media path


def test_t5_05_anti_repeat_observe(tmp_path: Path):
    payload = b"\xff\xd8\xff" + b"rpt" * 40
    t0 = datetime(2023, 11, 14, 22, 15, tzinfo=UTC)
    person = ok_event_result(
        execution_id="exec-t5-05",
        native_id="evt-t5-05",
        label="car",
        start=t0,
        end=t0 + timedelta(seconds=15),
    )
    calls = {"n": 0}
    vision = StaticVisualProvider(observation_text="Possible vehicle visible.")
    original_observe = vision.observe

    def _counting(request):
        calls["n"] += 1
        return original_observe(request)

    vision.observe = _counting  # type: ignore[method-assign]

    provider = StaticModelProvider(
        initial_proposal=InvestigationProposal(
            action="invoke_tool",
            tool_name="search_events",
            label="car",
            rationale="search",
            camera_ids=("front_yard",),
        ),
        proposals=[],
    )
    runtime = InvestigationRuntime.from_fixture(
        case=make_case(
            case_id="case-t5-05",
            question=VISUAL_Q,
            start=datetime(2023, 11, 14, 22, 0, tzinfo=UTC),
            end=datetime(2023, 11, 14, 23, 30, tzinfo=UTC),
            vms_instance_id="vms-local-1",
        ),
        scenario=ScenarioConfig(
            name="t5_05",
            tool_backend="stub",
            use_model_initial_plan=True,
            use_model_follow_up=True,
            max_model_calls=6,
            max_tool_attempts=5,
            enable_observe_media=True,
            tool_scripts={"search_events": [person]},
            media_store_root=str(tmp_path / "media_root"),
        ),
        model_provider=provider,
        visual_provider=vision,
    )

    phase = {"step": 0}
    original_invoke = runtime.registry.invoke

    def _invoke(request: ToolRequest):
        if request.tool_name == "resolve_media":
            runtime.registry.invocations.append("resolve_media")
            return resolve_media(
                request,
                evidence_by_id=runtime.store.evidence,
                observations_by_id=runtime.store.observations,
                scope=runtime.case.effective_scope,
                case_id=runtime.case.case_id,
                media_store=runtime.media_store,
                transport=_FakeTransport(snapshot=payload),
            )
        return original_invoke(request)

    runtime.registry.invoke = _invoke  # type: ignore[method-assign]

    def _follow_up(context):
        phase["step"] += 1
        if phase["step"] == 1:
            return InvestigationProposal(
                action="invoke_tool",
                tool_name="resolve_media",
                label=None,
                rationale="acquire",
                camera_ids=("front_yard",),
                media_kind="snapshot",
                observation_id="obs-exec-t5-05-0",
            )
        snap = next(
            e
            for e in runtime.store.evidence.values()
            if e.evidence_kind is EvidenceKind.SNAPSHOT
        )
        if phase["step"] in {2, 3}:
            return InvestigationProposal(
                action="invoke_tool",
                tool_name="observe_media",
                label=None,
                rationale="examine",
                camera_ids=("front_yard",),
                evidence_id=snap.evidence_id,
                objective="Does the image appear to contain a vehicle?",
            )
        return InvestigationProposal(
            action="proceed_to_report",
            tool_name=None,
            label=None,
            rationale="done",
        )

    provider.propose_follow_up = _follow_up  # type: ignore[method-assign]
    run_investigation(runtime, thread_id="thread-t5-05")
    assert calls["n"] == 1
    assert runtime.registry.invocations.count("observe_media") == 1


def test_t5_capability_observe_only_when_enabled_and_available():
    runtime = InvestigationRuntime.from_fixture(
        case=make_case(case_id="case-t5-caps", question=PERSON_Q),
        scenario=ScenarioConfig(
            name="t5_caps_off",
            tool_backend="stub",
            enable_observe_media=False,
        ),
        visual_provider=StaticVisualProvider(),
    )
    assert runtime.capabilities.for_operation("observe_media") is None

    # Flag on but backend unavailable → do not advertise SUPPORTED.
    runtime_unavail = InvestigationRuntime.from_fixture(
        case=make_case(case_id="case-t5-caps-unavail", question=PERSON_Q),
        scenario=ScenarioConfig(
            name="t5_caps_unavail",
            tool_backend="stub",
            enable_observe_media=True,
        ),
        visual_provider=StaticVisualProvider(available=False),
    )
    assert runtime_unavail.capabilities.for_operation("observe_media") is None

    runtime2 = InvestigationRuntime.from_fixture(
        case=make_case(case_id="case-t5-caps2", question=PERSON_Q),
        scenario=ScenarioConfig(
            name="t5_caps_on",
            tool_backend="stub",
            enable_observe_media=True,
        ),
        visual_provider=StaticVisualProvider(),
    )
    cap = runtime2.capabilities.for_operation("observe_media")
    assert cap is not None
    assert cap.availability.value == "supported"


def test_t5_real_moondream_on_retained_frigate_snapshot(tmp_path: Path):
    """Real local VLM on a privately retained Frigate snapshot."""
    import urllib.error
    import urllib.request

    snap_path = PRIVATE_MEDIA / "retained-sample.snapshot.jpg"
    if not snap_path.is_file():
        pytest.skip(
            "Optional real-VLM check: provide a locally retained sample snapshot "
            f"at {snap_path} (git-ignored). Not required for public regression."
        )

    ollama_url = "http://127.0.0.1:11434"
    try:
        with urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=5) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        pytest.fail(f"T5 requires reachable local Ollama: {exc}")
    names = {m.get("name") for m in payload.get("models", []) if isinstance(m, dict)}
    if "moondream:1.8b" not in names and "moondream" not in names:
        pytest.fail("T5 requires installed local vision model moondream:1.8b")

    image = snap_path.read_bytes()
    evidence, store = _media_evidence(
        tmp_path, payload=image, case_id="case-t5-real"
    )
    # Align evidence id path with case.
    retained_id = evidence.retrieval_spec["retained_media_id"]
    # Re-put under case-t5-real (already done by helper).
    provider = OllamaVisualProvider(model="moondream:1.8b", base_url=ollama_url)
    assert provider.is_available()
    result = observe_media(
        ToolRequest(
            tool_name="observe_media",
            arguments={
                "evidence_id": evidence.evidence_id,
                "objective": "Does the image appear to contain a person or a vehicle?",
            },
            source_id="vms-1",
            camera_ids=("front_yard",),
        ),
        evidence_by_id={evidence.evidence_id: evidence},
        scope=_scope(),
        case_id="case-t5-real",
        media_store=store,
        visual_provider=provider,
    )
    assert result.status is ToolResultStatus.OK, (
        None if result.error is None else result.error.message
    )
    record = result.data[0]
    assert record["model_name"].startswith("moondream")
    source_digest = hashlib.sha256(image).hexdigest()
    assert record["content_sha256"] == source_digest
    assert record["source_content_sha256"] == source_digest
    assert record["inference_input_sha256"]
    # Normalization may change inference bytes; retained source hash stays authoritative.
    assert record["preprocessing"] in {"none", "ffmpeg_reencode_q5_max_edge"}
    if record["preprocessing"] != "none":
        assert record["inference_input_sha256"] != source_digest
    assert record["physical_world_fact"] is False
    assert isinstance(record["observation_text"], str)
    assert len(record["observation_text"].strip()) > 0
    # Keep attribution bounded — no definitive physical-world claim encoded.
    assert record.get("visual_interpretation") is True
    # Absence-style model wording must remain model opinion only.
    lower = record["observation_text"].lower()
    if "not a person" in lower or "no person" in lower:
        assert any(
            "absence" in lim.lower() for lim in (record.get("limitations") or [])
        )
    _ = retained_id


def test_t5f_gate_b_mismatched_media_receipt_is_unsupported():
    """Arbitrary observe receipt cannot ground a claim for unrelated media."""
    from nanexus_investigation.contracts.claim import Claim
    from nanexus_investigation.contracts.search import SearchExecutionReceipt
    from nanexus_investigation.contracts.tool import ToolResult
    from nanexus_investigation.evidence.claims import assess_claim

    t0 = datetime(2023, 11, 14, 22, 15, tzinfo=UTC)
    media_a = EvidenceRef(
        evidence_id="ev-media-a",
        source_locator=SourceLocator(
            vms_instance_id="vms-1",
            record_type=RecordType.EVENT,
            native_record_id="evt-a",
        ),
        evidence_kind=EvidenceKind.SNAPSHOT,
        camera_id="front_yard",
        interval=Interval(start_utc=t0, end_utc=t0 + timedelta(seconds=5)),
        retrieval_spec={"retained_media_id": "media-a", "bytes_retained": True},
        availability=EvidenceAvailability.AVAILABLE,
        content_sha256="a" * 64,
        content_type="image/jpeg",
        provenance="adapter:frigate-media",
    )
    media_b = EvidenceRef(
        evidence_id="ev-media-b",
        source_locator=SourceLocator(
            vms_instance_id="vms-1",
            record_type=RecordType.EVENT,
            native_record_id="evt-b",
        ),
        evidence_kind=EvidenceKind.SNAPSHOT,
        camera_id="front_yard",
        interval=Interval(start_utc=t0, end_utc=t0 + timedelta(seconds=5)),
        retrieval_spec={"retained_media_id": "media-b", "bytes_retained": True},
        availability=EvidenceAvailability.AVAILABLE,
        content_sha256="b" * 64,
        content_type="image/jpeg",
        provenance="adapter:frigate-media",
    )
    obs_text = "The image contains a car, not a person."
    receipt = SearchExecutionReceipt(
        search_execution_id="receipt-observe-b",
        tool_name="observe_media",
        source_id="vms-1",
        camera_ids=("front_yard",),
        time_bounds=_scope().time_bounds,
        filters={"media_evidence_id": "ev-media-b"},
        retrieved_at=t0,
        pages_fetched=1,
        match_count=1,
        coverage=SearchCoverage.EXHAUSTED,
    )
    tool = ToolResult(
        status=ToolResultStatus.OK,
        execution_id="exec-observe-b",
        data=[
            {
                "record_kind": "visual_observation",
                "media_evidence_id": "ev-media-b",
                "observation_text": obs_text,
                "physical_world_fact": False,
            }
        ],
        source_id="vms-1",
        retrieved_at=t0,
        coverage=SearchCoverage.EXHAUSTED,
        provenance="adapter:local-vlm",
    )
    # Claim cites media A but attaches receipt for media B.
    claim = Claim(
        claim_id="claim-mismatch",
        statement=(
            f"The local visual model (moondream:1.8b) interpreted retained "
            f"snapshot ev-media-a as: {obs_text}"
        ),
        claim_type=ClaimType.MODEL_INTERPRETATION,
        evidence_ids=("ev-media-a",),
        search_receipt_ids=("receipt-observe-b",),
        producer="local-vlm:moondream:1.8b",
        validation_status=ClaimValidationStatus.QUALIFIED,
    )
    result = assess_claim(
        claim,
        effective_scope=_scope(),
        evidence_by_id={"ev-media-a": media_a, "ev-media-b": media_b},
        receipts_by_id={"receipt-observe-b": receipt},
        tool_results_by_receipt_id={"receipt-observe-b": tool},
        require_grant_membership=False,
    )
    assert result.status is ClaimValidationStatus.UNSUPPORTED


def test_t5f_gate_d_temporal_then_visual_no_interaction_claim(tmp_path: Path):
    """Same Agent can combine temporal + media + observe without interaction facts."""
    payload = b"\xff\xd8\xff" + b"gate-d" * 40
    t0 = datetime(2023, 11, 14, 22, 10, tzinfo=UTC)
    person = ok_event_result(
        execution_id="exec-t5f-dp",
        native_id="evt-t5f-dp",
        label="person",
        start=t0,
        end=t0 + timedelta(seconds=20),
    )
    car = ok_event_result(
        execution_id="exec-t5f-dc",
        native_id="evt-t5f-dc",
        label="car",
        start=t0 + timedelta(seconds=30),
        end=t0 + timedelta(seconds=50),
    )
    provider = StaticModelProvider(
        initial_proposal=InvestigationProposal(
            action="invoke_tool",
            tool_name="search_events",
            label="person",
            rationale="person first",
            camera_ids=("front_yard",),
        ),
        proposals=[],
    )
    runtime = InvestigationRuntime.from_fixture(
        case=make_case(
            case_id="case-t5f-d",
            question=(
                "Investigate person and nearby vehicle detections and assess whether "
                "available visual evidence supports a proposed relationship."
            ),
            start=datetime(2023, 11, 14, 22, 0, tzinfo=UTC),
            end=datetime(2023, 11, 14, 23, 30, tzinfo=UTC),
            vms_instance_id="vms-local-1",
        ),
        scenario=ScenarioConfig(
            name="t5f_d",
            tool_backend="stub",
            use_model_initial_plan=True,
            use_model_follow_up=True,
            max_model_calls=10,
            max_tool_attempts=10,
            max_plan_steps=6,
            enable_observe_media=True,
            tool_scripts={"search_events": [person, car]},
            media_store_root=str(tmp_path / "media_root"),
        ),
        model_provider=provider,
        visual_provider=StaticVisualProvider(
            observation_text="The image appears to show a possible vehicle."
        ),
    )
    original_invoke = runtime.registry.invoke

    def _invoke(request: ToolRequest):
        if request.tool_name == "resolve_media":
            runtime.registry.invocations.append("resolve_media")
            return resolve_media(
                request,
                evidence_by_id=runtime.store.evidence,
                observations_by_id=runtime.store.observations,
                scope=runtime.case.effective_scope,
                case_id=runtime.case.case_id,
                media_store=runtime.media_store,
                transport=_FakeTransport(snapshot=payload),
            )
        return original_invoke(request)

    runtime.registry.invoke = _invoke  # type: ignore[method-assign]

    def _follow(context):
        if "car" not in {
            label
            for obs in runtime.store.observations.values()
            for label in obs.labels
        }:
            return InvestigationProposal(
                action="invoke_tool",
                tool_name="search_events",
                label="car",
                rationale="vehicle next",
                camera_ids=("front_yard",),
            )
        if not any(
            r.tool_name == "analyze_temporal_relations"
            for r in runtime.store.receipts.values()
        ):
            return InvestigationProposal(
                action="invoke_tool",
                tool_name="analyze_temporal_relations",
                label=None,
                rationale="proximity check",
                camera_ids=("front_yard",),
                predicate="near",
                window_seconds=180.0,
                anchor_label="person",
                candidate_label="car",
            )
        snap = next(
            (
                e
                for e in runtime.store.evidence.values()
                if e.evidence_kind is EvidenceKind.SNAPSHOT
            ),
            None,
        )
        if snap is None:
            obs_id = next(
                oid
                for oid, obs in runtime.store.observations.items()
                if "car" in obs.labels
            )
            return InvestigationProposal(
                action="invoke_tool",
                tool_name="resolve_media",
                label=None,
                rationale="acquire car snapshot",
                camera_ids=("front_yard",),
                media_kind="snapshot",
                observation_id=obs_id,
            )
        if not any(
            r.tool_name == "observe_media" for r in runtime.store.receipts.values()
        ):
            return InvestigationProposal(
                action="invoke_tool",
                tool_name="observe_media",
                label=None,
                rationale="examine snapshot",
                camera_ids=("front_yard",),
                evidence_id=snap.evidence_id,
                objective="Does the image appear to contain a vehicle?",
            )
        return InvestigationProposal(
            action="proceed_to_report",
            tool_name=None,
            label=None,
            rationale="insufficient evidence for interaction",
        )

    provider.propose_follow_up = _follow  # type: ignore[method-assign]
    run_investigation(runtime, thread_id="thread-t5f-d")
    assert "analyze_temporal_relations" in runtime.registry.invocations
    assert "resolve_media" in runtime.registry.invocations
    assert "observe_media" in runtime.registry.invocations
    assert any(
        e.evidence_kind is EvidenceKind.SNAPSHOT for e in runtime.store.evidence.values()
    )
    joined = " ".join(c.statement.lower() for c in runtime.store.claims.values())
    assert "entered a vehicle" not in joined
    assert "same person" not in joined
    assert "interaction" not in joined or "does not establish" in " ".join(
        " ".join(c.limitations).lower() for c in runtime.store.claims.values()
    )
    observe_claims = [
        c
        for c in runtime.store.claims.values()
        if c.claim_type is ClaimType.MODEL_INTERPRETATION
    ]
    assert observe_claims
    assert all(
        "not an independently verified" in " ".join(c.limitations).lower()
        for c in observe_claims
    )
    assert all(
        "absence" in " ".join(c.limitations).lower() for c in observe_claims
    )
    assert runtime.last_gate is not None and runtime.last_gate.accepted
