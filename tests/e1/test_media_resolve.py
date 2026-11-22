"""V0.1 T4: selective media retrieval and durable provenance.

Deterministic unit/stub tests always run. Live Frigate media tests are opt-in
via ``pytest -m live_frigate --live-frigate``. Private recorded media under
``.local/`` is used when present and never committed.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from nanexus_investigation.contracts.enums import (
    CaseStatus,
    EvidenceAvailability,
    EvidenceKind,
    ObservationType,
    RecordType,
    ToolErrorCode,
    ToolResultStatus,
)
from nanexus_investigation.contracts.evidence import EvidenceRef, Observation
from nanexus_investigation.contracts.identity import Interval, SourceLocator
from nanexus_investigation.contracts.scope import InvestigationScope
from nanexus_investigation.contracts.tool import ToolRequest
from nanexus_investigation.evidence.media_store import LocalMediaStore, MediaStoreError
from nanexus_investigation.graph.runtime import (
    ScenarioConfig,
    make_case,
    ok_event_result,
    register_tool_artifacts,
)
from nanexus_investigation.graph.workflow import InvestigationRuntime, run_investigation
from nanexus_investigation.model.provider import (
    InvestigationProposal,
    StaticModelProvider,
)
from nanexus_investigation.persistence.db import InvestigationDatabase
from nanexus_investigation.persistence.repository import InvestigationRepository
from nanexus_investigation.tools.media_resolve import resolve_media
from nanexus_investigation.vms.frigate.client import (
    FrigateHttpResponse,
    FrigateMediaTooLargeError,
    FrigateNotFoundError,
)

UTC = timezone.utc
PRIVATE_MEDIA = Path(".local/frigate_recordings/media")
VISUAL_Q = (
    "Retrieve a snapshot for a registered driveway detection so we have "
    "visual evidence on file. Do not invent what the image shows."
)
PERSON_Q = "Find person detections in this camera and time window."


class _FakeTransport:
    def __init__(
        self,
        *,
        snapshot: bytes | None = None,
        clip: bytes | None = None,
        content_type_snapshot: str = "image/jpeg",
        content_type_clip: str = "video/mp4",
        missing: bool = False,
        oversized: bool = False,
    ) -> None:
        self.snapshot = snapshot
        self.clip = clip
        self.content_type_snapshot = content_type_snapshot
        self.content_type_clip = content_type_clip
        self.missing = missing
        self.oversized = oversized

    def get_event_snapshot(self, event_id: str, *, max_bytes: int = 0):
        if self.missing or self.snapshot is None:
            raise FrigateNotFoundError("Frigate media not found (snapshot.jpg)")
        if self.oversized or (max_bytes and len(self.snapshot) > max_bytes):
            raise FrigateMediaTooLargeError(
                f"media response exceeds max_bytes={max_bytes}"
            )
        return FrigateHttpResponse(
            status_code=200,
            path=f"/api/events/{event_id}/snapshot.jpg",
            query={},
            body=self.snapshot,
            headers={"Content-Type": self.content_type_snapshot},
            retrieved_at=datetime.now(tz=UTC),
        )

    def get_event_clip(self, event_id: str, *, max_bytes: int = 0):
        if self.missing or self.clip is None:
            raise FrigateNotFoundError("Frigate media not found (clip.mp4)")
        if self.oversized or (max_bytes and len(self.clip) > max_bytes):
            raise FrigateMediaTooLargeError(
                f"media response exceeds max_bytes={max_bytes}"
            )
        return FrigateHttpResponse(
            status_code=200,
            path=f"/api/events/{event_id}/clip.mp4",
            query={},
            body=self.clip,
            headers={"Content-Type": self.content_type_clip},
            retrieved_at=datetime.now(tz=UTC),
        )


def _source_artifacts(tmp_path: Path, *, native_id: str = "evt-media-1"):
    t0 = datetime(2023, 11, 14, 22, 15, tzinfo=UTC)
    obs = Observation(
        observation_id="obs-media-1",
        source_locator=SourceLocator(
            vms_instance_id="vms-1",
            record_type=RecordType.EVENT,
            native_record_id=native_id,
        ),
        source_revision="rev-1",
        observation_type=ObservationType.DETECTION_EVENT,
        camera_id="front_yard",
        start_utc=t0,
        end_utc=t0 + timedelta(seconds=20),
        labels=("car",),
        retrieved_at=t0,
        provenance="stub:search_events",
    )
    evidence = EvidenceRef(
        evidence_id="ev-media-src-1",
        source_locator=obs.source_locator,
        evidence_kind=EvidenceKind.EVENT_METADATA,
        camera_id="front_yard",
        interval=Interval(start_utc=obs.start_utc, end_utc=obs.end_utc),
        retrieval_spec={"operation": "search_events", "native_id": native_id},
        availability=EvidenceAvailability.AVAILABLE,
        provenance="stub:search_events",
    )
    scope = InvestigationScope(
        vms_instance_id="vms-1",
        camera_ids=("front_yard",),
        time_bounds=Interval(
            start_utc=datetime(2023, 11, 14, 22, 0, tzinfo=UTC),
            end_utc=datetime(2023, 11, 14, 23, 30, tzinfo=UTC),
        ),
    )
    store = LocalMediaStore(tmp_path / "media")
    return obs, evidence, scope, store


def test_t4_unit_snapshot_hash_and_retention(tmp_path: Path):
    obs, evidence, scope, store = _source_artifacts(tmp_path)
    payload = b"\xff\xd8\xff" + b"jpeg-bytes" * 100
    result = resolve_media(
        ToolRequest(
            tool_name="resolve_media",
            arguments={"media_kind": "snapshot", "evidence_id": evidence.evidence_id},
            source_id="vms-1",
            camera_ids=("front_yard",),
        ),
        evidence_by_id={evidence.evidence_id: evidence},
        observations_by_id={obs.observation_id: obs},
        scope=scope,
        case_id="case-t4-unit",
        media_store=store,
        transport=_FakeTransport(snapshot=payload),
    )
    assert result.status is ToolResultStatus.OK
    record = result.data[0]
    assert record["content_sha256"] == hashlib.sha256(payload).hexdigest()
    assert record["bytes_retained"] is True
    assert record["visual_interpretation"] is False
    retained = store.get(
        case_id="case-t4-unit",
        retained_media_id=record["retained_media_id"],
    )
    assert retained.bytes == payload


def test_t4_unit_missing_media_and_rejects_urls(tmp_path: Path):
    obs, evidence, scope, store = _source_artifacts(tmp_path)
    missing = resolve_media(
        ToolRequest(
            tool_name="resolve_media",
            arguments={"media_kind": "snapshot", "evidence_id": evidence.evidence_id},
            source_id="vms-1",
            camera_ids=("front_yard",),
        ),
        evidence_by_id={evidence.evidence_id: evidence},
        observations_by_id={obs.observation_id: obs},
        scope=scope,
        case_id="case-t4-missing",
        media_store=store,
        transport=_FakeTransport(missing=True),
    )
    assert missing.status is ToolResultStatus.EMPTY
    assert missing.error.code is ToolErrorCode.MISSING_MEDIA

    banned = resolve_media(
        ToolRequest(
            tool_name="resolve_media",
            arguments={
                "media_kind": "snapshot",
                "evidence_id": evidence.evidence_id,
                "url": "https://evil.example/img.jpg",
            },
            source_id="vms-1",
            camera_ids=("front_yard",),
        ),
        evidence_by_id={evidence.evidence_id: evidence},
        observations_by_id={obs.observation_id: obs},
        scope=scope,
        case_id="case-t4-url",
        media_store=store,
        transport=_FakeTransport(snapshot=b"x"),
    )
    assert banned.status is ToolResultStatus.ERROR
    assert "rejects model-supplied url" in banned.error.message

    foreign = resolve_media(
        ToolRequest(
            tool_name="resolve_media",
            arguments={"media_kind": "snapshot", "evidence_id": "ev-foreign"},
            source_id="vms-1",
            camera_ids=("front_yard",),
        ),
        evidence_by_id={evidence.evidence_id: evidence},
        observations_by_id={obs.observation_id: obs},
        scope=scope,
        case_id="case-t4-foreign",
        media_store=store,
        transport=_FakeTransport(snapshot=b"x"),
    )
    assert foreign.status is ToolResultStatus.ERROR
    assert "unknown or foreign" in foreign.error.message


def test_t4_unit_invalid_type_empty_oversized_hash_mismatch(tmp_path: Path):
    obs, evidence, scope, store = _source_artifacts(tmp_path)
    bad_type = resolve_media(
        ToolRequest(
            tool_name="resolve_media",
            arguments={"media_kind": "snapshot", "evidence_id": evidence.evidence_id},
            source_id="vms-1",
            camera_ids=("front_yard",),
        ),
        evidence_by_id={evidence.evidence_id: evidence},
        observations_by_id={obs.observation_id: obs},
        scope=scope,
        case_id="case-t4-type",
        media_store=store,
        transport=_FakeTransport(
            snapshot=b"not-jpeg", content_type_snapshot="text/plain"
        ),
    )
    assert bad_type.status is ToolResultStatus.ERROR
    assert bad_type.error.code is ToolErrorCode.INVALID_RESPONSE

    oversized = resolve_media(
        ToolRequest(
            tool_name="resolve_media",
            arguments={"media_kind": "snapshot", "evidence_id": evidence.evidence_id},
            source_id="vms-1",
            camera_ids=("front_yard",),
        ),
        evidence_by_id={evidence.evidence_id: evidence},
        observations_by_id={obs.observation_id: obs},
        scope=scope,
        case_id="case-t4-over",
        media_store=store,
        transport=_FakeTransport(snapshot=b"x" * 100, oversized=True),
        max_snapshot_bytes=10,
    )
    assert oversized.status is ToolResultStatus.ERROR

    # Integrity mismatch on reopen.
    store.put(
        case_id="case-t4-integrity",
        retained_media_id="media-test",
        media_kind="snapshot",
        content_type="image/jpeg",
        payload=b"abc",
    )
    bin_path = tmp_path / "media" / "case-t4-integrity" / "media-test.bin"
    bin_path.write_bytes(b"tampered")
    with pytest.raises(MediaStoreError, match="hash mismatch"):
        store.get(case_id="case-t4-integrity", retained_media_id="media-test")


def test_t4f_gate_a_media_store_rejects_path_escape(tmp_path: Path):
    store = LocalMediaStore(tmp_path / "media")
    with pytest.raises(MediaStoreError):
        store.put(
            case_id="../escape",
            retained_media_id="media-ok",
            media_kind="snapshot",
            content_type="image/jpeg",
            payload=b"abc",
        )
    with pytest.raises(MediaStoreError):
        store.put(
            case_id="case-ok",
            retained_media_id="../escape",
            media_kind="snapshot",
            content_type="image/jpeg",
            payload=b"abc",
        )
    with pytest.raises(MediaStoreError):
        store.get(case_id="case-ok", retained_media_id="../../etc/passwd")
    assert not (tmp_path / "escape").exists()


def test_t4f_gate_a_failed_registration_does_not_persist_media_evidence(tmp_path: Path):
    """Bytes may exist locally, but failed domain commit must not yield EvidenceRef."""
    from nanexus_investigation.graph.runtime import DomainRegistrationError

    payload = b"\xff\xd8\xff" + b"gate-a" * 50
    t0 = datetime(2023, 11, 14, 22, 15, tzinfo=UTC)
    person = ok_event_result(
        execution_id="exec-t4f-fail",
        native_id="evt-t4f-fail",
        label="car",
        start=t0,
        end=t0 + timedelta(seconds=15),
        camera_id="front_yard",
    )
    domain_db = InvestigationDatabase.open(tmp_path / "t4f-fail.sqlite")
    case = make_case(
        case_id="case-t4f-fail",
        question=VISUAL_Q,
        cameras=("front_yard",),
        start=datetime(2023, 11, 14, 22, 0, tzinfo=UTC),
        end=datetime(2023, 11, 14, 23, 30, tzinfo=UTC),
        vms_instance_id="vms-local-1",
    )
    runtime = InvestigationRuntime.from_durable_fixture(
        case=case,
        scenario=ScenarioConfig(
            name="t4f_fail",
            tool_backend="stub",
            use_model_initial_plan=False,
            use_model_follow_up=False,
            max_tool_attempts=2,
            tool_scripts={"search_events": [person]},
            media_store_root=str(tmp_path / "media_root"),
        ),
        domain_db=domain_db,
    )
    search_req = ToolRequest(
        tool_name="search_events",
        arguments={
            "label": "car",
            "camera_ids": ["front_yard"],
            "time_bounds": {
                "start_utc": "2023-11-14T22:00:00+00:00",
                "end_utc": "2023-11-14T23:30:00+00:00",
            },
        },
        source_id="vms-local-1",
        camera_ids=("front_yard",),
    )
    search_result = runtime.registry.invoke(search_req)
    register_tool_artifacts(
        runtime, request=search_req, result=search_result, plan_step_id="step-search"
    )
    source_ev = next(iter(runtime.store.evidence.values()))
    media_result = resolve_media(
        ToolRequest(
            tool_name="resolve_media",
            arguments={
                "media_kind": "snapshot",
                "evidence_id": source_ev.evidence_id,
            },
            source_id="vms-local-1",
            camera_ids=("front_yard",),
        ),
        evidence_by_id=runtime.store.evidence,
        observations_by_id=runtime.store.observations,
        scope=runtime.case.effective_scope,
        case_id=runtime.case.case_id,
        media_store=runtime.media_store,
        transport=_FakeTransport(snapshot=payload),
    )
    assert media_result.status is ToolResultStatus.OK

    def _boom(*args, **kwargs):
        raise RuntimeError("forced persistence failure")

    runtime.domain_repo.register_artifact_group = _boom  # type: ignore[method-assign]
    media_req = ToolRequest(
        tool_name="resolve_media",
        arguments={
            "media_kind": "snapshot",
            "evidence_id": source_ev.evidence_id,
        },
        source_id="vms-local-1",
        camera_ids=("front_yard",),
    )
    with pytest.raises(DomainRegistrationError):
        register_tool_artifacts(
            runtime,
            request=media_req,
            result=media_result,
            plan_step_id="step-media",
        )

    assert not any(
        e.evidence_kind is EvidenceKind.SNAPSHOT for e in runtime.store.evidence.values()
    )
    repo = InvestigationRepository(domain_db)
    for op_id in repo.list_registration_group_operation_ids(case.case_id):
        group = repo.get_registration_group(op_id)
        assert all(
            item.evidence_kind is not EvidenceKind.SNAPSHOT for item in group.evidences
        )
    domain_db.close()


def test_t4f_gate_b_anti_repeat_and_budget_bound_media_downloads(tmp_path: Path):
    payload = b"\xff\xd8\xff" + b"budget" * 40
    t0 = datetime(2023, 11, 14, 22, 15, tzinfo=UTC)
    person = ok_event_result(
        execution_id="exec-t4f-budget",
        native_id="evt-t4f-budget",
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
        proposals=[
            InvestigationProposal(
                action="invoke_tool",
                tool_name="resolve_media",
                label=None,
                rationale="acquire snapshot once",
                camera_ids=("front_yard",),
                media_kind="snapshot",
                observation_id="obs-exec-t4f-budget-0",
            ),
            InvestigationProposal(
                action="invoke_tool",
                tool_name="resolve_media",
                label=None,
                rationale="repeat same media request",
                camera_ids=("front_yard",),
                media_kind="snapshot",
                observation_id="obs-exec-t4f-budget-0",
            ),
            InvestigationProposal(
                action="proceed_to_report",
                tool_name=None,
                label=None,
                rationale="done",
            ),
        ],
    )
    calls = {"n": 0}

    class _CountingTransport(_FakeTransport):
        def get_event_snapshot(self, event_id: str, *, max_bytes: int = 0):
            calls["n"] += 1
            return super().get_event_snapshot(event_id, max_bytes=max_bytes)

    runtime = InvestigationRuntime.from_fixture(
        case=make_case(
            case_id="case-t4f-budget",
            question=VISUAL_Q,
            cameras=("front_yard",),
            start=datetime(2023, 11, 14, 22, 0, tzinfo=UTC),
            end=datetime(2023, 11, 14, 23, 30, tzinfo=UTC),
            vms_instance_id="vms-local-1",
        ),
        scenario=ScenarioConfig(
            name="t4f_budget",
            tool_backend="stub",
            use_model_initial_plan=True,
            use_model_follow_up=True,
            max_model_calls=5,
            max_tool_attempts=4,
            tool_scripts={"search_events": [person]},
            media_store_root=str(tmp_path / "media_root"),
        ),
        model_provider=provider,
    )
    from nanexus_investigation.tools.registry import StubToolRegistry

    assert isinstance(runtime.registry, StubToolRegistry)
    original_invoke = runtime.registry.invoke
    transport = _CountingTransport(snapshot=payload)

    def _invoke(request: ToolRequest):
        if request.tool_name != "resolve_media":
            return original_invoke(request)
        runtime.registry.invocations.append("resolve_media")
        return resolve_media(
            request,
            evidence_by_id=runtime.store.evidence,
            observations_by_id=runtime.store.observations,
            scope=runtime.case.effective_scope,
            case_id=runtime.case.case_id,
            media_store=runtime.media_store,
            transport=transport,
        )

    runtime.registry.invoke = _invoke  # type: ignore[method-assign]
    run_investigation(runtime, thread_id="thread-t4f-budget")
    assert calls["n"] == 1
    assert runtime.registry.invocations.count("resolve_media") == 1
    assert any(
        e.evidence_kind is EvidenceKind.SNAPSHOT for e in runtime.store.evidence.values()
    )


def test_t4f_gate_c_report_limits_media_is_not_visual_verification(tmp_path: Path):
    payload = b"\xff\xd8\xff" + b"gate-c" * 40
    t0 = datetime(2023, 11, 14, 22, 15, tzinfo=UTC)
    person = ok_event_result(
        execution_id="exec-t4f-c",
        native_id="evt-t4f-c",
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
        proposals=[
            InvestigationProposal(
                action="invoke_tool",
                tool_name="resolve_media",
                label=None,
                rationale="acquire snapshot bytes",
                camera_ids=("front_yard",),
                media_kind="snapshot",
                observation_id="obs-exec-t4f-c-0",
            ),
            InvestigationProposal(
                action="proceed_to_report",
                tool_name=None,
                label=None,
                rationale="media acquired; no visual analysis claimed",
            ),
        ],
    )
    runtime = InvestigationRuntime.from_fixture(
        case=make_case(
            case_id="case-t4f-c",
            question=VISUAL_Q,
            cameras=("front_yard",),
            start=datetime(2023, 11, 14, 22, 0, tzinfo=UTC),
            end=datetime(2023, 11, 14, 23, 30, tzinfo=UTC),
            vms_instance_id="vms-local-1",
        ),
        scenario=ScenarioConfig(
            name="t4f_c",
            tool_backend="stub",
            use_model_initial_plan=True,
            use_model_follow_up=True,
            max_model_calls=4,
            max_tool_attempts=4,
            tool_scripts={"search_events": [person]},
            media_store_root=str(tmp_path / "media_root"),
        ),
        model_provider=provider,
    )
    from nanexus_investigation.tools.registry import StubToolRegistry

    assert isinstance(runtime.registry, StubToolRegistry)
    original_invoke = runtime.registry.invoke

    def _invoke(request: ToolRequest):
        if request.tool_name != "resolve_media":
            return original_invoke(request)
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

    runtime.registry.invoke = _invoke  # type: ignore[method-assign]
    run_investigation(runtime, thread_id="thread-t4f-c")
    assert runtime.last_gate is not None and runtime.last_gate.accepted
    assert runtime.last_gate.report is not None
    joined_limits = " ".join(runtime.last_gate.report.limitations).lower()
    assert "not visual interpretation" in joined_limits
    joined_claims = " ".join(c.statement.lower() for c in runtime.store.claims.values())
    assert "visually verified" not in joined_claims
    assert "inspected the image" not in joined_claims
    assert "person identity" not in joined_claims
    for claim in runtime.store.claims.values():
        if any(eid.startswith("ev-media-") for eid in claim.evidence_ids):
            assert any("not visually analyzed" in lim.lower() for lim in claim.limitations)


def test_t4_01_static_snapshot_orchestration_and_durable_reopen(tmp_path: Path):
    payload = b"\xff\xd8\xff" + b"snap" * 200
    t0 = datetime(2023, 11, 14, 22, 15, tzinfo=UTC)
    person = ok_event_result(
        execution_id="exec-t4-01",
        native_id="evt-t4-01",
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
        proposals=[
            InvestigationProposal(
                action="invoke_tool",
                tool_name="resolve_media",
                label=None,
                rationale="acquire snapshot bytes",
                camera_ids=("front_yard",),
                media_kind="snapshot",
                observation_id="obs-exec-t4-01-0",
            ),
            InvestigationProposal(
                action="proceed_to_report",
                tool_name=None,
                label=None,
                rationale="media acquired; no visual analysis claimed",
            ),
        ],
    )
    domain_db = InvestigationDatabase.open(tmp_path / "t4-01.sqlite")
    case = make_case(
        case_id="case-t4-01",
        question=VISUAL_Q,
        cameras=("front_yard",),
        start=datetime(2023, 11, 14, 22, 0, tzinfo=UTC),
        end=datetime(2023, 11, 14, 23, 30, tzinfo=UTC),
        vms_instance_id="vms-local-1",
    )
    runtime = InvestigationRuntime.from_durable_fixture(
        case=case,
        scenario=ScenarioConfig(
            name="t4_01",
            tool_backend="stub",
            use_model_initial_plan=True,
            use_model_follow_up=True,
            max_model_calls=4,
            max_tool_attempts=4,
            tool_scripts={"search_events": [person]},
            media_store_root=str(tmp_path / "media_root"),
        ),
        domain_db=domain_db,
        model_provider=provider,
    )
    # Bind fake transport onto stub by injecting resolve_media script after search.
    # Stub scripts for resolve_media: run real resolve_media via registry hook.
    from nanexus_investigation.tools.registry import StubToolRegistry

    assert isinstance(runtime.registry, StubToolRegistry)
    original_invoke = runtime.registry.invoke

    def _invoke(request: ToolRequest):
        if request.tool_name != "resolve_media":
            return original_invoke(request)
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

    runtime.registry.invoke = _invoke  # type: ignore[method-assign]
    run_investigation(runtime, thread_id="thread-t4-01")

    assert runtime.registry.invocations == ["search_events", "resolve_media"]
    media_ev = [
        e
        for e in runtime.store.evidence.values()
        if e.evidence_kind is EvidenceKind.SNAPSHOT
    ]
    assert len(media_ev) == 1
    assert media_ev[0].content_sha256 == hashlib.sha256(payload).hexdigest()
    assert media_ev[0].retrieval_spec.get("visual_interpretation") is False
    joined = " ".join(c.statement.lower() for c in runtime.store.claims.values())
    assert "visual interpretation" in joined or "not visual" in joined
    assert runtime.last_gate is not None and runtime.last_gate.accepted
    assert runtime.last_gate.report is not None
    assert any(
        "not visual interpretation" in item.lower()
        for item in runtime.last_gate.report.limitations
    )

    retained_id = str(media_ev[0].retrieval_spec["retained_media_id"])
    media_evidence_id = media_ev[0].evidence_id
    digest = media_ev[0].content_sha256
    del runtime
    repo = InvestigationRepository(domain_db)
    loaded = repo.get_evidence(media_evidence_id)
    assert loaded is not None
    assert loaded.content_sha256 == digest
    assert loaded.retrieval_spec.get("source_evidence_id")
    store = LocalMediaStore(tmp_path / "media_root")
    reopened = store.get(case_id="case-t4-01", retained_media_id=retained_id)
    assert reopened.content_sha256 == digest
    assert reopened.bytes == payload
    domain_db.close()


def test_t4_03_missing_media_orchestration(tmp_path: Path):
    t0 = datetime(2023, 11, 14, 22, 15, tzinfo=UTC)
    person = ok_event_result(
        execution_id="exec-t4-03",
        native_id="evt-t4-03",
        label="car",
        start=t0,
        end=t0 + timedelta(seconds=10),
        camera_id="front_yard",
    )
    provider = StaticModelProvider(
        initial_proposal=InvestigationProposal(
            action="invoke_tool",
            tool_name="search_events",
            label="car",
            rationale="search",
            camera_ids=("front_yard",),
        ),
        proposals=[
            InvestigationProposal(
                action="invoke_tool",
                tool_name="resolve_media",
                label=None,
                rationale="try snapshot",
                camera_ids=("front_yard",),
                media_kind="snapshot",
                observation_id="obs-exec-t4-03-0",
            ),
            InvestigationProposal(
                action="proceed_to_report",
                tool_name=None,
                label=None,
                rationale="media unavailable",
            ),
        ],
    )
    runtime = InvestigationRuntime.from_fixture(
        case=make_case(
            case_id="case-t4-03",
            question=VISUAL_Q,
            start=datetime(2023, 11, 14, 22, 0, tzinfo=UTC),
            end=datetime(2023, 11, 14, 23, 30, tzinfo=UTC),
        ),
        scenario=ScenarioConfig(
            name="t4_03",
            tool_backend="stub",
            use_model_initial_plan=True,
            use_model_follow_up=True,
            max_model_calls=4,
            max_tool_attempts=4,
            tool_scripts={"search_events": [person]},
            media_store_root=str(tmp_path / "media_root"),
        ),
        model_provider=provider,
    )
    original_invoke = runtime.registry.invoke

    def _invoke(request: ToolRequest):
        if request.tool_name != "resolve_media":
            return original_invoke(request)
        runtime.registry.invocations.append("resolve_media")
        return resolve_media(
            request,
            evidence_by_id=runtime.store.evidence,
            observations_by_id=runtime.store.observations,
            scope=runtime.case.effective_scope,
            case_id=runtime.case.case_id,
            media_store=runtime.media_store,
            transport=_FakeTransport(missing=True),
        )

    runtime.registry.invoke = _invoke  # type: ignore[method-assign]
    run_investigation(runtime, thread_id="thread-t4-03")
    assert "resolve_media" in runtime.registry.invocations
    assert not any(
        e.evidence_kind is EvidenceKind.SNAPSHOT for e in runtime.store.evidence.values()
    )
    receipts = [
        r for r in runtime.store.receipts.values() if r.tool_name == "resolve_media"
    ]
    assert receipts
    tool = runtime.store.tool_results_by_receipt[receipts[0].search_execution_id]
    assert tool.status is ToolResultStatus.EMPTY
    assert runtime.last_gate is not None and runtime.last_gate.accepted


def test_t4_05_simple_retrieval_skips_media():
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
            case_id="case-t4-05",
            question=PERSON_Q,
            start=datetime(2023, 11, 14, 22, 0, tzinfo=UTC),
            end=datetime(2023, 11, 14, 23, 30, tzinfo=UTC),
        ),
        scenario=ScenarioConfig(
            name="t4_05",
            tool_backend="stub",
            use_model_initial_plan=True,
            use_model_follow_up=True,
            max_model_calls=3,
            max_tool_attempts=3,
            tool_scripts={
                "search_events": [
                    ok_event_result(
                        execution_id="exec-t4-05",
                        native_id="evt-t4-05",
                        label="person",
                        start=t0,
                        end=t0 + timedelta(seconds=60),
                    )
                ]
            },
        ),
        model_provider=provider,
    )
    run_investigation(runtime, thread_id="thread-t4-05")
    assert runtime.registry.invocations == ["search_events"]
    assert "resolve_media" not in runtime.registry.invocations
    assert runtime.store.evidence
    assert runtime.last_gate is not None and runtime.last_gate.accepted


def test_t4_capability_advertises_resolve_media_for_frigate():
    runtime = InvestigationRuntime.from_fixture(
        case=make_case(case_id="case-t4-caps", question=PERSON_Q),
        scenario=ScenarioConfig(
            name="t4_caps",
            tool_backend="frigate_recorded",
            frigate_recording_path=str(
                Path(__file__).parent / "fixtures" / "frigate" / "events_list_sanitized.json"
            ),
            use_model_initial_plan=False,
            use_model_follow_up=False,
        ),
    )
    cap = runtime.capabilities.for_operation("resolve_media")
    assert cap is not None
    assert cap.availability.value == "supported"


def test_t4_real_ollama_visual_request_may_select_resolve_media(tmp_path: Path):
    """Genuine Ollama: visual-evidence question may select resolve_media after search."""
    import json as _json
    import urllib.error
    import urllib.request

    from nanexus_investigation.model.provider import OllamaChatProvider

    ollama_url = "http://127.0.0.1:11434"
    model_name = "qwen3:14b"
    try:
        with urllib.request.urlopen(f"{ollama_url}/api/tags", timeout=5) as resp:
            payload = _json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, _json.JSONDecodeError) as exc:
        pytest.fail(f"T4 requires reachable local Ollama at {ollama_url}: {exc}")
    names = {m.get("name") for m in payload.get("models", []) if isinstance(m, dict)}
    if model_name not in names:
        pytest.fail(f"T4 requires installed model {model_name!r}")

    t0 = datetime(2023, 11, 14, 22, 15, tzinfo=UTC)
    event = ok_event_result(
        execution_id="exec-t4-ollama",
        native_id="evt-t4-ollama",
        label="car",
        start=t0,
        end=t0 + timedelta(seconds=20),
        camera_id="front_yard",
    )
    provider = OllamaChatProvider(model=model_name, base_url=ollama_url, timeout_s=180.0)
    runtime = InvestigationRuntime.from_fixture(
        case=make_case(
            case_id="case-t4-ollama",
            question=VISUAL_Q,
            start=datetime(2023, 11, 14, 22, 0, tzinfo=UTC),
            end=datetime(2023, 11, 14, 23, 30, tzinfo=UTC),
        ),
        scenario=ScenarioConfig(
            name="t4_ollama_media",
            tool_backend="stub",
            use_model_initial_plan=True,
            use_model_follow_up=True,
            max_model_calls=5,
            max_tool_attempts=5,
            tool_scripts={"search_events": [event]},
            media_store_root=str(tmp_path / "media"),
            enable_temporal_analysis=False,
        ),
        model_provider=provider,
    )
    payload = b"\xff\xd8\xff" + b"ollama-snap" * 50
    original_invoke = runtime.registry.invoke

    def _invoke(request: ToolRequest):
        if request.tool_name != "resolve_media":
            return original_invoke(request)
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

    runtime.registry.invoke = _invoke  # type: ignore[method-assign]
    run_investigation(runtime, thread_id="thread-t4-ollama")
    print(
        _json.dumps(
            {
                "invocations": list(runtime.registry.invocations),
                "proposals": [p.to_dict() for p in runtime.model_proposals],
                "gate": None if runtime.last_gate is None else runtime.last_gate.accepted,
                "status": runtime.case.status.value,
            }
        )
    )
    assert runtime.registry.invocations
    assert runtime.registry.invocations[0] == "search_events"
    proposed_media = any(
        getattr(p, "tool_name", None) == "resolve_media" for p in runtime.model_proposals
    )
    assert proposed_media or "resolve_media" in runtime.registry.invocations
    # Successful acquisition is ideal; invalid model-invented IDs must fail closed.
    snapshots = [
        e
        for e in runtime.store.evidence.values()
        if e.evidence_kind is EvidenceKind.SNAPSHOT
    ]
    if snapshots:
        joined = " ".join(c.statement.lower() for c in runtime.store.claims.values())
        assert "visual" in joined
        assert runtime.last_gate is not None and runtime.last_gate.accepted
    else:
        assert runtime.case.status in {
            CaseStatus.COMPLETED,
            CaseStatus.PARTIAL,
            CaseStatus.FAILED,
        }


@pytest.mark.live_frigate
def test_t4_live_snapshot_and_clip_acquisition(tmp_path: Path):
    from nanexus_investigation.vms.frigate.record import (
        config_from_env,
        live_client_ready,
        load_frigate_env,
    )
    from nanexus_investigation.vms.frigate.client import FrigateReadOnlyClient

    env = load_frigate_env()
    ready, reason = live_client_ready(env)
    if not ready:
        pytest.fail(f"live Frigate unavailable: {reason}")
    client = FrigateReadOnlyClient(config_from_env(env))
    events = json.loads(
        client.list_events(
            cameras=client.config.authorized_cameras[:1], limit=10
        ).body.decode()
    )
    target = next(
        (e for e in events if e.get("has_snapshot") and e.get("has_clip")), None
    )
    assert target is not None
    eid = target["id"]
    obs, evidence, scope, store = _source_artifacts(
        tmp_path, native_id=eid
    )
    # Align VMS/camera with live client.
    scope = InvestigationScope(
        vms_instance_id=client.config.vms_instance_id,
        camera_ids=client.config.authorized_cameras[:1],
        time_bounds=scope.time_bounds,
    )
    evidence = evidence.model_copy(
        update={
            "camera_id": target["camera"],
            "source_locator": evidence.source_locator.model_copy(
                update={
                    "vms_instance_id": client.config.vms_instance_id,
                    "native_record_id": eid,
                }
            ),
        }
    )
    obs = obs.model_copy(
        update={
            "camera_id": target["camera"],
            "source_locator": evidence.source_locator,
        }
    )
    snap = resolve_media(
        ToolRequest(
            tool_name="resolve_media",
            arguments={"media_kind": "snapshot", "evidence_id": evidence.evidence_id},
            source_id=client.config.vms_instance_id,
            camera_ids=(target["camera"],),
        ),
        evidence_by_id={evidence.evidence_id: evidence},
        observations_by_id={obs.observation_id: obs},
        scope=scope,
        case_id="case-t4-live",
        media_store=store,
        transport=client,
    )
    assert snap.status is ToolResultStatus.OK
    assert snap.data[0]["byte_count"] > 0
    assert snap.data[0]["content_type"].startswith("image/")

    clip = resolve_media(
        ToolRequest(
            tool_name="resolve_media",
            arguments={"media_kind": "clip", "evidence_id": evidence.evidence_id},
            source_id=client.config.vms_instance_id,
            camera_ids=(target["camera"],),
        ),
        evidence_by_id={evidence.evidence_id: evidence},
        observations_by_id={obs.observation_id: obs},
        scope=scope,
        case_id="case-t4-live",
        media_store=store,
        transport=client,
    )
    assert clip.status is ToolResultStatus.OK
    assert clip.data[0]["byte_count"] > 0
    assert "mp4" in clip.data[0]["content_type"]


def test_t4_recorded_media_when_private_capture_present(tmp_path: Path):
    manifest = PRIVATE_MEDIA / "manifest.json"
    if not manifest.is_file():
        pytest.skip("private recorded media capture not present under .local/")
    meta = json.loads(manifest.read_text())
    eid = meta["event_id"]
    obs, evidence, scope, store = _source_artifacts(tmp_path, native_id=eid)
    from nanexus_investigation.tools.frigate_backend import _RecordedMediaTransport

    transport = _RecordedMediaTransport(PRIVATE_MEDIA)
    result = resolve_media(
        ToolRequest(
            tool_name="resolve_media",
            arguments={"media_kind": "snapshot", "evidence_id": evidence.evidence_id},
            source_id="vms-1",
            camera_ids=("front_yard",),
        ),
        evidence_by_id={evidence.evidence_id: evidence},
        observations_by_id={obs.observation_id: obs},
        scope=scope,
        case_id="case-t4-recorded",
        media_store=store,
        transport=transport,
    )
    assert result.status is ToolResultStatus.OK
    assert result.data[0]["byte_count"] == meta["snapshot_bytes"]
