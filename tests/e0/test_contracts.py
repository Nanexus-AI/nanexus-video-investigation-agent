"""Deterministic contract and validation tests for G4-V1-B."""

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest
from pydantic import ValidationError

from nanexus_investigation.contracts import (
    CapabilityAvailability,
    CapabilitySnapshot,
    CaseStatus,
    Claim,
    ClaimType,
    ClaimValidationStatus,
    EvidenceAvailability,
    EvidenceGrant,
    EvidenceKind,
    EvidenceRef,
    Interval,
    InvestigationCase,
    InvestigationReport,
    InvestigationScope,
    Observation,
    ObservationType,
    OperationCapability,
    RecordType,
    SearchCoverage,
    SearchExecutionReceipt,
    SourceLocator,
    TemporalEvaluationStatus,
    TemporalRelation,
    TemporalRelationType,
    TimeContext,
    ToolError,
    ToolErrorCode,
    ToolRequest,
    ToolResult,
    ToolResultStatus,
    apply_requested_scope,
    validate_case_transition,
)

UTC = timezone.utc
T0 = datetime(2026, 11, 19, 20, 0, tzinfo=UTC)
T1 = datetime(2026, 11, 19, 21, 0, tzinfo=UTC)


def _interval() -> Interval:
    return Interval(start_utc=T0, end_utc=T1)


def _scope(*, cameras: tuple[str, ...] = ("front_yard",)) -> InvestigationScope:
    return InvestigationScope(
        vms_instance_id="vms-local-1",
        camera_ids=cameras,
        time_bounds=_interval(),
    )


def _time_context() -> TimeContext:
    return TimeContext(
        timezone="America/New_York",
        start_utc=T0,
        end_utc=T1,
        original_expression="yesterday evening",
    )


def _capability() -> CapabilitySnapshot:
    return CapabilitySnapshot(
        vms_instance_id="vms-local-1",
        operations=(
            OperationCapability(
                operation="search_events",
                availability=CapabilityAvailability.SUPPORTED,
                authorized_camera_ids=frozenset({"front_yard"}),
            ),
            OperationCapability(
                operation="semantic_search",
                availability=CapabilityAvailability.UNSUPPORTED,
            ),
        ),
    )


def _case(**overrides: Any) -> InvestigationCase:
    payload = {
        "case_id": "case-1",
        "question": "Was a person detected at the front yard?",
        "requested_scope": _scope(cameras=("front_yard", "driveway")),
        "effective_scope": _scope(cameras=("front_yard",)),
        "time_context": _time_context(),
        "status": CaseStatus.CREATED,
        "policy_ref": "policy/local-read-only-v1",
        "created_at": T0,
        "capability_snapshot": _capability(),
    }
    payload.update(overrides)
    return InvestigationCase(**payload)


def _source_locator() -> SourceLocator:
    return SourceLocator(
        vms_instance_id="vms-local-1",
        record_type=RecordType.EVENT,
        native_record_id="evt-1001",
    )


def _round_trip(model: Any) -> Any:
    return type(model).model_validate_json(model.model_dump_json())


def test_investigation_case_round_trip():
    case = _case()
    restored = _round_trip(case)
    assert restored == case
    assert restored.status == CaseStatus.CREATED
    assert restored.requested_scope.camera_ids == ("front_yard", "driveway")
    assert restored.effective_scope.camera_ids == ("front_yard",)


def test_unknown_case_status_rejected():
    with pytest.raises(ValidationError):
        _case(status="investigating")


@pytest.mark.parametrize(
    ("current", "new"),
    [
        (CaseStatus.CREATED, CaseStatus.RUNNING),
        (CaseStatus.RUNNING, CaseStatus.AWAITING_INPUT),
        (CaseStatus.AWAITING_INPUT, CaseStatus.RUNNING),
        (CaseStatus.RUNNING, CaseStatus.COMPLETED),
        (CaseStatus.RUNNING, CaseStatus.PARTIAL),
        (CaseStatus.RUNNING, CaseStatus.FAILED),
        (CaseStatus.RUNNING, CaseStatus.CANCELLED),
        (CaseStatus.AWAITING_INPUT, CaseStatus.CANCELLED),
    ],
)
def test_legal_case_transitions(current: CaseStatus, new: CaseStatus):
    validate_case_transition(current, new)
    case = _case(status=current).transition_to(new)
    assert case.status == new


@pytest.mark.parametrize(
    ("current", "new"),
    [
        (CaseStatus.COMPLETED, CaseStatus.RUNNING),
        (CaseStatus.PARTIAL, CaseStatus.RUNNING),
        (CaseStatus.FAILED, CaseStatus.RUNNING),
        (CaseStatus.CANCELLED, CaseStatus.RUNNING),
        (CaseStatus.CREATED, CaseStatus.COMPLETED),
        (CaseStatus.AWAITING_INPUT, CaseStatus.COMPLETED),
    ],
)
def test_illegal_case_transitions_rejected(current: CaseStatus, new: CaseStatus):
    with pytest.raises(ValueError):
        validate_case_transition(current, new)


def test_requested_and_effective_scope_remain_distinct():
    case = _case()
    assert case.requested_scope != case.effective_scope
    assert set(case.effective_scope.camera_ids).issubset(
        set(case.requested_scope.camera_ids)
    )


def test_model_request_cannot_expand_authorized_scope():
    authorized = _scope(cameras=("front_yard",))
    requested = _scope(cameras=("front_yard", "back_yard"))
    with pytest.raises(ValueError, match="expands cameras"):
        apply_requested_scope(
            authorized_scope=authorized,
            requested_scope=requested,
        )


def test_model_request_cannot_expand_time_bounds():
    authorized = _scope()
    requested = InvestigationScope(
        vms_instance_id="vms-local-1",
        camera_ids=("front_yard",),
        time_bounds=Interval(
            start_utc=T0 - timedelta(hours=1),
            end_utc=T1 + timedelta(hours=1),
        ),
    )
    with pytest.raises(ValueError, match="time bounds"):
        apply_requested_scope(
            authorized_scope=authorized,
            requested_scope=requested,
        )


def test_evidence_grant_admits_only_listed_ids():
    grant = EvidenceGrant(
        case_id="case-1",
        effective_scope=_scope(),
        evidence_ids=frozenset({"ev-1"}),
    )
    assert grant.admits("ev-1")
    assert not grant.admits("ev-plausible-looking")
    restored = _round_trip(grant)
    assert restored.evidence_ids == frozenset({"ev-1"})


def test_evidence_grant_construction_is_not_authorization():
    # Placing an ID in a grant record is membership data only. It does not
    # prove VMS permission or Evidence Engine validation.
    fabricated = EvidenceGrant(
        case_id="case-1",
        effective_scope=_scope(),
        evidence_ids=frozenset({"ev-never-fetched"}),
    )
    assert fabricated.admits("ev-never-fetched")
    assert fabricated.admits("ev-never-fetched") is (
        "ev-never-fetched" in fabricated.evidence_ids
    )


def test_capability_snapshot_keeps_availability_and_permission_distinct():
    snapshot = _capability()
    search = snapshot.for_operation("search_events")
    assert search is not None
    assert search.availability == CapabilityAvailability.SUPPORTED
    assert search.authorized_camera_ids == frozenset({"front_yard"})
    semantic = snapshot.for_operation("semantic_search")
    assert semantic is not None
    assert semantic.availability == CapabilityAvailability.UNSUPPORTED
    restored = _round_trip(snapshot)
    assert restored == snapshot


def test_unknown_capability_availability_rejected():
    with pytest.raises(ValidationError):
        OperationCapability(
            operation="search_events",
            availability="maybe",
        )


def test_tool_result_status_vocabularies_and_semantics():
    ok = ToolResult(
        status=ToolResultStatus.OK,
        execution_id="exec-1",
        data={"count": 1},
        retrieved_at=T0,
        coverage=SearchCoverage.EXHAUSTED,
    )
    empty = ToolResult(
        status=ToolResultStatus.EMPTY,
        execution_id="exec-2",
        data=[],
        retrieved_at=T0,
        coverage=SearchCoverage.EXHAUSTED,
    )
    partial = ToolResult(
        status=ToolResultStatus.PARTIAL,
        execution_id="exec-3",
        data={"items": [1]},
        retrieved_at=T0,
        coverage=SearchCoverage.TRUNCATED,
    )
    error = ToolResult(
        status=ToolResultStatus.ERROR,
        execution_id="exec-4",
        error=ToolError(
            code=ToolErrorCode.TIMEOUT,
            message="source timed out",
        ),
        retrieved_at=T0,
    )
    unsupported = ToolResult(
        status=ToolResultStatus.UNSUPPORTED,
        execution_id="exec-5",
        error=ToolError(
            code=ToolErrorCode.DISABLED_CAPABILITY,
            message="semantic search disabled",
        ),
    )
    for result in (ok, empty, partial, error, unsupported):
        assert _round_trip(result) == result

    with pytest.raises(ValidationError):
        ToolResult(status="SUCCESS", execution_id="exec-x")
    with pytest.raises(ValidationError):
        ToolResult(
            status=ToolResultStatus.EMPTY,
            execution_id="exec-bad",
            data={"items": [1]},
        )
    with pytest.raises(ValidationError):
        ToolResult(status=ToolResultStatus.ERROR, execution_id="exec-bad")


def test_tool_request_rejects_credential_arguments():
    with pytest.raises(ValidationError):
        ToolRequest(
            tool_name="search_events",
            arguments={"api_key": "secret-value"},
        )


def test_search_execution_receipt_coverage_and_round_trip():
    receipt = SearchExecutionReceipt(
        search_execution_id="search-1",
        tool_name="search_events",
        source_id="vms-local-1",
        camera_ids=("front_yard",),
        time_bounds=_interval(),
        filters={"label": "person"},
        retrieved_at=T0,
        pages_fetched=1,
        page_limit=100,
        match_count=0,
        coverage=SearchCoverage.EXHAUSTED,
        consistency_limitations=(
            "Empty exhausted search describes the completed query only."
        ),
    )
    assert _round_trip(receipt) == receipt
    with pytest.raises(ValidationError):
        SearchExecutionReceipt.model_validate(
            {
                **receipt.model_dump(mode="json"),
                "coverage": "complete",
            }
        )


def test_observation_and_evidence_preserve_source_identity():
    observation = Observation(
        observation_id="obs-1",
        source_locator=_source_locator(),
        source_revision="meta-rev-1",
        observation_type=ObservationType.DETECTION_EVENT,
        camera_id="front_yard",
        start_utc=T0,
        end_utc=T0 + timedelta(seconds=12),
        labels=("person",),
        zones=("driveway",),
        retrieved_at=T0,
        provenance="adapter:frigate-read/search_events",
    )
    evidence = EvidenceRef(
        evidence_id="ev-1",
        source_locator=_source_locator(),
        evidence_kind=EvidenceKind.EVENT_METADATA,
        camera_id="front_yard",
        interval=Interval(start_utc=T0, end_utc=T0 + timedelta(seconds=12)),
        retrieval_spec={"operation": "get_event", "native_record_id": "evt-1001"},
        availability=EvidenceAvailability.AVAILABLE,
        checked_at=T0,
        metadata_fingerprint="fp-1",
        provenance="adapter:frigate-read/get_event",
    )
    restored_obs = _round_trip(observation)
    restored_ev = _round_trip(evidence)
    assert restored_obs.source_locator.vms_instance_id == "vms-local-1"
    assert restored_obs.source_locator.record_type == RecordType.EVENT
    assert restored_obs.source_locator.native_record_id == "evt-1001"
    assert restored_ev.source_locator == observation.source_locator


def test_malformed_evidence_references_rejected():
    with pytest.raises(ValidationError):
        SourceLocator(
            vms_instance_id="vms-local-1",
            record_type=RecordType.EVENT,
            native_record_id="https://example.invalid/event/1",
        )
    with pytest.raises(ValidationError):
        EvidenceRef(
            evidence_id="ev-1",
            source_locator=_source_locator(),
            evidence_kind=EvidenceKind.CLIP,
            camera_id="front_yard",
            retrieval_spec={"url": "https://example.invalid/clip.mp4"},
            provenance="adapter:bad",
        )
    with pytest.raises(ValidationError):
        EvidenceRef(
            evidence_id="ev-1",
            source_locator=_source_locator(),
            evidence_kind=EvidenceKind.CLIP,
            camera_id="front_yard",
            content_sha256="not-a-hash",
            provenance="adapter:bad",
        )
    with pytest.raises(ValidationError):
        EvidenceRef(
            evidence_id="ev-1",
            source_locator=_source_locator(),
            evidence_kind=EvidenceKind.CLIP,
            camera_id="front_yard",
            content_sha256="a" * 64,
            provenance="adapter:bad",
        )


def test_content_hash_only_when_bytes_hashed():
    evidence = EvidenceRef(
        evidence_id="ev-2",
        source_locator=_source_locator(),
        evidence_kind=EvidenceKind.SNAPSHOT,
        camera_id="front_yard",
        retrieval_spec={"operation": "resolve_media"},
        content_sha256="ab" * 32,
        content_type="image/jpeg",
        provenance="adapter:frigate-read/resolve_media",
    )
    assert evidence.content_sha256 is not None
    metadata_only = EvidenceRef(
        evidence_id="ev-3",
        source_locator=_source_locator(),
        evidence_kind=EvidenceKind.EVENT_METADATA,
        camera_id="front_yard",
        retrieval_spec={"operation": "get_event"},
        provenance="adapter:frigate-read/get_event",
    )
    assert metadata_only.content_sha256 is None


def test_content_type_is_not_integrity_proof():
    typed_only = EvidenceRef(
        evidence_id="ev-typed",
        source_locator=_source_locator(),
        evidence_kind=EvidenceKind.SNAPSHOT,
        camera_id="front_yard",
        retrieval_spec={"operation": "resolve_media"},
        content_type="image/jpeg",
        provenance="adapter:frigate-read/resolve_media",
    )
    assert typed_only.content_type == "image/jpeg"
    assert typed_only.content_sha256 is None


def test_naive_datetimes_are_rejected():
    with pytest.raises(ValidationError):
        Interval(
            start_utc=datetime(2026, 11, 19, 20, 0),
            end_utc=datetime(2026, 11, 19, 21, 0),
        )
    with pytest.raises(ValidationError):
        TimeContext(
            timezone="America/New_York",
            start_utc=datetime(2026, 11, 19, 20, 0),
            end_utc=datetime(2026, 11, 19, 21, 0),
        )
    with pytest.raises(ValidationError):
        TimeContext(
            timezone="Not/A_Zone",
            start_utc=T0,
            end_utc=T1,
        )


def test_temporal_relation_contract_only():
    relation = TemporalRelation(
        relation_id="rel-1",
        left_observation_id="obs-1",
        right_observation_id="obs-2",
        relation_type=TemporalRelationType.BEFORE,
        evaluation_status=TemporalEvaluationStatus.ESTABLISHED,
        camera_relation="same_camera",
        time_basis="completed_intervals",
        calculation_provenance="temporal-v0",
    )
    assert _round_trip(relation) == relation
    with pytest.raises(ValidationError):
        TemporalRelation(
            relation_id="rel-2",
            left_observation_id="obs-1",
            right_observation_id="obs-2",
            relation_type="during",
            evaluation_status=TemporalEvaluationStatus.ESTABLISHED,
            camera_relation="same_camera",
            time_basis="completed_intervals",
            calculation_provenance="temporal-v0",
        )


def test_claim_and_report_vocabularies():
    claim = Claim(
        claim_id="claim-1",
        statement="Frigate returned a person detection at 20:00 UTC.",
        claim_type=ClaimType.SOURCE_OBSERVATION,
        evidence_ids=("ev-1",),
        producer="rule:source-observation-v1",
        validation_status=ClaimValidationStatus.SUPPORTED,
        validated_at=T0,
    )
    report = InvestigationReport(
        report_id="report-1",
        case_id="case-1",
        generated_at=T0,
        scope_summary="front_yard within authorized window",
        execution_status=CaseStatus.COMPLETED,
        timeline=("obs-1",),
        claims=("claim-1",),
        evidence_index=("ev-1",),
        limitations=(),
        unresolved_questions=(),
        validation_summary="all cited claims supported",
        report_version="0.1",
    )
    assert _round_trip(claim) == claim
    assert _round_trip(report) == report

    with pytest.raises(ValidationError):
        Claim(
            claim_id="claim-2",
            statement="guess",
            claim_type="speculation",
            producer="model:stub",
            validation_status=ClaimValidationStatus.SUPPORTED,
            evidence_ids=("ev-1",),
        )
    with pytest.raises(ValidationError):
        Claim(
            claim_id="claim-3",
            statement="guess",
            claim_type=ClaimType.MODEL_INTERPRETATION,
            producer="model:stub",
            validation_status="accepted",
        )
    with pytest.raises(ValidationError):
        Claim(
            claim_id="claim-4",
            statement="supported without citations",
            claim_type=ClaimType.SOURCE_OBSERVATION,
            producer="rule:source-observation-v1",
            validation_status=ClaimValidationStatus.SUPPORTED,
        )
    with pytest.raises(ValidationError):
        InvestigationReport(
            report_id="report-2",
            case_id="case-1",
            generated_at=T0,
            scope_summary="scope",
            execution_status=CaseStatus.RUNNING,
            validation_summary="invalid",
            report_version="0.1",
        )


def test_supported_claim_schema_is_not_evidence_assessment():
    # Schema acceptance of validation_status=supported with citation IDs
    # present does not mean the Evidence Engine or Output Gate ran.
    claim = Claim(
        claim_id="claim-schema-only",
        statement="A model-written supported label with an unverified citation.",
        claim_type=ClaimType.MODEL_INTERPRETATION,
        evidence_ids=("ev-unverified",),
        producer="model:stub",
        validation_status=ClaimValidationStatus.SUPPORTED,
        limitations=("schema-valid only; evidence not assessed here",),
        validated_at=T0,
    )
    assert claim.validation_status == ClaimValidationStatus.SUPPORTED
    assert claim.evidence_ids == ("ev-unverified",)
    # No grant lookup, durable store check, or Output Gate exists in this layer.


def test_canonical_models_forbid_extra_and_secret_fields():
    with pytest.raises(ValidationError):
        InvestigationCase.model_validate(
            {
                **_case().model_dump(mode="json"),
                "api_token": "should-not-exist",
            }
        )
    with pytest.raises(ValidationError):
        EvidenceRef.model_validate(
            {
                **EvidenceRef(
                    evidence_id="ev-1",
                    source_locator=_source_locator(),
                    evidence_kind=EvidenceKind.EVENT_METADATA,
                    camera_id="front_yard",
                    provenance="adapter:frigate-read/get_event",
                ).model_dump(mode="json"),
                "media_bytes": b"raw",
            }
        )


def test_no_raw_media_or_credential_fields_on_canonical_models():
    forbidden = {
        "password",
        "token",
        "api_key",
        "secret",
        "credential",
        "credentials",
        "media_bytes",
        "raw_bytes",
        "video_bytes",
        "image_bytes",
    }
    models = (
        InvestigationCase,
        InvestigationScope,
        EvidenceGrant,
        CapabilitySnapshot,
        ToolRequest,
        ToolResult,
        SearchExecutionReceipt,
        Observation,
        EvidenceRef,
        TemporalRelation,
        Claim,
        InvestigationReport,
    )
    for model in models:
        field_names = set(model.model_fields)
        assert field_names.isdisjoint(forbidden)
