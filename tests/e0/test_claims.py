"""Deterministic claim assessment tests. G3-A §5.5."""

from datetime import datetime, timedelta, timezone

from nanexus_investigation.contracts.claim import Claim
from nanexus_investigation.contracts.enums import (
    ClaimType,
    ClaimValidationStatus,
    EvidenceAvailability,
    EvidenceKind,
    ObservationType,
    RecordType,
    SearchCoverage,
    TemporalEvaluationStatus,
    TemporalRelationType,
    ToolResultStatus,
)
from nanexus_investigation.contracts.evidence import EvidenceRef, Observation
from nanexus_investigation.contracts.identity import Interval, SourceLocator
from nanexus_investigation.contracts.scope import EvidenceGrant, InvestigationScope
from nanexus_investigation.contracts.search import SearchExecutionReceipt
from nanexus_investigation.contracts.temporal import TemporalRelation
from nanexus_investigation.contracts.tool import ToolResult
from nanexus_investigation.evidence.claims import assess_claim, observation_interval
from nanexus_investigation.evidence.time import build_interval

UTC = timezone.utc
T0 = datetime(2026, 11, 19, 20, 0, tzinfo=UTC)
T1 = datetime(2026, 11, 19, 21, 0, tzinfo=UTC)


def _scope() -> InvestigationScope:
    return InvestigationScope(
        vms_instance_id="vms-local-1",
        camera_ids=("front_yard",),
        time_bounds=Interval(start_utc=T0, end_utc=T1),
    )


def _locator(native_id: str = "evt-1") -> SourceLocator:
    return SourceLocator(
        vms_instance_id="vms-local-1",
        record_type=RecordType.EVENT,
        native_record_id=native_id,
    )


def _observation(obs_id: str, native_id: str, start_s: int, end_s: int) -> Observation:
    return Observation(
        observation_id=obs_id,
        source_locator=_locator(native_id),
        source_revision="rev-1",
        observation_type=ObservationType.DETECTION_EVENT,
        camera_id="front_yard",
        start_utc=T0 + timedelta(seconds=start_s),
        end_utc=T0 + timedelta(seconds=end_s),
        retrieved_at=T0,
        provenance="adapter:test/search_events",
    )


def _evidence(evidence_id: str, native_id: str, start_s: int, end_s: int) -> EvidenceRef:
    return EvidenceRef(
        evidence_id=evidence_id,
        source_locator=_locator(native_id),
        evidence_kind=EvidenceKind.EVENT_METADATA,
        camera_id="front_yard",
        interval=Interval(
            start_utc=T0 + timedelta(seconds=start_s),
            end_utc=T0 + timedelta(seconds=end_s),
        ),
        retrieval_spec={"operation": "get_event"},
        availability=EvidenceAvailability.AVAILABLE,
        provenance="adapter:test/get_event",
    )


def test_valid_source_observation_claim():
    evidence = _evidence("ev-1", "evt-1", 0, 10)
    observation = _observation("obs-1", "evt-1", 0, 10)
    grant = EvidenceGrant(
        case_id="case-1",
        effective_scope=_scope(),
        evidence_ids=frozenset({"ev-1"}),
    )
    claim = Claim(
        claim_id="claim-1",
        statement="Frigate returned a person detection.",
        claim_type=ClaimType.SOURCE_OBSERVATION,
        evidence_ids=("ev-1",),
        producer="rule:source-observation-v1",
        validation_status=ClaimValidationStatus.QUALIFIED,
    )
    result = assess_claim(
        claim,
        effective_scope=_scope(),
        evidence_by_id={"ev-1": evidence},
        observations_by_id={"obs-1": observation},
        grant=grant,
    )
    assert result.status is ClaimValidationStatus.SUPPORTED


def test_missing_citations_unsupported():
    claim = Claim(
        claim_id="claim-2",
        statement="unsupported guess",
        claim_type=ClaimType.MODEL_INTERPRETATION,
        producer="model:stub",
        validation_status=ClaimValidationStatus.UNSUPPORTED,
    )
    result = assess_claim(
        claim,
        effective_scope=_scope(),
        evidence_by_id={},
        grant=EvidenceGrant(
            case_id="case-1",
            effective_scope=_scope(),
            evidence_ids=frozenset(),
        ),
    )
    assert result.status is ClaimValidationStatus.UNSUPPORTED


def test_fabricated_evidence_reference():
    claim = Claim(
        claim_id="claim-3",
        statement="cites missing evidence",
        claim_type=ClaimType.SOURCE_OBSERVATION,
        evidence_ids=("ev-missing",),
        producer="model:stub",
        validation_status=ClaimValidationStatus.SUPPORTED,
    )
    result = assess_claim(
        claim,
        effective_scope=_scope(),
        evidence_by_id={},
        grant=EvidenceGrant(
            case_id="case-1",
            effective_scope=_scope(),
            evidence_ids=frozenset(),
        ),
    )
    assert result.status is ClaimValidationStatus.UNSUPPORTED
    assert any("fabricated or missing" in item for item in result.reasons)


def test_contradictory_temporal_claim():
    obs_a = _observation("obs-a", "evt-a", 0, 10)
    obs_b = _observation("obs-b", "evt-b", 20, 30)
    relation = TemporalRelation(
        relation_id="rel-1",
        left_observation_id="obs-a",
        right_observation_id="obs-b",
        relation_type=TemporalRelationType.OVERLAPS,
        evaluation_status=TemporalEvaluationStatus.ESTABLISHED,
        camera_relation="same_camera",
        time_basis="test",
        calculation_provenance="fixture",
    )
    claim = Claim(
        claim_id="claim-4",
        statement="A overlaps B",
        claim_type=ClaimType.COMPUTED_RELATION,
        relation_ids=("rel-1",),
        evidence_ids=("ev-a",),
        producer="rule:temporal-v1",
        validation_status=ClaimValidationStatus.SUPPORTED,
    )
    result = assess_claim(
        claim,
        effective_scope=_scope(),
        evidence_by_id={
            "ev-a": _evidence("ev-a", "evt-a", 0, 10),
        },
        observations_by_id={"obs-a": obs_a, "obs-b": obs_b},
        relations_by_id={"rel-1": relation},
        observation_intervals={
            "obs-a": observation_interval(obs_a),
            "obs-b": observation_interval(obs_b),
        },
        grant=EvidenceGrant(
            case_id="case-1",
            effective_scope=_scope(),
            evidence_ids=frozenset({"ev-a"}),
        ),
    )
    assert result.status is ClaimValidationStatus.UNSUPPORTED
    assert any("contradicted by recalculation" in item for item in result.reasons)


def test_indeterminate_temporal_recomputation_is_not_contradiction():
    obs_a = _observation("obs-a", "evt-a", 0, 10)
    obs_b = Observation(
        observation_id="obs-b",
        source_locator=_locator("evt-b"),
        source_revision="rev-1",
        observation_type=ObservationType.DETECTION_EVENT,
        camera_id="front_yard",
        start_utc=T0 + timedelta(seconds=20),
        end_utc=None,
        retrieved_at=T0,
        provenance="adapter:test/search_events",
    )
    relation = TemporalRelation(
        relation_id="rel-open",
        left_observation_id="obs-a",
        right_observation_id="obs-b",
        relation_type=TemporalRelationType.BEFORE,
        evaluation_status=TemporalEvaluationStatus.ESTABLISHED,
        camera_relation="same_camera",
        time_basis="test",
        calculation_provenance="fixture",
    )
    claim = Claim(
        claim_id="claim-open",
        statement="A before B",
        claim_type=ClaimType.COMPUTED_RELATION,
        relation_ids=("rel-open",),
        evidence_ids=("ev-a",),
        producer="rule:temporal-v1",
        validation_status=ClaimValidationStatus.SUPPORTED,
    )
    result = assess_claim(
        claim,
        effective_scope=_scope(),
        evidence_by_id={"ev-a": _evidence("ev-a", "evt-a", 0, 10)},
        observations_by_id={"obs-a": obs_a, "obs-b": obs_b},
        relations_by_id={"rel-open": relation},
        observation_intervals={
            "obs-a": observation_interval(obs_a),
            "obs-b": observation_interval(obs_b),
        },
        grant=EvidenceGrant(
            case_id="case-1",
            effective_scope=_scope(),
            evidence_ids=frozenset({"ev-a"}),
        ),
    )
    assert result.status is ClaimValidationStatus.UNVERIFIABLE
    assert any("cannot be reconfirmed" in item for item in result.reasons)
    assert not any("contradicted" in item for item in result.reasons)


def test_model_supported_label_is_not_independent_verification():
    evidence = _evidence("ev-1", "evt-1", 0, 10)
    observation = _observation("obs-1", "evt-1", 0, 10)
    claim = Claim(
        claim_id="claim-5",
        statement="The same person entered a vehicle.",
        claim_type=ClaimType.MODEL_INTERPRETATION,
        evidence_ids=("ev-1",),
        producer="model:stub",
        validation_status=ClaimValidationStatus.SUPPORTED,
    )
    result = assess_claim(
        claim,
        effective_scope=_scope(),
        evidence_by_id={"ev-1": evidence},
        observations_by_id={"obs-1": observation},
        grant=EvidenceGrant(
            case_id="case-1",
            effective_scope=_scope(),
            evidence_ids=frozenset({"ev-1"}),
        ),
    )
    assert result.status is ClaimValidationStatus.UNVERIFIABLE
    assert any(
        "not independent verification" in item or "cannot prove" in item
        for item in result.reasons
    )


def test_near_alone_cannot_prove_identity():
    obs_a = _observation("obs-a", "evt-a", 0, 10)
    obs_b = _observation("obs-b", "evt-b", 12, 20)
    relation = TemporalRelation(
        relation_id="rel-near",
        left_observation_id="obs-a",
        right_observation_id="obs-b",
        relation_type=TemporalRelationType.NEAR,
        evaluation_status=TemporalEvaluationStatus.ESTABLISHED,
        window_seconds=5,
        camera_relation="same_camera",
        time_basis="test",
        calculation_provenance="fixture",
    )
    claim = Claim(
        claim_id="claim-6",
        statement="These detections are the same person.",
        claim_type=ClaimType.MODEL_INTERPRETATION,
        relation_ids=("rel-near",),
        evidence_ids=("ev-a",),
        producer="model:stub",
        validation_status=ClaimValidationStatus.SUPPORTED,
    )
    result = assess_claim(
        claim,
        effective_scope=_scope(),
        evidence_by_id={"ev-a": _evidence("ev-a", "evt-a", 0, 10)},
        observations_by_id={"obs-a": obs_a, "obs-b": obs_b},
        relations_by_id={"rel-near": relation},
        observation_intervals={
            "obs-a": build_interval(obs_a.start_utc, obs_a.end_utc),
            "obs-b": build_interval(obs_b.start_utc, obs_b.end_utc),
        },
        grant=EvidenceGrant(
            case_id="case-1",
            effective_scope=_scope(),
            evidence_ids=frozenset({"ev-a"}),
        ),
    )
    assert result.status is ClaimValidationStatus.UNSUPPORTED
    assert any("near alone cannot prove" in item for item in result.reasons)


def test_qualified_empty_search_claim():
    receipt = SearchExecutionReceipt(
        search_execution_id="search-1",
        tool_name="search_events",
        source_id="vms-local-1",
        camera_ids=("front_yard",),
        time_bounds=Interval(start_utc=T0, end_utc=T1),
        retrieved_at=T0,
        pages_fetched=1,
        match_count=0,
        coverage=SearchCoverage.EXHAUSTED,
    )
    tool_result = ToolResult(
        status=ToolResultStatus.EMPTY,
        execution_id="exec-1",
        data=[],
        coverage=SearchCoverage.EXHAUSTED,
        retrieved_at=T0,
    )
    claim = Claim(
        claim_id="claim-7",
        statement="The completed query returned no matching records.",
        claim_type=ClaimType.QUALIFIED_SEARCH_RESULT,
        search_receipt_ids=("search-1",),
        producer="rule:search-receipt-v1",
        validation_status=ClaimValidationStatus.QUALIFIED,
    )
    result = assess_claim(
        claim,
        effective_scope=_scope(),
        evidence_by_id={},
        receipts_by_id={"search-1": receipt},
        tool_results_by_receipt_id={"search-1": tool_result},
        grant=EvidenceGrant(
            case_id="case-1",
            effective_scope=_scope(),
            evidence_ids=frozenset(),
        ),
        require_grant_membership=False,
    )
    assert result.status is ClaimValidationStatus.QUALIFIED


def test_truncated_empty_search_cannot_support_no_match_claim():
    receipt = SearchExecutionReceipt(
        search_execution_id="search-2",
        tool_name="search_events",
        source_id="vms-local-1",
        camera_ids=("front_yard",),
        time_bounds=Interval(start_utc=T0, end_utc=T1),
        retrieved_at=T0,
        pages_fetched=1,
        match_count=0,
        coverage=SearchCoverage.TRUNCATED,
    )
    tool_result = ToolResult(
        status=ToolResultStatus.EMPTY,
        execution_id="exec-2",
        data=[],
        coverage=SearchCoverage.TRUNCATED,
        retrieved_at=T0,
    )
    claim = Claim(
        claim_id="claim-8",
        statement="No matching records for the completed query.",
        claim_type=ClaimType.QUALIFIED_SEARCH_RESULT,
        search_receipt_ids=("search-2",),
        producer="rule:search-receipt-v1",
        validation_status=ClaimValidationStatus.SUPPORTED,
    )
    result = assess_claim(
        claim,
        effective_scope=_scope(),
        evidence_by_id={},
        receipts_by_id={"search-2": receipt},
        tool_results_by_receipt_id={"search-2": tool_result},
        require_grant_membership=False,
    )
    assert result.status is ClaimValidationStatus.UNSUPPORTED
