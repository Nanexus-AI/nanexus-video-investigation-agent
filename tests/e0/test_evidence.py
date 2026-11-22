"""Evidence provenance and search-coverage tests."""

from datetime import datetime, timedelta, timezone

import pytest

from nanexus_investigation.contracts.enums import (
    EvidenceAvailability,
    EvidenceKind,
    ObservationType,
    RecordType,
    SearchCoverage,
    ToolErrorCode,
    ToolResultStatus,
)
from nanexus_investigation.contracts.evidence import EvidenceRef, Observation
from nanexus_investigation.contracts.identity import Interval, SourceLocator
from nanexus_investigation.contracts.scope import EvidenceGrant, InvestigationScope
from nanexus_investigation.contracts.search import SearchExecutionReceipt
from nanexus_investigation.contracts.tool import ToolError, ToolResult
from nanexus_investigation.evidence.provenance import validate_evidence_against_scope
from nanexus_investigation.evidence.search_coverage import (
    interpret_search,
    rejects_real_world_absence,
)

UTC = timezone.utc
T0 = datetime(2026, 11, 19, 20, 0, tzinfo=UTC)
T1 = datetime(2026, 11, 19, 21, 0, tzinfo=UTC)


def _scope(**kwargs) -> InvestigationScope:
    payload = {
        "vms_instance_id": "vms-local-1",
        "camera_ids": ("front_yard",),
        "time_bounds": Interval(start_utc=T0, end_utc=T1),
    }
    payload.update(kwargs)
    return InvestigationScope(**payload)


def _locator(**kwargs) -> SourceLocator:
    payload = {
        "vms_instance_id": "vms-local-1",
        "record_type": RecordType.EVENT,
        "native_record_id": "evt-1",
    }
    payload.update(kwargs)
    return SourceLocator(**payload)


def _observation(**kwargs) -> Observation:
    payload = {
        "observation_id": "obs-1",
        "source_locator": _locator(),
        "source_revision": "rev-1",
        "observation_type": ObservationType.DETECTION_EVENT,
        "camera_id": "front_yard",
        "start_utc": T0,
        "end_utc": T0 + timedelta(seconds=10),
        "retrieved_at": T0,
        "provenance": "adapter:test/search_events",
    }
    payload.update(kwargs)
    return Observation(**payload)


def _evidence(**kwargs) -> EvidenceRef:
    payload = {
        "evidence_id": "ev-1",
        "source_locator": _locator(),
        "evidence_kind": EvidenceKind.EVENT_METADATA,
        "camera_id": "front_yard",
        "interval": Interval(start_utc=T0, end_utc=T0 + timedelta(seconds=10)),
        "retrieval_spec": {"operation": "get_event"},
        "availability": EvidenceAvailability.AVAILABLE,
        "provenance": "adapter:test/get_event",
    }
    payload.update(kwargs)
    return EvidenceRef(**payload)


def test_valid_source_identity_and_reference():
    evidence = _evidence()
    observation = _observation()
    result = validate_evidence_against_scope(
        evidence=evidence,
        effective_scope=_scope(),
        observations={"obs-1": observation},
        grant=EvidenceGrant(
            case_id="case-1",
            effective_scope=_scope(),
            evidence_ids=frozenset({"ev-1"}),
        ),
        require_grant_membership=True,
        expected_revision="rev-1",
    )
    assert result.accepted is True


def test_missing_observation_rejected():
    result = validate_evidence_against_scope(
        evidence=_evidence(),
        effective_scope=_scope(),
        observations={},
    )
    assert result.accepted is False
    assert any("no observation resolves" in item for item in result.reasons)


def test_camera_mismatch_rejected():
    result = validate_evidence_against_scope(
        evidence=_evidence(camera_id="back_yard"),
        effective_scope=_scope(),
        observations={"obs-1": _observation()},
    )
    assert result.accepted is False
    assert any("camera_id is outside" in item for item in result.reasons)


def test_vms_instance_mismatch_rejected():
    result = validate_evidence_against_scope(
        evidence=_evidence(source_locator=_locator(vms_instance_id="other-vms")),
        effective_scope=_scope(),
        observations={"obs-1": _observation()},
    )
    assert result.accepted is False
    assert any("vms_instance_id" in item for item in result.reasons)


def test_time_window_scope_rejection():
    late = Interval(
        start_utc=T1,
        end_utc=T1 + timedelta(seconds=10),
    )
    result = validate_evidence_against_scope(
        evidence=_evidence(interval=late),
        effective_scope=_scope(),
        observations={"obs-1": _observation()},
    )
    assert result.accepted is False
    assert any("outside effective scope window" in item for item in result.reasons)


def test_invalid_grant_membership():
    result = validate_evidence_against_scope(
        evidence=_evidence(),
        effective_scope=_scope(),
        observations={"obs-1": _observation()},
        grant=EvidenceGrant(
            case_id="case-1",
            effective_scope=_scope(),
            evidence_ids=frozenset({"ev-other"}),
        ),
        require_grant_membership=True,
    )
    assert result.accepted is False
    assert any("absent from the supplied grant" in item for item in result.reasons)


def test_syntactic_grant_is_not_live_authorization():
    # Constructing a grant with an ID does not itself authorize anything beyond
    # the explicit membership check requested by the caller.
    grant = EvidenceGrant(
        case_id="case-1",
        effective_scope=_scope(),
        evidence_ids=frozenset({"ev-never-fetched"}),
    )
    assert grant.admits("ev-never-fetched")
    result = validate_evidence_against_scope(
        evidence=_evidence(evidence_id="ev-never-fetched"),
        effective_scope=_scope(),
        observations={},
        grant=grant,
        require_grant_membership=True,
    )
    assert result.accepted is False
    assert any("no observation resolves" in item for item in result.reasons)


def test_revision_inconsistency():
    result = validate_evidence_against_scope(
        evidence=_evidence(),
        effective_scope=_scope(),
        observations={"obs-1": _observation(source_revision="rev-2")},
        expected_revision="rev-1",
    )
    assert result.accepted is False
    assert any("revision" in item for item in result.reasons)


def _receipt(**kwargs) -> SearchExecutionReceipt:
    payload = {
        "search_execution_id": "search-1",
        "tool_name": "search_events",
        "source_id": "vms-local-1",
        "camera_ids": ("front_yard",),
        "time_bounds": Interval(start_utc=T0, end_utc=T1),
        "retrieved_at": T0,
        "pages_fetched": 1,
        "match_count": 0,
        "coverage": SearchCoverage.EXHAUSTED,
    }
    payload.update(kwargs)
    return SearchExecutionReceipt(**payload)


def test_exhausted_versus_truncated_empty_search():
    exhausted = interpret_search(
        ToolResult(
            status=ToolResultStatus.EMPTY,
            execution_id="exec-1",
            data=[],
            coverage=SearchCoverage.EXHAUSTED,
            retrieved_at=T0,
        ),
        _receipt(coverage=SearchCoverage.EXHAUSTED),
    )
    truncated = interpret_search(
        ToolResult(
            status=ToolResultStatus.EMPTY,
            execution_id="exec-2",
            data=[],
            coverage=SearchCoverage.TRUNCATED,
            retrieved_at=T0,
        ),
        _receipt(
            search_execution_id="search-2",
            coverage=SearchCoverage.TRUNCATED,
            match_count=0,
        ),
    )
    assert exhausted.allows_completed_query_absence_statement is True
    assert exhausted.allows_real_world_absence_statement is False
    assert truncated.allows_completed_query_absence_statement is False
    assert truncated.is_exhaustive_search is False


def test_empty_versus_error_and_unsupported():
    empty = interpret_search(
        ToolResult(
            status=ToolResultStatus.EMPTY,
            execution_id="exec-1",
            data=[],
            coverage=SearchCoverage.EXHAUSTED,
            retrieved_at=T0,
        )
    )
    error = interpret_search(
        ToolResult(
            status=ToolResultStatus.ERROR,
            execution_id="exec-2",
            error=ToolError(code=ToolErrorCode.TIMEOUT, message="timeout"),
        )
    )
    unsupported = interpret_search(
        ToolResult(
            status=ToolResultStatus.UNSUPPORTED,
            execution_id="exec-3",
            error=ToolError(
                code=ToolErrorCode.DISABLED_CAPABILITY,
                message="semantic search disabled",
            ),
        )
    )
    assert empty.allows_completed_query_absence_statement is True
    assert error.allows_completed_query_absence_statement is False
    assert unsupported.allows_completed_query_absence_statement is False
    assert rejects_real_world_absence(empty)
    assert rejects_real_world_absence(error)
    assert rejects_real_world_absence(unsupported)


def test_partial_coverage_is_not_exhaustive():
    partial = interpret_search(
        ToolResult(
            status=ToolResultStatus.PARTIAL,
            execution_id="exec-1",
            data={"items": [1]},
            coverage=SearchCoverage.PARTIAL,
            retrieved_at=T0,
        )
    )
    assert partial.is_exhaustive_search is False
    assert partial.allows_real_world_absence_statement is False
