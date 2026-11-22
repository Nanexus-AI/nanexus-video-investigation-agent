"""Canonical domain contracts for the Nanexus investigation agent.

These models encode the approved G3 conceptual contracts. They do not
authorize access, validate evidence against a live grant store, or execute
a graph.
"""

from nanexus_investigation.contracts.capability import (
    CapabilitySnapshot,
    OperationCapability,
)
from nanexus_investigation.contracts.case import (
    CASE_TRANSITIONS,
    InvestigationCase,
    validate_case_transition,
)
from nanexus_investigation.contracts.claim import Claim, InvestigationReport
from nanexus_investigation.contracts.enums import (
    CapabilityAvailability,
    CaseStatus,
    ClaimType,
    ClaimValidationStatus,
    EvidenceAvailability,
    EvidenceKind,
    ObservationType,
    RecordType,
    SearchCoverage,
    TemporalEvaluationStatus,
    TemporalRelationType,
    ToolErrorCode,
    ToolResultStatus,
)
from nanexus_investigation.contracts.evidence import EvidenceRef, Observation
from nanexus_investigation.contracts.identity import (
    Interval,
    SourceLocator,
    TimeContext,
)
from nanexus_investigation.contracts.scope import (
    EvidenceGrant,
    InvestigationScope,
    apply_requested_scope,
)
from nanexus_investigation.contracts.search import SearchExecutionReceipt
from nanexus_investigation.contracts.temporal import TemporalRelation
from nanexus_investigation.contracts.tool import ToolError, ToolRequest, ToolResult

__all__ = [
    "CASE_TRANSITIONS",
    "CapabilityAvailability",
    "CapabilitySnapshot",
    "CaseStatus",
    "Claim",
    "ClaimType",
    "ClaimValidationStatus",
    "EvidenceAvailability",
    "EvidenceGrant",
    "EvidenceKind",
    "EvidenceRef",
    "Interval",
    "InvestigationCase",
    "InvestigationReport",
    "InvestigationScope",
    "Observation",
    "ObservationType",
    "OperationCapability",
    "RecordType",
    "SearchCoverage",
    "SearchExecutionReceipt",
    "SourceLocator",
    "TemporalEvaluationStatus",
    "TemporalRelation",
    "TemporalRelationType",
    "TimeContext",
    "ToolError",
    "ToolErrorCode",
    "ToolRequest",
    "ToolResult",
    "ToolResultStatus",
    "apply_requested_scope",
    "validate_case_transition",
]
