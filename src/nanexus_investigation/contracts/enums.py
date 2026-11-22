"""Closed vocabularies from the approved G3 contracts."""

from enum import StrEnum


class CaseStatus(StrEnum):
    """External InvestigationCase lifecycle. G3-A §5.1, G3-B §13."""

    CREATED = "created"
    RUNNING = "running"
    AWAITING_INPUT = "awaiting_input"
    COMPLETED = "completed"
    PARTIAL = "partial"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ToolResultStatus(StrEnum):
    """Closed ToolResult status vocabulary. G3-A §6.3."""

    OK = "OK"
    EMPTY = "EMPTY"
    PARTIAL = "PARTIAL"
    ERROR = "ERROR"
    UNSUPPORTED = "UNSUPPORTED"


class ToolErrorCode(StrEnum):
    """Machine-readable error categories from G3-A §6.3.

    The spellings are an implementation choice. The categories match the
    architecture text.
    """

    AUTHORIZATION_DENIED = "authorization_denied"
    INVALID_SCOPE = "invalid_scope"
    SOURCE_UNAVAILABLE = "source_unavailable"
    TIMEOUT = "timeout"
    INVALID_RESPONSE = "invalid_response"
    RATE_LIMITED = "rate_limited"
    MISSING_MEDIA = "missing_media"
    DISABLED_CAPABILITY = "disabled_capability"


class SearchCoverage(StrEnum):
    """SearchExecutionReceipt coverage. G3-A §6.1."""

    EXHAUSTED = "exhausted"
    TRUNCATED = "truncated"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class RecordType(StrEnum):
    """Native source record kinds. G3-A §4.1."""

    EVENT = "event"
    REVIEW = "review"
    RECORDING_SEGMENT = "recording_segment"


class ObservationType(StrEnum):
    """Normalized observation kinds. G3-A §5.2."""

    DETECTION_EVENT = "detection_event"
    REVIEW_SEGMENT = "review_segment"


class EvidenceKind(StrEnum):
    """EvidenceRef kinds. G3-A §5.3."""

    EVENT_METADATA = "event_metadata"
    REVIEW_METADATA = "review_metadata"
    SNAPSHOT = "snapshot"
    CLIP = "clip"
    RECORDING_INTERVAL = "recording_interval"


class EvidenceAvailability(StrEnum):
    """Evidence availability assessment. G3-A §5.3."""

    NOT_CHECKED = "not_checked"
    AVAILABLE = "available"
    MISSING = "missing"
    INACCESSIBLE = "inaccessible"
    ERROR = "error"


class TemporalRelationType(StrEnum):
    """Approved temporal predicates. G3-A §5.4, G3-B §7.2."""

    BEFORE = "before"
    AFTER = "after"
    OVERLAPS = "overlaps"
    CONTAINS = "contains"
    NEAR = "near"


class TemporalEvaluationStatus(StrEnum):
    """Whether a temporal predicate can be established. G3-A §5.4."""

    ESTABLISHED = "established"
    INDETERMINATE = "indeterminate"
    UNSUPPORTED = "unsupported"


class ClaimType(StrEnum):
    """Claim kinds. G3-A §5.5."""

    SOURCE_OBSERVATION = "source_observation"
    COMPUTED_RELATION = "computed_relation"
    MODEL_INTERPRETATION = "model_interpretation"
    QUALIFIED_SEARCH_RESULT = "qualified_search_result"


class ClaimValidationStatus(StrEnum):
    """Claim validation outcomes. G3-A §5.5."""

    SUPPORTED = "supported"
    QUALIFIED = "qualified"
    UNSUPPORTED = "unsupported"
    UNVERIFIABLE = "unverifiable"


class CapabilityAvailability(StrEnum):
    """Per-operation capability states. G3-C2 §7.2."""

    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    TEMPORARILY_UNAVAILABLE = "temporarily_unavailable"
    NOT_AUTHORIZED = "not_authorized"
