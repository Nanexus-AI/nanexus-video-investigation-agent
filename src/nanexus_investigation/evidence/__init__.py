"""Deterministic Evidence and Temporal Engine helpers.

These pure functions implement G3 time, relation, provenance, search-coverage,
and claim-assessment rules. They do not call a VMS, run a graph, or authorize
access.
"""

from nanexus_investigation.evidence.claims import (
    ClaimAssessment,
    assess_claim,
    observation_interval,
)
from nanexus_investigation.evidence.provenance import (
    EvidenceCheckResult,
    validate_evidence_against_scope,
)
from nanexus_investigation.evidence.relations import (
    TemporalPredicateResult,
    evaluate_predicate,
    evaluate_relation,
)
from nanexus_investigation.evidence.search_coverage import (
    SearchInterpretation,
    interpret_search,
    rejects_real_world_absence,
)
from nanexus_investigation.evidence.time import (
    CompletedIntervalMicros,
    InstantMicros,
    IntervalMicros,
    build_completed_interval,
    build_interval,
    datetime_to_utc_micros,
    micros_to_utc_datetime,
    unix_seconds_to_utc_micros,
)

__all__ = [
    "ClaimAssessment",
    "CompletedIntervalMicros",
    "EvidenceCheckResult",
    "InstantMicros",
    "IntervalMicros",
    "SearchInterpretation",
    "TemporalPredicateResult",
    "assess_claim",
    "build_completed_interval",
    "build_interval",
    "datetime_to_utc_micros",
    "evaluate_predicate",
    "evaluate_relation",
    "interpret_search",
    "micros_to_utc_datetime",
    "observation_interval",
    "rejects_real_world_absence",
    "unix_seconds_to_utc_micros",
    "validate_evidence_against_scope",
]
