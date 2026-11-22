"""Evidence provenance and effective-scope checks. G3-A §4–§6, G3-C2 §7."""

from __future__ import annotations

from dataclasses import dataclass

from nanexus_investigation.contracts.evidence import EvidenceRef, Observation
from nanexus_investigation.contracts.identity import Interval, SourceLocator
from nanexus_investigation.contracts.scope import EvidenceGrant, InvestigationScope
from nanexus_investigation.evidence.time import IntervalMicros, build_interval


@dataclass(frozen=True, slots=True)
class EvidenceCheckResult:
    """Outcome of a pure evidence/scope check against caller-supplied records."""

    accepted: bool
    reasons: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return self.accepted


def _fail(*reasons: str) -> EvidenceCheckResult:
    return EvidenceCheckResult(accepted=False, reasons=reasons)


def _ok() -> EvidenceCheckResult:
    return EvidenceCheckResult(accepted=True)


def observation_matches_locator(
    observation: Observation,
    locator: SourceLocator,
) -> bool:
    return observation.source_locator == locator


def validate_evidence_against_scope(
    *,
    evidence: EvidenceRef,
    effective_scope: InvestigationScope,
    observations: dict[str, Observation] | None = None,
    grant: EvidenceGrant | None = None,
    require_grant_membership: bool = False,
    expected_revision: str | None = None,
) -> EvidenceCheckResult:
    """Validate evidence provenance against an explicit effective scope.

    A syntactically valid ``EvidenceGrant`` is not treated as live authorization.
    When ``require_grant_membership`` is true, the caller-supplied grant is used
    only as an explicit membership set for this check.
    """
    reasons: list[str] = []

    if evidence.source_locator.vms_instance_id != effective_scope.vms_instance_id:
        reasons.append("vms_instance_id does not match effective scope")

    if evidence.camera_id not in effective_scope.camera_ids:
        reasons.append("camera_id is outside effective scope")

    if evidence.interval is not None:
        try:
            evidence_interval = _interval_micros(evidence.interval)
            scope_interval = _interval_micros(effective_scope.time_bounds)
        except ValueError as exc:
            return _fail(f"invalid interval: {exc}")
        if not _interval_within_scope(evidence_interval, scope_interval):
            reasons.append("evidence time is outside effective scope window")
    else:
        reasons.append("evidence has no interval for scope time validation")

    if require_grant_membership:
        if grant is None:
            return _fail("grant membership was required but no grant was supplied")
        if grant.effective_scope != effective_scope:
            reasons.append(
                "supplied grant effective_scope does not match the check scope"
            )
        if not grant.admits(evidence.evidence_id):
            reasons.append(
                "evidence_id is absent from the supplied grant membership set"
            )

    if observations is not None:
        matched = [
            obs
            for obs in observations.values()
            if observation_matches_locator(obs, evidence.source_locator)
        ]
        if not matched:
            reasons.append("no observation resolves the evidence source locator")
        elif expected_revision is not None and all(
            obs.source_revision != expected_revision for obs in matched
        ):
            reasons.append("observation revision does not match expected revision")

    if reasons:
        return _fail(*reasons)
    return _ok()


def _interval_micros(interval: Interval) -> IntervalMicros:
    return build_interval(interval.start_utc, interval.end_utc)


def _interval_within_scope(
    evidence: IntervalMicros,
    scope: IntervalMicros,
) -> bool:
    """True when a completed evidence interval lies inside a completed scope window."""
    if scope.end is None or evidence.end is None:
        return False
    return (
        evidence.start.utc_micros >= scope.start.utc_micros
        and evidence.end.utc_micros <= scope.end.utc_micros
    )
