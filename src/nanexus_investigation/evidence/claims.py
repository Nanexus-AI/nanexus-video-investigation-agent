"""Deterministic claim assessment. G3-A §5.5, G3-B §5.10 and §7.3."""

from __future__ import annotations

from dataclasses import dataclass

from nanexus_investigation.contracts.claim import Claim
from nanexus_investigation.contracts.enums import (
    ClaimType,
    ClaimValidationStatus,
    EvidenceKind,
    TemporalEvaluationStatus,
    TemporalRelationType,
)
from nanexus_investigation.contracts.evidence import EvidenceRef, Observation
from nanexus_investigation.contracts.scope import EvidenceGrant, InvestigationScope
from nanexus_investigation.contracts.temporal import TemporalRelation
from nanexus_investigation.evidence.provenance import validate_evidence_against_scope
from nanexus_investigation.evidence.relations import evaluate_predicate
from nanexus_investigation.evidence.search_coverage import interpret_search
from nanexus_investigation.evidence.time import IntervalMicros, build_interval
from nanexus_investigation.contracts.search import SearchExecutionReceipt
from nanexus_investigation.contracts.tool import ToolResult


@dataclass(frozen=True, slots=True)
class ClaimAssessment:
    """Deterministic assessment of a Claim against caller-supplied records.

    The model-written ``claim.validation_status`` is never treated as authority.
    """

    claim_id: str
    status: ClaimValidationStatus
    reasons: tuple[str, ...] = ()
    checked_relation_ids: tuple[str, ...] = ()
    checked_evidence_ids: tuple[str, ...] = ()


def assess_claim(
    claim: Claim,
    *,
    effective_scope: InvestigationScope,
    evidence_by_id: dict[str, EvidenceRef],
    observations_by_id: dict[str, Observation] | None = None,
    relations_by_id: dict[str, TemporalRelation] | None = None,
    observation_intervals: dict[str, IntervalMicros] | None = None,
    receipts_by_id: dict[str, SearchExecutionReceipt] | None = None,
    tool_results_by_receipt_id: dict[str, ToolResult] | None = None,
    grant: EvidenceGrant | None = None,
    require_grant_membership: bool = True,
) -> ClaimAssessment:
    """Assess a claim using structured evidence only.

    Natural-language entailment is not attempted. Model interpretations that
    cannot be grounded in structured records remain unverifiable or unsupported.
    A ``near`` relation alone never proves identity or causation.
    """
    observations_by_id = observations_by_id or {}
    relations_by_id = relations_by_id or {}
    observation_intervals = observation_intervals or {}
    receipts_by_id = receipts_by_id or {}
    tool_results_by_receipt_id = tool_results_by_receipt_id or {}

    reasons: list[str] = []
    checked_evidence: list[str] = []
    checked_relations: list[str] = []

    # Ignore model-supplied validation labels as authority.
    if claim.validation_status is ClaimValidationStatus.SUPPORTED:
        reasons.append(
            "model-or-producer validation_status=supported is not independent verification"
        )

    if not claim.evidence_ids and not claim.search_receipt_ids:
        return ClaimAssessment(
            claim_id=claim.claim_id,
            status=ClaimValidationStatus.UNSUPPORTED,
            reasons=tuple(reasons + ["claim has no evidence or search-receipt citations"]),
        )

    for evidence_id in claim.evidence_ids:
        evidence = evidence_by_id.get(evidence_id)
        if evidence is None:
            reasons.append(f"fabricated or missing evidence reference: {evidence_id}")
            continue
        checked_evidence.append(evidence_id)
        check = validate_evidence_against_scope(
            evidence=evidence,
            effective_scope=effective_scope,
            observations=observations_by_id or None,
            grant=grant,
            require_grant_membership=require_grant_membership and grant is not None,
        )
        if not check.accepted:
            reasons.extend(f"{evidence_id}: {item}" for item in check.reasons)

    for receipt_id in claim.search_receipt_ids:
        receipt = receipts_by_id.get(receipt_id)
        if receipt is None:
            reasons.append(f"missing search receipt: {receipt_id}")
            continue
        tool_result = tool_results_by_receipt_id.get(receipt_id)
        if tool_result is None:
            reasons.append(f"missing tool result for search receipt: {receipt_id}")
            continue
        interpretation = interpret_search(tool_result, receipt)
        if claim.claim_type is ClaimType.QUALIFIED_SEARCH_RESULT:
            if interpretation.allows_real_world_absence_statement:
                reasons.append("search interpretation incorrectly allows real-world absence")
            if (
                tool_result.status.value == "EMPTY"
                and not interpretation.allows_completed_query_absence_statement
            ):
                reasons.append(
                    "empty search is not exhausted and cannot support a completed no-match statement"
                )

    for relation_id in claim.relation_ids:
        relation = relations_by_id.get(relation_id)
        if relation is None:
            reasons.append(f"missing temporal relation: {relation_id}")
            continue
        checked_relations.append(relation_id)
        left = observation_intervals.get(relation.left_observation_id)
        right = observation_intervals.get(relation.right_observation_id)
        if left is None or right is None:
            reasons.append(
                f"{relation_id}: observation intervals unavailable for recomputation"
            )
            continue
        recomputed = evaluate_predicate(
            left,
            right,
            relation.relation_type,
            window_seconds=relation.window_seconds,
        )
        if relation.evaluation_status is TemporalEvaluationStatus.ESTABLISHED:
            # False is a proven contradiction. Unknown is uncertainty, not
            # contradiction. G3 evaluation statuses are unchanged; holds is the
            # engine discriminator.
            if recomputed.is_false:
                reasons.append(
                    f"{relation_id}: cited established relation is contradicted by recalculation"
                )
            elif recomputed.is_unknown:
                reasons.append(
                    f"{relation_id}: cited established relation cannot be "
                    "reconfirmed from available intervals"
                )
        if relation.relation_type is TemporalRelationType.NEAR:
            if claim.claim_type is ClaimType.MODEL_INTERPRETATION:
                reasons.append(
                    f"{relation_id}: near alone cannot prove identity or causation"
                )

    # Structured support decisions.
    hard_failures = [
        item
        for item in reasons
        if item.startswith("fabricated")
        or "missing evidence" in item
        or "missing temporal relation" in item
        or "missing search receipt" in item
        or "contradicted by recalculation" in item
        or "outside effective scope" in item
        or "does not match effective scope" in item
        or "absent from the supplied grant" in item
    ]
    uncertainty = [
        item
        for item in reasons
        if "cannot be reconfirmed from available intervals" in item
        or "observation intervals unavailable for recomputation" in item
    ]

    if hard_failures:
        return ClaimAssessment(
            claim_id=claim.claim_id,
            status=ClaimValidationStatus.UNSUPPORTED,
            reasons=tuple(reasons),
            checked_relation_ids=tuple(checked_relations),
            checked_evidence_ids=tuple(checked_evidence),
        )

    if uncertainty:
        return ClaimAssessment(
            claim_id=claim.claim_id,
            status=ClaimValidationStatus.UNVERIFIABLE,
            reasons=tuple(reasons),
            checked_relation_ids=tuple(checked_relations),
            checked_evidence_ids=tuple(checked_evidence),
        )

    if claim.claim_type is ClaimType.MODEL_INTERPRETATION:
        near_only = bool(claim.relation_ids) and not claim.evidence_ids
        if near_only or any("near alone cannot prove" in item for item in reasons):
            return ClaimAssessment(
                claim_id=claim.claim_id,
                status=ClaimValidationStatus.UNSUPPORTED,
                reasons=tuple(reasons + ["model interpretation is not structurally verified"]),
                checked_relation_ids=tuple(checked_relations),
                checked_evidence_ids=tuple(checked_evidence),
            )

        # T5: QUALIFIED when grounded in retained SNAPSHOT/CLIP + observe_media receipt.
        media_kinds = {EvidenceKind.SNAPSHOT, EvidenceKind.CLIP}
        grounded_media = [
            evidence_id
            for evidence_id in checked_evidence
            if (evidence_by_id.get(evidence_id) is not None)
            and evidence_by_id[evidence_id].evidence_kind in media_kinds
            and bool(evidence_by_id[evidence_id].content_sha256)
        ]
        observe_receipts: list[str] = []
        for receipt_id in claim.search_receipt_ids:
            receipt = receipts_by_id.get(receipt_id)
            tool_result = tool_results_by_receipt_id.get(receipt_id)
            if receipt is None or tool_result is None:
                continue
            if receipt.tool_name != "observe_media":
                continue
            if tool_result.status.value != "OK":
                continue
            data = tool_result.data if isinstance(tool_result.data, list) else []
            if not data or not isinstance(data[0], dict):
                continue
            record = data[0]
            if record.get("record_kind") != "visual_observation":
                continue
            if record.get("physical_world_fact") is True:
                reasons.append(
                    f"{receipt_id}: visual observation must not claim physical_world_fact"
                )
                continue
            media_id = str(record.get("media_evidence_id") or "")
            if not media_id:
                reasons.append(
                    f"{receipt_id}: visual observation missing media_evidence_id"
                )
                continue
            # Exact media binding: cited SNAPSHOT/CLIP IDs must match the observed media.
            if set(grounded_media) != {media_id}:
                reasons.append(
                    f"{receipt_id}: visual claim media citations must match "
                    "observe_media media_evidence_id exactly"
                )
                continue
            observation_text = str(record.get("observation_text") or "").strip()
            if not observation_text:
                reasons.append(
                    f"{receipt_id}: visual observation missing observation_text"
                )
                continue
            # Faithful quotation: statement must include the registered model text.
            if observation_text not in claim.statement:
                reasons.append(
                    f"{receipt_id}: visual claim statement must quote registered "
                    "observation_text"
                )
                continue
            observe_receipts.append(receipt_id)

        if grounded_media and observe_receipts and not hard_failures:
            return ClaimAssessment(
                claim_id=claim.claim_id,
                status=ClaimValidationStatus.QUALIFIED,
                reasons=tuple(
                    reasons
                    + [
                        "local VLM interpretation is grounded in retained media and "
                        "observe_media receipt; not an independently verified fact"
                    ]
                ),
                checked_relation_ids=tuple(checked_relations),
                checked_evidence_ids=tuple(checked_evidence),
            )
        if claim.search_receipt_ids and not observe_receipts:
            return ClaimAssessment(
                claim_id=claim.claim_id,
                status=ClaimValidationStatus.UNSUPPORTED,
                reasons=tuple(
                    reasons
                    + [
                        "visual-review claim lacks a valid observe_media inference record"
                    ]
                ),
                checked_relation_ids=tuple(checked_relations),
                checked_evidence_ids=tuple(checked_evidence),
            )
        return ClaimAssessment(
            claim_id=claim.claim_id,
            status=ClaimValidationStatus.UNVERIFIABLE,
            reasons=tuple(
                reasons
                + [
                    "structured evidence cannot prove the natural-language interpretation"
                ]
            ),
            checked_relation_ids=tuple(checked_relations),
            checked_evidence_ids=tuple(checked_evidence),
        )

    if claim.claim_type is ClaimType.QUALIFIED_SEARCH_RESULT:
        if any("cannot support a completed no-match" in item for item in reasons):
            return ClaimAssessment(
                claim_id=claim.claim_id,
                status=ClaimValidationStatus.UNSUPPORTED,
                reasons=tuple(reasons),
                checked_relation_ids=tuple(checked_relations),
                checked_evidence_ids=tuple(checked_evidence),
            )
        return ClaimAssessment(
            claim_id=claim.claim_id,
            status=ClaimValidationStatus.QUALIFIED,
            reasons=tuple(
                reasons
                + [
                    "qualified search claims remain limited to the completed query scope"
                ]
            ),
            checked_relation_ids=tuple(checked_relations),
            checked_evidence_ids=tuple(checked_evidence),
        )

    if claim.claim_type is ClaimType.COMPUTED_RELATION:
        if claim.relation_ids:
            return ClaimAssessment(
                claim_id=claim.claim_id,
                status=ClaimValidationStatus.SUPPORTED,
                reasons=tuple(
                    [
                        item
                        for item in reasons
                        if "validation_status=supported" not in item
                    ]
                    or ["cited temporal relations recomputed successfully"]
                ),
                checked_relation_ids=tuple(checked_relations),
                checked_evidence_ids=tuple(checked_evidence),
            )
        return ClaimAssessment(
            claim_id=claim.claim_id,
            status=ClaimValidationStatus.UNVERIFIABLE,
            reasons=tuple(reasons + ["computed relation could not be fully verified"]),
            checked_relation_ids=tuple(checked_relations),
            checked_evidence_ids=tuple(checked_evidence),
        )

    # SOURCE_OBSERVATION
    if checked_evidence and not hard_failures:
        unresolved = [item for item in reasons if "validation_status=supported" not in item]
        if unresolved:
            return ClaimAssessment(
                claim_id=claim.claim_id,
                status=ClaimValidationStatus.QUALIFIED,
                reasons=tuple(reasons),
                checked_relation_ids=tuple(checked_relations),
                checked_evidence_ids=tuple(checked_evidence),
            )
        return ClaimAssessment(
            claim_id=claim.claim_id,
            status=ClaimValidationStatus.SUPPORTED,
            reasons=("cited evidence passed provenance and scope checks",),
            checked_relation_ids=tuple(checked_relations),
            checked_evidence_ids=tuple(checked_evidence),
        )

    return ClaimAssessment(
        claim_id=claim.claim_id,
        status=ClaimValidationStatus.UNVERIFIABLE,
        reasons=tuple(reasons or ["insufficient structured evidence"]),
        checked_relation_ids=tuple(checked_relations),
        checked_evidence_ids=tuple(checked_evidence),
    )


def observation_interval(observation: Observation) -> IntervalMicros:
    """Build an IntervalMicros from an Observation's timestamps."""
    return build_interval(observation.start_utc, observation.end_utc)
