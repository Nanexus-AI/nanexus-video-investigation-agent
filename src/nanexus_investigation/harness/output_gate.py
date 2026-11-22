"""Output Gate: independent structural and evidence checks before report return."""

from __future__ import annotations

from dataclasses import dataclass

from nanexus_investigation.contracts.case import InvestigationCase
from nanexus_investigation.contracts.claim import Claim, InvestigationReport
from nanexus_investigation.contracts.enums import (
    CaseStatus,
    ClaimType,
    ClaimValidationStatus,
)
from nanexus_investigation.evidence.claims import ClaimAssessment, assess_claim
from nanexus_investigation.evidence.search_coverage import interpret_search


@dataclass(frozen=True, slots=True)
class OutputGateDecision:
    """Harness Output Gate result. Does not perform general NL fact checking."""

    accepted: bool
    case_status: CaseStatus
    reasons: tuple[str, ...] = ()
    claim_assessments: tuple[ClaimAssessment, ...] = ()
    report: InvestigationReport | None = None


def run_output_gate(
    *,
    case: InvestigationCase,
    report: InvestigationReport,
    claims_by_id: dict[str, Claim],
    assessments: dict[str, ClaimAssessment],
    evidence_ids: frozenset[str],
    observation_ids: frozenset[str],
    relation_ids: frozenset[str],
    search_receipt_ids: frozenset[str],
    evidence_by_id: dict,
    observations_by_id: dict,
    relations_by_id: dict,
    receipts_by_id: dict,
    tool_results_by_receipt_id: dict,
    observation_intervals: dict,
) -> OutputGateDecision:
    """Validate report structure, citations, and claim assessment outcomes.

    Producer ``validation_status`` labels are ignored as authority. This gate
    does not perform general natural-language factual verification.
    """
    reasons: list[str] = []
    assessed: list[ClaimAssessment] = []

    if report.case_id != case.case_id:
        reasons.append("report case_id does not match InvestigationCase")

    for evidence_id in report.evidence_index:
        if evidence_id not in evidence_ids:
            reasons.append(f"report cites unregistered evidence: {evidence_id}")

    for claim_id in report.claims:
        claim = claims_by_id.get(claim_id)
        if claim is None:
            reasons.append(f"report cites missing claim: {claim_id}")
            continue
        for evidence_id in claim.evidence_ids:
            if evidence_id not in evidence_ids:
                reasons.append(
                    f"claim {claim_id} cites unregistered evidence: {evidence_id}"
                )
        for relation_id in claim.relation_ids:
            if relation_id not in relation_ids:
                reasons.append(
                    f"claim {claim_id} cites unregistered relation: {relation_id}"
                )
        for receipt_id in claim.search_receipt_ids:
            if receipt_id not in search_receipt_ids:
                reasons.append(
                    f"claim {claim_id} cites unregistered search receipt: {receipt_id}"
                )

        assessment = assessments.get(claim_id) or assess_claim(
            claim,
            effective_scope=case.effective_scope,
            evidence_by_id=evidence_by_id,
            observations_by_id=observations_by_id,
            relations_by_id=relations_by_id,
            observation_intervals=observation_intervals,
            receipts_by_id=receipts_by_id,
            tool_results_by_receipt_id=tool_results_by_receipt_id,
            require_grant_membership=False,
        )
        assessed.append(assessment)

        if claim.validation_status is ClaimValidationStatus.SUPPORTED:
            reasons.append(
                f"claim {claim_id}: producer validation_status=supported is not trusted"
            )
        if (
            assessment.status is ClaimValidationStatus.UNSUPPORTED
            and claim.claim_type is not ClaimType.QUALIFIED_SEARCH_RESULT
        ):
            # Unsupported factual promotions must not pass the gate as completed facts.
            if "supported conclusion" in claim.statement.lower() or (
                claim.claim_type is ClaimType.COMPUTED_RELATION
                and "contradiction" in " ".join(assessment.reasons).lower()
            ):
                reasons.append(
                    f"claim {claim_id}: unsupported or contradicted claim blocked"
                )
            elif any(
                "contradicted by recalculation" in item for item in assessment.reasons
            ):
                reasons.append(
                    f"claim {claim_id}: temporal contradiction cannot be a supported conclusion"
                )
            elif any(
                "fabricated or missing evidence" in item for item in assessment.reasons
            ):
                reasons.append(
                    f"claim {claim_id}: missing evidence prevents supported report claim"
                )

        if claim.claim_type is ClaimType.QUALIFIED_SEARCH_RESULT:
            for receipt_id in claim.search_receipt_ids:
                receipt = receipts_by_id.get(receipt_id)
                tool_result = tool_results_by_receipt_id.get(receipt_id)
                if receipt is None or tool_result is None:
                    continue
                interpretation = interpret_search(tool_result, receipt)
                if interpretation.allows_real_world_absence_statement:
                    reasons.append(
                        f"claim {claim_id}: real-world absence statement is forbidden"
                    )
                if "nothing happened" in claim.statement.lower():
                    reasons.append(
                        f"claim {claim_id}: EMPTY search must not become real-world absence"
                    )

    # Limitations from coverage must remain visible on non-empty reports.
    if search_receipt_ids and not report.limitations:
        reasons.append("search limitations must remain visible on the report")

    hard = [
        item
        for item in reasons
        if "does not match" in item
        or "unregistered" in item
        or "missing claim" in item
        or "blocked" in item
        or "contradiction" in item
        or "missing evidence prevents" in item
        or "real-world absence" in item
        or "producer validation_status" in item
    ]

    if hard:
        return OutputGateDecision(
            accepted=False,
            case_status=CaseStatus.FAILED,
            reasons=tuple(reasons),
            claim_assessments=tuple(assessed),
            report=None,
        )

    supported_or_qualified = [
        item
        for item in assessed
        if item.status
        in {ClaimValidationStatus.SUPPORTED, ClaimValidationStatus.QUALIFIED}
    ]
    blocking_assessment = any(
        item.status
        in {ClaimValidationStatus.UNVERIFIABLE, ClaimValidationStatus.UNSUPPORTED}
        for item in assessed
        if item.claim_id in report.claims
    )
    material_search_limits = any(
        "EMPTY" in item
        or "ERROR" in item
        or "UNSUPPORTED" in item
        or "partial" in item.lower()
        or "real-world absence" in item.lower()
        for item in report.limitations
    )

    if supported_or_qualified and not blocking_assessment and not material_search_limits:
        status = CaseStatus.COMPLETED
    elif supported_or_qualified or report.claims:
        status = CaseStatus.PARTIAL
    else:
        status = CaseStatus.FAILED
        reasons.append("no evidence-grounded claims available for report")
        return OutputGateDecision(
            accepted=False,
            case_status=status,
            reasons=tuple(reasons),
            claim_assessments=tuple(assessed),
            report=None,
        )

    gated = report.model_copy(
        update={
            "execution_status": status,
            "validation_summary": (
                f"output_gate:{status.value}; assessments="
                + ",".join(f"{a.claim_id}:{a.status.value}" for a in assessed)
            ),
        }
    )
    return OutputGateDecision(
        accepted=True,
        case_status=status,
        reasons=tuple(reasons),
        claim_assessments=tuple(assessed),
        report=gated,
    )
