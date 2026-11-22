"""Bounded in-memory Case artifact store for E0.

Development substitute for Investigation Persistence. Not crash-safe.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from nanexus_investigation.contracts.case import InvestigationCase
from nanexus_investigation.contracts.claim import Claim, InvestigationReport
from nanexus_investigation.contracts.evidence import EvidenceRef, Observation
from nanexus_investigation.contracts.search import SearchExecutionReceipt
from nanexus_investigation.contracts.temporal import TemporalRelation
from nanexus_investigation.contracts.tool import ToolResult
from nanexus_investigation.evidence.claims import ClaimAssessment
from nanexus_investigation.evidence.time import IntervalMicros


@dataclass
class InMemoryCaseRepository:
    """Case-scoped artifact registry. IDs only enter graph state."""

    case: InvestigationCase
    observations: dict[str, Observation] = field(default_factory=dict)
    evidence: dict[str, EvidenceRef] = field(default_factory=dict)
    receipts: dict[str, SearchExecutionReceipt] = field(default_factory=dict)
    relations: dict[str, TemporalRelation] = field(default_factory=dict)
    claims: dict[str, Claim] = field(default_factory=dict)
    assessments: dict[str, ClaimAssessment] = field(default_factory=dict)
    tool_results: dict[str, ToolResult] = field(default_factory=dict)
    tool_results_by_receipt: dict[str, ToolResult] = field(default_factory=dict)
    observation_intervals: dict[str, IntervalMicros] = field(default_factory=dict)
    report: InvestigationReport | None = None

    def register_observation(self, observation: Observation) -> None:
        self.observations[observation.observation_id] = observation

    def register_evidence(self, evidence: EvidenceRef) -> None:
        self.evidence[evidence.evidence_id] = evidence

    def register_receipt(self, receipt: SearchExecutionReceipt) -> None:
        self.receipts[receipt.search_execution_id] = receipt

    def register_relation(self, relation: TemporalRelation) -> None:
        self.relations[relation.relation_id] = relation

    def register_claim(self, claim: Claim) -> None:
        self.claims[claim.claim_id] = claim

    def set_assessment(self, assessment: ClaimAssessment) -> None:
        self.assessments[assessment.claim_id] = assessment

    def register_tool_result(
        self,
        result: ToolResult,
        *,
        receipt_id: str | None = None,
    ) -> None:
        self.tool_results[result.execution_id] = result
        if receipt_id is not None:
            self.tool_results_by_receipt[receipt_id] = result
