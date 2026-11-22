"""Claim and InvestigationReport. G3-A §5.5 and §5.6."""

from datetime import datetime

from pydantic import Field, field_validator, model_validator

from nanexus_investigation.contracts.enums import (
    CaseStatus,
    ClaimType,
    ClaimValidationStatus,
)
from nanexus_investigation.contracts.identity import ContractModel, _as_utc


class Claim(ContractModel):
    """Assertion prepared for investigation output. G3-A §5.5.

    ``validation_status`` is a recorded vocabulary value. Setting it to
    ``supported`` here is schema-valid only when citation fields are present.
    It is not a completed Evidence Engine assessment, and it does not satisfy
    the Output Gate.
    """

    claim_id: str = Field(min_length=1)
    statement: str = Field(min_length=1)
    claim_type: ClaimType
    evidence_ids: tuple[str, ...] = ()
    relation_ids: tuple[str, ...] = ()
    search_receipt_ids: tuple[str, ...] = ()
    producer: str = Field(min_length=1)
    validation_status: ClaimValidationStatus
    limitations: tuple[str, ...] = ()
    validated_at: datetime | None = None

    @field_validator("claim_id", "statement", "producer")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if value.strip() != value or not value.strip():
            raise ValueError("claim fields must be non-empty and trimmed")
        return value

    @field_validator("evidence_ids", "relation_ids", "search_receipt_ids")
    @classmethod
    def _reject_blank_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            if item.strip() != item or not item.strip():
                raise ValueError("claim references must be non-empty and trimmed")
        return value

    @field_validator("validated_at")
    @classmethod
    def _require_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return value
        return _as_utc(value)

    @model_validator(mode="after")
    def _citation_rules(self) -> "Claim":
        # Structural citation presence only. Citation IDs are not verified
        # against a grant or durable evidence store in this contract layer.
        if self.validation_status == ClaimValidationStatus.SUPPORTED:
            if not self.evidence_ids and not self.search_receipt_ids:
                raise ValueError(
                    "supported claims require evidence or search-receipt citations"
                )
        if (
            self.claim_type == ClaimType.QUALIFIED_SEARCH_RESULT
            and not self.search_receipt_ids
        ):
            raise ValueError("qualified search claims require search_receipt_ids")
        return self


class InvestigationReport(ContractModel):
    """Case-level investigation output. G3-A §5.6.

    Stores claim and evidence identifiers, not media bytes.
    """

    report_id: str = Field(min_length=1)
    case_id: str = Field(min_length=1)
    generated_at: datetime
    scope_summary: str = Field(min_length=1)
    execution_status: CaseStatus
    timeline: tuple[str, ...] = ()
    claims: tuple[str, ...] = ()
    evidence_index: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    unresolved_questions: tuple[str, ...] = ()
    validation_summary: str = Field(min_length=1)
    report_version: str = Field(min_length=1)

    @field_validator(
        "report_id",
        "case_id",
        "scope_summary",
        "validation_summary",
        "report_version",
    )
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if value.strip() != value or not value.strip():
            raise ValueError("report fields must be non-empty and trimmed")
        return value

    @field_validator("generated_at")
    @classmethod
    def _require_utc(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @field_validator("claims", "evidence_index")
    @classmethod
    def _reject_blank_refs(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for item in value:
            if item.strip() != item or not item.strip():
                raise ValueError("report references must be non-empty and trimmed")
        return value

    @model_validator(mode="after")
    def _terminal_status_only(self) -> "InvestigationReport":
        terminal = {
            CaseStatus.COMPLETED,
            CaseStatus.PARTIAL,
            CaseStatus.FAILED,
            CaseStatus.CANCELLED,
        }
        if self.execution_status not in terminal:
            raise ValueError("report execution_status must be a terminal Case status")
        return self
