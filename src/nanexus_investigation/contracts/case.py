"""InvestigationCase and lifecycle transitions. G3-A §5.1, G3-B §13."""

from datetime import datetime

from pydantic import Field, field_validator, model_validator

from nanexus_investigation.contracts.capability import CapabilitySnapshot
from nanexus_investigation.contracts.enums import CaseStatus
from nanexus_investigation.contracts.identity import ContractModel, TimeContext, _as_utc
from nanexus_investigation.contracts.scope import InvestigationScope

CASE_TRANSITIONS: frozenset[tuple[CaseStatus, CaseStatus]] = frozenset(
    {
        (CaseStatus.CREATED, CaseStatus.RUNNING),
        (CaseStatus.RUNNING, CaseStatus.AWAITING_INPUT),
        (CaseStatus.AWAITING_INPUT, CaseStatus.RUNNING),
        (CaseStatus.RUNNING, CaseStatus.COMPLETED),
        (CaseStatus.RUNNING, CaseStatus.PARTIAL),
        (CaseStatus.RUNNING, CaseStatus.FAILED),
        (CaseStatus.RUNNING, CaseStatus.CANCELLED),
        (CaseStatus.AWAITING_INPUT, CaseStatus.CANCELLED),
    }
)

TERMINAL_CASE_STATUSES: frozenset[CaseStatus] = frozenset(
    {
        CaseStatus.COMPLETED,
        CaseStatus.PARTIAL,
        CaseStatus.FAILED,
        CaseStatus.CANCELLED,
    }
)


def validate_case_transition(current: CaseStatus, new: CaseStatus) -> None:
    """Reject illegal Case lifecycle transitions. G3-B §13."""
    if current == new:
        raise ValueError(f"case status is already {current}")
    if current in TERMINAL_CASE_STATUSES:
        raise ValueError(f"terminal case status {current} cannot transition")
    if (current, new) not in CASE_TRANSITIONS:
        raise ValueError(f"illegal case transition: {current} -> {new}")


class InvestigationCase(ContractModel):
    """Authorized investigation request. G3-A §5.1."""

    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    requested_scope: InvestigationScope
    effective_scope: InvestigationScope
    time_context: TimeContext
    status: CaseStatus
    policy_ref: str = Field(min_length=1)
    created_at: datetime
    capability_snapshot: CapabilitySnapshot | None = None
    budget_profile_ref: str | None = None

    @field_validator("case_id", "question", "policy_ref")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if value.strip() != value or not value.strip():
            raise ValueError("case fields must be non-empty and trimmed")
        return value

    @field_validator("created_at")
    @classmethod
    def _require_utc(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @model_validator(mode="after")
    def _scope_consistency(self) -> "InvestigationCase":
        if (
            self.effective_scope.vms_instance_id
            != self.requested_scope.vms_instance_id
        ):
            raise ValueError("effective scope must keep the requested VMS instance")
        # Pure Case construction keeps effective cameras inside the request.
        # Harness authorization remains a separate responsibility.
        unauthorized = [
            camera_id
            for camera_id in self.effective_scope.camera_ids
            if camera_id not in self.requested_scope.camera_ids
        ]
        if unauthorized:
            raise ValueError(
                "effective scope cameras must be contained in requested scope"
            )
        return self

    def transition_to(self, new_status: CaseStatus) -> "InvestigationCase":
        validate_case_transition(self.status, new_status)
        return self.model_copy(update={"status": new_status})
