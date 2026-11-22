"""TemporalRelation contract. G3-A §5.4.

Arithmetic and predicate evaluation belong to V1-C.
"""

from pydantic import Field, field_validator, model_validator

from nanexus_investigation.contracts.enums import (
    TemporalEvaluationStatus,
    TemporalRelationType,
)
from nanexus_investigation.contracts.identity import ContractModel


class TemporalRelation(ContractModel):
    """Typed temporal relationship between two observations."""

    relation_id: str = Field(min_length=1)
    left_observation_id: str = Field(min_length=1)
    right_observation_id: str = Field(min_length=1)
    relation_type: TemporalRelationType
    evaluation_status: TemporalEvaluationStatus
    window_seconds: float | None = Field(default=None, ge=0)
    gap_seconds: float | None = None
    camera_relation: str = Field(min_length=1)
    time_basis: str = Field(min_length=1)
    calculation_provenance: str = Field(min_length=1)

    @field_validator(
        "relation_id",
        "left_observation_id",
        "right_observation_id",
        "camera_relation",
        "time_basis",
        "calculation_provenance",
    )
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if value.strip() != value or not value.strip():
            raise ValueError("temporal relation fields must be non-empty and trimmed")
        return value

    @model_validator(mode="after")
    def _distinct_observations(self) -> "TemporalRelation":
        if self.left_observation_id == self.right_observation_id:
            raise ValueError("temporal relations require two distinct observations")
        if self.relation_type == TemporalRelationType.NEAR and self.window_seconds is None:
            raise ValueError("near relations require an explicit window_seconds")
        return self
