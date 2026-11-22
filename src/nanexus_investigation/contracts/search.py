"""SearchExecutionReceipt. G3-A §6.1."""

from datetime import datetime

from pydantic import Field, field_validator, model_validator

from nanexus_investigation.contracts.enums import SearchCoverage
from nanexus_investigation.contracts.identity import ContractModel, Interval, _as_utc
from nanexus_investigation.contracts.tool import ToolError


class SearchExecutionReceipt(ContractModel):
    """Provenance for a bounded search execution. G3-A §6.1.

    An EMPTY result with exhausted coverage supports only a statement about
    that completed query. It is not proof that nothing happened in the world.
    """

    search_execution_id: str = Field(min_length=1)
    tool_name: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    camera_ids: tuple[str, ...] = Field(min_length=1)
    time_bounds: Interval
    filters: dict[str, str | int | float | bool | None] = Field(default_factory=dict)
    retrieved_at: datetime
    pages_fetched: int = Field(ge=0)
    page_limit: int | None = Field(default=None, ge=1)
    match_count: int = Field(ge=0)
    coverage: SearchCoverage
    source_capability: str | None = None
    consistency_limitations: str | None = None
    error: ToolError | None = None

    @field_validator(
        "search_execution_id",
        "tool_name",
        "source_id",
    )
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if value.strip() != value or not value.strip():
            raise ValueError("receipt identity fields must be non-empty and trimmed")
        return value

    @field_validator("camera_ids")
    @classmethod
    def _validate_cameras(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("camera_ids must contain at least one camera")
        for camera_id in value:
            if camera_id.strip() != camera_id or not camera_id.strip():
                raise ValueError("camera identifiers must be non-empty and trimmed")
        return value

    @field_validator("retrieved_at")
    @classmethod
    def _require_utc(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @model_validator(mode="after")
    def _coverage_consistency(self) -> "SearchExecutionReceipt":
        if self.error is not None and self.match_count > 0:
            raise ValueError("error receipts must not report matching records")
        if (
            self.coverage == SearchCoverage.EXHAUSTED
            and self.match_count == 0
            and self.error is not None
        ):
            raise ValueError(
                "an exhausted empty search cannot also record an execution error"
            )
        return self
