"""Tool request and ToolResult envelope. G3-A §6.3."""

from datetime import datetime
from typing import Any

from pydantic import Field, field_validator, model_validator

from nanexus_investigation.contracts.enums import (
    SearchCoverage,
    ToolErrorCode,
    ToolResultStatus,
)
from nanexus_investigation.contracts.identity import ContractModel, _as_utc


class ToolRequest(ContractModel):
    """Validated structured tool request. G3-B §5.4."""

    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    source_id: str | None = None
    camera_ids: tuple[str, ...] = ()

    @field_validator("tool_name")
    @classmethod
    def _reject_blank_tool(cls, value: str) -> str:
        if value.strip() != value or not value.strip():
            raise ValueError("tool_name must be non-empty and trimmed")
        return value

    @field_validator("camera_ids")
    @classmethod
    def _validate_cameras(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        for camera_id in value:
            if camera_id.strip() != camera_id or not camera_id.strip():
                raise ValueError("camera identifiers must be non-empty and trimmed")
        return value

    @field_validator("arguments")
    @classmethod
    def _reject_credential_keys(cls, value: dict[str, Any]) -> dict[str, Any]:
        forbidden = {
            "password",
            "token",
            "api_key",
            "secret",
            "authorization",
            "credential",
            "credentials",
        }
        for key in value:
            if key.lower() in forbidden or key.lower().endswith("_token"):
                raise ValueError("tool arguments must not contain credentials")
        return value


class ToolError(ContractModel):
    """Stable error detail for a ToolResult. G3-A §6.3."""

    code: ToolErrorCode
    message: str = Field(min_length=1)

    @field_validator("message")
    @classmethod
    def _reject_blank_message(cls, value: str) -> str:
        if value.strip() != value or not value.strip():
            raise ValueError("error message must be non-empty and trimmed")
        return value


class ToolResult(ContractModel):
    """Typed tool outcome envelope. G3-A §6.3."""

    status: ToolResultStatus
    execution_id: str = Field(min_length=1)
    data: Any = None
    error: ToolError | None = None
    source_id: str | None = None
    retrieved_at: datetime | None = None
    coverage: SearchCoverage | None = None
    provenance: str | None = None
    usage: dict[str, int | float] = Field(default_factory=dict)

    @field_validator("execution_id")
    @classmethod
    def _reject_blank_execution(cls, value: str) -> str:
        if value.strip() != value or not value.strip():
            raise ValueError("execution_id must be non-empty and trimmed")
        return value

    @field_validator("retrieved_at")
    @classmethod
    def _require_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return value
        return _as_utc(value)

    @model_validator(mode="after")
    def _status_consistency(self) -> "ToolResult":
        if self.status in {ToolResultStatus.ERROR, ToolResultStatus.UNSUPPORTED}:
            if self.error is None:
                raise ValueError(f"{self.status} results require an error detail")
        if self.status == ToolResultStatus.OK and self.error is not None:
            raise ValueError("OK results must not carry an error detail")
        if self.status == ToolResultStatus.EMPTY and self.data not in (None, [], {}, ()):
            raise ValueError("EMPTY results must not carry matching records")
        return self
