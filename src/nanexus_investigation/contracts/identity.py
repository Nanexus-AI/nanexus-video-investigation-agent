"""Shared identity and time representations. G3-A §4."""

from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ContractModel(BaseModel):
    """JSON-friendly base for canonical contracts.

    ``extra="forbid"`` rejects unknown top-level fields. ``frozen=True``
    blocks attribute reassignment on the model instance. Neither setting is
    authorization, evidence verification, or deep immutability of values
    supplied inside open containers such as ``dict`` payloads.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware UTC instants")
    return value.astimezone(timezone.utc)


class SourceLocator(ContractModel):
    """Qualified native source identity. G3-A §4.1."""

    vms_instance_id: str = Field(min_length=1)
    record_type: str = Field(min_length=1)
    native_record_id: str = Field(min_length=1)

    @field_validator("vms_instance_id", "record_type", "native_record_id")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if value.strip() != value or not value.strip():
            raise ValueError("source identity fields must be non-empty and trimmed")
        if "://" in value or value.startswith("file:"):
            raise ValueError("source identity must not be a URL or file path")
        return value


class Interval(ContractModel):
    """Half-open UTC interval [start_utc, end_utc). G3-A §4.3."""

    start_utc: datetime
    end_utc: datetime | None = None

    @field_validator("start_utc", "end_utc")
    @classmethod
    def _require_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return value
        return _as_utc(value)

    @model_validator(mode="after")
    def _reject_inverted(self) -> "Interval":
        if self.end_utc is not None and self.end_utc <= self.start_utc:
            raise ValueError("interval end must be after start for completed intervals")
        return self


class TimeContext(ContractModel):
    """Resolved investigation time context. G3-A §5.1."""

    timezone: str = Field(min_length=1)
    start_utc: datetime
    end_utc: datetime
    original_expression: str | None = None

    @field_validator("start_utc", "end_utc")
    @classmethod
    def _require_utc(cls, value: datetime) -> datetime:
        return _as_utc(value)

    @field_validator("timezone")
    @classmethod
    def _validate_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except ZoneInfoNotFoundError as exc:
            raise ValueError(f"unknown timezone: {value}") from exc
        return value

    @model_validator(mode="after")
    def _reject_inverted(self) -> "TimeContext":
        if self.end_utc <= self.start_utc:
            raise ValueError("time context end must be after start")
        return self
