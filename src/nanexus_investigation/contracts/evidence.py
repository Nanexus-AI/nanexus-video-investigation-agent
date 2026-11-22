"""Observation and EvidenceRef. G3-A §5.2 and §5.3."""

from datetime import datetime
from typing import Any

from pydantic import Field, field_validator, model_validator

from nanexus_investigation.contracts.enums import (
    EvidenceAvailability,
    EvidenceKind,
    ObservationType,
)
from nanexus_investigation.contracts.identity import (
    ContractModel,
    Interval,
    SourceLocator,
    _as_utc,
)


class Observation(ContractModel):
    """Normalized statement about data returned by a source. G3-A §5.2."""

    observation_id: str = Field(min_length=1)
    source_locator: SourceLocator
    source_revision: str = Field(min_length=1)
    observation_type: ObservationType
    camera_id: str = Field(min_length=1)
    start_utc: datetime
    end_utc: datetime | None = None
    labels: tuple[str, ...] = ()
    zones: tuple[str, ...] = ()
    attributes: dict[str, Any] = Field(default_factory=dict)
    retrieved_at: datetime
    provenance: str = Field(min_length=1)

    @field_validator("observation_id", "source_revision", "camera_id", "provenance")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if value.strip() != value or not value.strip():
            raise ValueError("observation fields must be non-empty and trimmed")
        return value

    @field_validator("start_utc", "end_utc", "retrieved_at")
    @classmethod
    def _require_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return value
        return _as_utc(value)

    @model_validator(mode="after")
    def _reject_inverted(self) -> "Observation":
        if self.end_utc is not None and self.end_utc <= self.start_utc:
            raise ValueError("observation end must be after start")
        return self


class EvidenceRef(ContractModel):
    """Traceable evidence reference. G3-A §5.3.

    Does not store media bytes. ``content_sha256`` must not be fabricated:
    supply it only after bytes were actually hashed. ``content_type`` records
    an observed MIME type when known; it is not proof of content integrity.
    Requiring ``content_type`` whenever ``content_sha256`` is set is a local
    serialization rule, not a separate G3 architectural requirement.
    """

    evidence_id: str = Field(min_length=1)
    source_locator: SourceLocator
    evidence_kind: EvidenceKind
    camera_id: str = Field(min_length=1)
    interval: Interval | None = None
    retrieval_spec: dict[str, str | int | float | bool | None] = Field(
        default_factory=dict
    )
    availability: EvidenceAvailability = EvidenceAvailability.NOT_CHECKED
    checked_at: datetime | None = None
    metadata_fingerprint: str | None = None
    content_sha256: str | None = None
    content_type: str | None = None
    provenance: str = Field(min_length=1)

    @field_validator("evidence_id", "camera_id", "provenance")
    @classmethod
    def _reject_blank(cls, value: str) -> str:
        if value.strip() != value or not value.strip():
            raise ValueError("evidence fields must be non-empty and trimmed")
        return value

    @field_validator("checked_at")
    @classmethod
    def _require_utc(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return value
        return _as_utc(value)

    @field_validator("content_sha256")
    @classmethod
    def _validate_hash(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if value.strip() != value or not value.strip():
            raise ValueError("content_sha256 must be non-empty when provided")
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value.lower()):
            raise ValueError("content_sha256 must be a 64-character hex digest")
        return value.lower()

    @field_validator("retrieval_spec")
    @classmethod
    def _reject_url_locators(
        cls, value: dict[str, str | int | float | bool | None]
    ) -> dict[str, str | int | float | bool | None]:
        for key, item in value.items():
            if key.lower() in {"url", "uri", "href", "path", "file"}:
                raise ValueError(
                    "retrieval_spec must use canonical read parameters, not URLs"
                )
            if isinstance(item, str) and (
                "://" in item or item.startswith("file:") or item.startswith("/")
            ):
                raise ValueError(
                    "retrieval_spec must not contain URL or filesystem paths"
                )
        return value

    @model_validator(mode="after")
    def _hash_requires_type(self) -> "EvidenceRef":
        # Local format rule: a recorded digest should name the observed type.
        # This does not verify bytes or prove integrity.
        if self.content_sha256 is not None and not self.content_type:
            raise ValueError("content_sha256 requires an observed content_type")
        return self
