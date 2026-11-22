"""Selective authorized media retrieval tool (T4).

``resolve_media`` acquires snapshot or event-clip bytes for an already-registered
Case EvidenceRef / Observation. It never interprets visual content, never accepts
model-supplied URLs/paths/hosts, and never fabricates media when unavailable.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import uuid4

from nanexus_investigation.contracts.enums import (
    EvidenceAvailability,
    EvidenceKind,
    SearchCoverage,
    ToolErrorCode,
    ToolResultStatus,
)
from nanexus_investigation.contracts.evidence import EvidenceRef, Observation
from nanexus_investigation.contracts.scope import InvestigationScope
from nanexus_investigation.contracts.tool import ToolError, ToolRequest, ToolResult
from nanexus_investigation.evidence.media_store import LocalMediaStore, MediaStoreError
from nanexus_investigation.vms.frigate.client import (
    DEFAULT_MAX_CLIP_BYTES,
    DEFAULT_MAX_SNAPSHOT_BYTES,
    FrigateAuthError,
    FrigateMediaError,
    FrigateMediaTooLargeError,
    FrigateMediaTypeError,
    FrigateNotFoundError,
    FrigateProtocolError,
    FrigateReadOnlyClient,
    FrigateScopeError,
    FrigateUnavailableError,
)

UTC = timezone.utc
MEDIA_TOOL = "resolve_media"
ALLOWED_MEDIA_KINDS = frozenset({"snapshot", "clip"})


class MediaResolveError(ValueError):
    """Authorization or argument failure before media transport."""


class _MediaTransport(Protocol):
    def get_event_snapshot(self, event_id: str, *, max_bytes: int = ...): ...
    def get_event_clip(self, event_id: str, *, max_bytes: int = ...): ...


def _error_result(
    *,
    execution_id: str,
    code: ToolErrorCode,
    message: str,
    source_id: str,
    status: ToolResultStatus = ToolResultStatus.ERROR,
) -> ToolResult:
    return ToolResult(
        status=status,
        execution_id=execution_id,
        error=ToolError(code=code, message=message),
        source_id=source_id,
        retrieved_at=datetime.now(tz=UTC),
        coverage=SearchCoverage.UNKNOWN,
        provenance="adapter:frigate-media",
        usage={"tool_attempts": 1},
    )


def _lookup_source(
    *,
    evidence_id: str | None,
    observation_id: str | None,
    evidence_by_id: dict[str, EvidenceRef],
    observations_by_id: dict[str, Observation],
) -> tuple[EvidenceRef | None, Observation | None, str]:
    """Resolve Case-bound source evidence and native event id."""
    if evidence_id and observation_id:
        raise MediaResolveError(
            "provide either evidence_id or observation_id, not both"
        )
    if not evidence_id and not observation_id:
        raise MediaResolveError("evidence_id or observation_id is required")

    evidence: EvidenceRef | None = None
    observation: Observation | None = None
    if evidence_id:
        evidence = evidence_by_id.get(evidence_id)
        if evidence is None:
            raise MediaResolveError(
                f"unknown or foreign evidence_id: {evidence_id}"
            )
        native = evidence.source_locator.native_record_id
        # Prefer a matching observation for camera/VMS cross-checks.
        for obs in observations_by_id.values():
            if obs.source_locator.native_record_id == native:
                observation = obs
                break
        return evidence, observation, native

    assert observation_id is not None
    observation = observations_by_id.get(observation_id)
    if observation is None:
        raise MediaResolveError(
            f"unknown or foreign observation_id: {observation_id}"
        )
    native = observation.source_locator.native_record_id
    for item in evidence_by_id.values():
        if item.source_locator.native_record_id == native:
            if item.evidence_kind is EvidenceKind.EVENT_METADATA:
                evidence = item
                break
            if evidence is None:
                evidence = item
    if evidence is None:
        raise MediaResolveError(
            f"no registered EvidenceRef for observation_id: {observation_id}"
        )
    return evidence, observation, native


def _assert_in_scope(
    *,
    evidence: EvidenceRef,
    observation: Observation | None,
    scope: InvestigationScope,
) -> None:
    camera_id = evidence.camera_id
    vms_id = evidence.source_locator.vms_instance_id
    if observation is not None:
        camera_id = observation.camera_id
        vms_id = observation.source_locator.vms_instance_id
        if observation.camera_id != evidence.camera_id:
            raise MediaResolveError("observation/evidence camera mismatch")
        if (
            observation.source_locator.vms_instance_id
            != evidence.source_locator.vms_instance_id
        ):
            raise MediaResolveError("observation/evidence VMS mismatch")
    if camera_id not in scope.camera_ids:
        raise MediaResolveError(
            f"media source camera is outside authorized scope: {camera_id}"
        )
    if vms_id != scope.vms_instance_id:
        raise MediaResolveError(
            "media source VMS instance does not match investigation scope"
        )


def resolve_media(
    request: ToolRequest,
    *,
    evidence_by_id: dict[str, EvidenceRef],
    observations_by_id: dict[str, Observation],
    scope: InvestigationScope,
    case_id: str,
    media_store: LocalMediaStore,
    transport: _MediaTransport | None = None,
    max_snapshot_bytes: int = DEFAULT_MAX_SNAPSHOT_BYTES,
    max_clip_bytes: int = DEFAULT_MAX_CLIP_BYTES,
) -> ToolResult:
    """Acquire authorized snapshot/clip bytes for a Case-bound evidence reference."""
    execution_id = f"exec-media-{uuid4().hex[:12]}"
    source_id = request.source_id or scope.vms_instance_id
    args = dict(request.arguments or {})

    # Hard reject model-supplied locators.
    for banned in ("url", "uri", "href", "path", "file", "hostname", "host", "base_url"):
        if banned in args and args[banned] is not None:
            return _error_result(
                execution_id=execution_id,
                code=ToolErrorCode.INVALID_SCOPE,
                message=f"resolve_media rejects model-supplied {banned}",
                source_id=source_id,
            )

    try:
        media_kind = args.get("media_kind")
        if not isinstance(media_kind, str) or media_kind.strip() not in ALLOWED_MEDIA_KINDS:
            raise MediaResolveError(
                f"media_kind must be one of {sorted(ALLOWED_MEDIA_KINDS)}"
            )
        media_kind = media_kind.strip()
        evidence_id = args.get("evidence_id")
        observation_id = args.get("observation_id")
        if evidence_id is not None and not isinstance(evidence_id, str):
            raise MediaResolveError("evidence_id must be a string when provided")
        if observation_id is not None and not isinstance(observation_id, str):
            raise MediaResolveError("observation_id must be a string when provided")
        evidence, observation, native_id = _lookup_source(
            evidence_id=evidence_id,
            observation_id=observation_id,
            evidence_by_id=evidence_by_id,
            observations_by_id=observations_by_id,
        )
        assert evidence is not None
        _assert_in_scope(evidence=evidence, observation=observation, scope=scope)
    except MediaResolveError as exc:
        return _error_result(
            execution_id=execution_id,
            code=ToolErrorCode.INVALID_SCOPE,
            message=str(exc),
            source_id=source_id,
        )

    if transport is None:
        return _error_result(
            execution_id=execution_id,
            code=ToolErrorCode.DISABLED_CAPABILITY,
            message="media transport is not bound for this investigation",
            source_id=source_id,
        )

    try:
        if media_kind == "snapshot":
            response = transport.get_event_snapshot(
                native_id, max_bytes=max_snapshot_bytes
            )
            expected_kind = EvidenceKind.SNAPSHOT
            default_type = "image/jpeg"
            allowed_types = frozenset({"image/jpeg", "image/jpg"})
        else:
            response = transport.get_event_clip(native_id, max_bytes=max_clip_bytes)
            expected_kind = EvidenceKind.CLIP
            default_type = "video/mp4"
            allowed_types = frozenset({"video/mp4", "application/mp4"})
    except FrigateNotFoundError as exc:
        return _error_result(
            execution_id=execution_id,
            code=ToolErrorCode.MISSING_MEDIA,
            message=str(exc),
            source_id=source_id,
            status=ToolResultStatus.EMPTY,
        )
    except FrigateMediaTooLargeError as exc:
        return _error_result(
            execution_id=execution_id,
            code=ToolErrorCode.INVALID_RESPONSE,
            message=str(exc),
            source_id=source_id,
        )
    except FrigateMediaTypeError as exc:
        return _error_result(
            execution_id=execution_id,
            code=ToolErrorCode.INVALID_RESPONSE,
            message=str(exc),
            source_id=source_id,
        )
    except FrigateAuthError as exc:
        return _error_result(
            execution_id=execution_id,
            code=ToolErrorCode.AUTHORIZATION_DENIED,
            message=str(exc),
            source_id=source_id,
        )
    except FrigateScopeError as exc:
        return _error_result(
            execution_id=execution_id,
            code=ToolErrorCode.INVALID_SCOPE,
            message=str(exc),
            source_id=source_id,
        )
    except FrigateUnavailableError as exc:
        return _error_result(
            execution_id=execution_id,
            code=ToolErrorCode.SOURCE_UNAVAILABLE,
            message=str(exc),
            source_id=source_id,
        )
    except (FrigateMediaError, FrigateProtocolError) as exc:
        return _error_result(
            execution_id=execution_id,
            code=ToolErrorCode.INVALID_RESPONSE,
            message=str(exc),
            source_id=source_id,
        )

    payload = response.body
    if not payload:
        return _error_result(
            execution_id=execution_id,
            code=ToolErrorCode.INVALID_RESPONSE,
            message=f"empty media body for {media_kind}",
            source_id=source_id,
        )
    content_type = (
        (response.headers.get("Content-Type") or response.headers.get("content-type") or "")
        .split(";", 1)[0]
        .strip()
        .lower()
        or default_type
    )
    if content_type not in allowed_types:
        return _error_result(
            execution_id=execution_id,
            code=ToolErrorCode.INVALID_RESPONSE,
            message=f"unexpected media content type for {media_kind}: {content_type}",
            source_id=source_id,
        )
    digest = hashlib.sha256(payload).hexdigest()
    retained_media_id = f"media-{digest[:24]}"
    try:
        retained = media_store.put(
            case_id=case_id,
            retained_media_id=retained_media_id,
            media_kind=media_kind,
            content_type=content_type,
            payload=payload,
        )
    except MediaStoreError as exc:
        return _error_result(
            execution_id=execution_id,
            code=ToolErrorCode.INVALID_RESPONSE,
            message=f"media retention failed: {exc}",
            source_id=source_id,
        )

    record: dict[str, Any] = {
        "record_kind": "media_artifact",
        "media_kind": media_kind,
        "evidence_kind": expected_kind.value,
        "source_evidence_id": evidence.evidence_id,
        "source_observation_id": None
        if observation is None
        else observation.observation_id,
        "native_record_id": native_id,
        "camera_id": evidence.camera_id,
        "vms_instance_id": evidence.source_locator.vms_instance_id,
        "content_type": retained.content_type,
        "content_sha256": retained.content_sha256,
        "byte_count": retained.byte_count,
        "retained_media_id": retained.retained_media_id,
        "bytes_retained": True,
        "visual_interpretation": False,
        "retrieved_at": response.retrieved_at.isoformat(),
    }
    return ToolResult(
        status=ToolResultStatus.OK,
        execution_id=execution_id,
        data=[record],
        source_id=source_id,
        retrieved_at=response.retrieved_at,
        coverage=SearchCoverage.EXHAUSTED,
        provenance="adapter:frigate-media",
        usage={
            "tool_attempts": 1,
            "byte_count": retained.byte_count,
        },
    )


def materialize_media_evidence_from_tool_data(
    data: list[dict[str, Any]],
    *,
    execution_id: str,
    source_evidence: EvidenceRef,
) -> list[EvidenceRef]:
    """Build SNAPSHOT/CLIP EvidenceRef contracts from OK resolve_media payloads."""
    results: list[EvidenceRef] = []
    for index, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        if item.get("record_kind") != "media_artifact":
            continue
        if item.get("bytes_retained") is not True:
            continue
        if item.get("visual_interpretation") is True:
            raise ValueError("resolve_media must not claim visual interpretation")
        kind = EvidenceKind(str(item["evidence_kind"]))
        if kind not in {EvidenceKind.SNAPSHOT, EvidenceKind.CLIP}:
            raise ValueError(f"unsupported media evidence kind: {kind}")
        digest = str(item["content_sha256"])
        content_type = str(item["content_type"])
        retained_media_id = str(item["retained_media_id"])
        byte_count = int(item["byte_count"])
        media_kind = str(item["media_kind"])
        retrieved_at = item.get("retrieved_at")
        checked_at = (
            datetime.now(tz=UTC)
            if not isinstance(retrieved_at, str)
            else datetime.fromisoformat(retrieved_at)
        )
        results.append(
            EvidenceRef(
                evidence_id=f"ev-media-{execution_id}-{index}",
                source_locator=source_evidence.source_locator,
                evidence_kind=kind,
                camera_id=source_evidence.camera_id,
                interval=source_evidence.interval,
                retrieval_spec={
                    "operation": MEDIA_TOOL,
                    "media_kind": media_kind,
                    "native_id": source_evidence.source_locator.native_record_id,
                    "source_evidence_id": source_evidence.evidence_id,
                    "retained_media_id": retained_media_id,
                    "byte_count": byte_count,
                    "bytes_retained": True,
                    "visual_interpretation": False,
                },
                availability=EvidenceAvailability.AVAILABLE,
                checked_at=checked_at,
                content_sha256=digest,
                content_type=content_type,
                provenance="adapter:frigate-media",
            )
        )
    return results
