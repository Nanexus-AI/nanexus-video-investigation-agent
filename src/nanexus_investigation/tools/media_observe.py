"""Selective visual observation tool over retained media (T5).

``observe_media`` invokes an approved local VLM against Case-owned,
integrity-verified SNAPSHOT bytes from LocalMediaStore. It never accepts
model-supplied URLs/paths/hosts, never fabricates Frigate Observations, and
never promotes interpretations to physical-world facts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from nanexus_investigation.contracts.enums import (
    EvidenceKind,
    SearchCoverage,
    ToolErrorCode,
    ToolResultStatus,
)
from nanexus_investigation.contracts.evidence import EvidenceRef
from nanexus_investigation.contracts.scope import InvestigationScope
from nanexus_investigation.contracts.tool import ToolError, ToolRequest, ToolResult
from nanexus_investigation.evidence.media_store import LocalMediaStore, MediaStoreError
from nanexus_investigation.model.visual_provider import (
    DEFAULT_PROMPT_TEMPLATE_ID,
    VisualObservationRequest,
    VisualProvider,
    VisualProviderError,
)

UTC = timezone.utc
OBSERVE_TOOL = "observe_media"
ALLOWED_OBSERVE_KINDS = frozenset({EvidenceKind.SNAPSHOT})
# Clip frame sampling is intentionally deferred for T5 snapshot-first scope.
DEFERRED_CLIP_MESSAGE = (
    "observe_media for CLIP evidence is deferred in T5; snapshot visual "
    "verification is the supported path"
)


class MediaObserveError(ValueError):
    """Authorization or argument failure before visual inference."""


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
        provenance="adapter:local-vlm",
        usage={"tool_attempts": 1},
    )


def observe_media(
    request: ToolRequest,
    *,
    evidence_by_id: dict[str, EvidenceRef],
    scope: InvestigationScope,
    case_id: str,
    media_store: LocalMediaStore,
    visual_provider: VisualProvider | None,
) -> ToolResult:
    """Run local visual observation on an authorized retained media EvidenceRef."""
    execution_id = f"exec-observe-{uuid4().hex[:12]}"
    source_id = request.source_id or scope.vms_instance_id
    args = dict(request.arguments or {})

    for banned in (
        "url",
        "uri",
        "href",
        "path",
        "file",
        "hostname",
        "host",
        "base_url",
        "retained_media_id",
        "endpoint",
        "model_url",
    ):
        if banned in args and args[banned] is not None:
            return _error_result(
                execution_id=execution_id,
                code=ToolErrorCode.INVALID_SCOPE,
                message=f"observe_media rejects model-supplied {banned}",
                source_id=source_id,
            )

    if visual_provider is None or not visual_provider.is_available():
        return _error_result(
            execution_id=execution_id,
            code=ToolErrorCode.DISABLED_CAPABILITY,
            message="local visual model backend is unavailable",
            source_id=source_id,
            status=ToolResultStatus.UNSUPPORTED,
        )

    try:
        evidence_id = args.get("evidence_id")
        if not isinstance(evidence_id, str) or not evidence_id.strip():
            raise MediaObserveError("evidence_id is required")
        evidence_id = evidence_id.strip()
        objective = args.get("objective") or args.get("question")
        if not isinstance(objective, str) or not objective.strip():
            raise MediaObserveError("objective is required")
        objective = objective.strip()
        if len(objective) > 500:
            raise MediaObserveError("objective exceeds 500 characters")

        evidence = evidence_by_id.get(evidence_id)
        if evidence is None:
            raise MediaObserveError(
                f"unknown or foreign evidence_id: {evidence_id}"
            )
        if evidence.camera_id not in scope.camera_ids:
            raise MediaObserveError(
                f"media evidence camera is outside authorized scope: {evidence.camera_id}"
            )
        if evidence.source_locator.vms_instance_id != scope.vms_instance_id:
            raise MediaObserveError(
                "media evidence VMS instance does not match investigation scope"
            )
        if evidence.evidence_kind is EvidenceKind.CLIP:
            return _error_result(
                execution_id=execution_id,
                code=ToolErrorCode.DISABLED_CAPABILITY,
                message=DEFERRED_CLIP_MESSAGE,
                source_id=source_id,
                status=ToolResultStatus.UNSUPPORTED,
            )
        if evidence.evidence_kind not in ALLOWED_OBSERVE_KINDS:
            raise MediaObserveError(
                f"observe_media requires SNAPSHOT evidence; got {evidence.evidence_kind.value}"
            )
        retained_media_id = evidence.retrieval_spec.get("retained_media_id")
        if not isinstance(retained_media_id, str) or not retained_media_id:
            raise MediaObserveError(
                "media evidence lacks retained_media_id; resolve_media must succeed first"
            )
        if not evidence.content_sha256 or not evidence.content_type:
            raise MediaObserveError(
                "media evidence requires content_sha256 and content_type"
            )
    except MediaObserveError as exc:
        return _error_result(
            execution_id=execution_id,
            code=ToolErrorCode.INVALID_SCOPE,
            message=str(exc),
            source_id=source_id,
        )

    try:
        retained = media_store.get(
            case_id=case_id, retained_media_id=retained_media_id
        )
    except MediaStoreError as exc:
        return _error_result(
            execution_id=execution_id,
            code=ToolErrorCode.MISSING_MEDIA,
            message=f"retained media reopen failed: {exc}",
            source_id=source_id,
            status=ToolResultStatus.EMPTY,
        )

    if retained.content_sha256 != evidence.content_sha256:
        return _error_result(
            execution_id=execution_id,
            code=ToolErrorCode.INVALID_RESPONSE,
            message="retained media hash does not match EvidenceRef content_sha256",
            source_id=source_id,
        )
    if retained.byte_count < 1 or not retained.bytes:
        return _error_result(
            execution_id=execution_id,
            code=ToolErrorCode.INVALID_RESPONSE,
            message="retained media payload is empty",
            source_id=source_id,
        )

    try:
        vision = visual_provider.observe(
            VisualObservationRequest(
                objective=objective,
                image_bytes=retained.bytes,
                content_type=evidence.content_type or retained.content_type,
                media_evidence_id=evidence.evidence_id,
                retained_media_id=retained.retained_media_id,
                content_sha256=retained.content_sha256,
                prompt_template_id=DEFAULT_PROMPT_TEMPLATE_ID,
            )
        )
    except VisualProviderError as exc:
        return _error_result(
            execution_id=execution_id,
            code=ToolErrorCode.INVALID_RESPONSE,
            message=f"local visual inference failed: {exc}",
            source_id=source_id,
        )

    if vision.outcome == "error" or not vision.observation_text.strip():
        return _error_result(
            execution_id=execution_id,
            code=ToolErrorCode.INVALID_RESPONSE,
            message="local visual model returned no usable observation",
            source_id=source_id,
        )

    retrieved_at = datetime.now(tz=UTC)
    # Retained source hash remains authoritative; inference_input_* describes
    # the bytes actually supplied to the local VLM after optional normalize.
    record: dict[str, Any] = {
        "record_kind": "visual_observation",
        "media_evidence_id": evidence.evidence_id,
        "source_evidence_id": evidence.retrieval_spec.get("source_evidence_id"),
        "retained_media_id": retained.retained_media_id,
        "content_sha256": retained.content_sha256,
        "content_type": retained.content_type,
        "byte_count": retained.byte_count,
        "media_kind": "snapshot",
        "objective": objective,
        "observation_text": vision.observation_text,
        "uncertainty": vision.uncertainty,
        "outcome": vision.outcome,
        "model_name": vision.model_name,
        "model_digest": vision.model_digest,
        "prompt_template_id": vision.prompt_template_id,
        "prompt_text": vision.prompt_text,
        "used_fallback_prompt": bool(vision.used_fallback_prompt),
        "temperature": vision.temperature,
        "num_predict": vision.num_predict,
        "source_content_sha256": vision.source_content_sha256 or retained.content_sha256,
        "inference_input_sha256": vision.inference_input_sha256,
        "inference_input_byte_count": vision.inference_input_byte_count,
        "inference_input_width": vision.inference_input_width,
        "inference_input_height": vision.inference_input_height,
        "preprocessing": vision.preprocessing,
        "preprocessing_max_edge": vision.preprocessing_max_edge,
        "limitations": list(vision.limitations),
        "physical_world_fact": False,
        "visual_interpretation": True,
        "sampled_frames": None,
        "retrieved_at": retrieved_at.isoformat(),
    }
    return ToolResult(
        status=ToolResultStatus.OK,
        execution_id=execution_id,
        data=[record],
        source_id=source_id,
        retrieved_at=retrieved_at,
        coverage=SearchCoverage.EXHAUSTED,
        provenance=f"adapter:local-vlm:{vision.model_name}",
        usage={
            "tool_attempts": 1,
            "byte_count": retained.byte_count,
            "response_chars": vision.raw_response_chars,
        },
    )
