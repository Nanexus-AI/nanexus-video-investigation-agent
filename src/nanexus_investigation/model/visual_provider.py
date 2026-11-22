"""Local Vision-Language Model provider for selective visual observation (T5).

Planning remains text-only via ModelProvider. This module is invoked only by
the ``observe_media`` tool after authorized, integrity-verified retained media
is resolved. No cloud fallback.
"""

from __future__ import annotations

import base64
import hashlib
import json
import subprocess
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

VisualOutcome = Literal["ok", "uncertain", "unable", "error"]

DEFAULT_PROMPT_TEMPLATE_ID = "observe_snapshot_v1"
DEFAULT_MAX_IMAGE_BYTES = 5 * 1024 * 1024
DEFAULT_MAX_RESPONSE_CHARS = 2000
DEFAULT_MAX_IMAGE_EDGE = 640
FALLBACK_PROMPT = "Describe the image."


class VisualProviderError(RuntimeError):
    """Local visual inference failed without producing a usable observation."""


@dataclass(frozen=True, slots=True)
class VisualObservationRequest:
    """Bounded visual question over integrity-verified image bytes."""

    objective: str
    image_bytes: bytes
    content_type: str
    media_evidence_id: str
    retained_media_id: str
    content_sha256: str
    prompt_template_id: str = DEFAULT_PROMPT_TEMPLATE_ID
    max_response_chars: int = DEFAULT_MAX_RESPONSE_CHARS


@dataclass(frozen=True, slots=True)
class NormalizedSnapshot:
    """Inference-time image input. Does not replace retained-media authority."""

    bytes: bytes
    content_sha256: str
    byte_count: int
    width: int | None
    height: int | None
    preprocessing: str
    max_edge: int


@dataclass(frozen=True, slots=True)
class VisualObservationResult:
    """Structured local-VLM outcome. Never a physical-world fact claim."""

    outcome: VisualOutcome
    observation_text: str
    uncertainty: str | None
    model_name: str
    model_digest: str | None
    prompt_template_id: str
    limitations: tuple[str, ...]
    raw_response_chars: int = 0
    source_content_sha256: str | None = None
    inference_input_sha256: str | None = None
    inference_input_byte_count: int | None = None
    inference_input_width: int | None = None
    inference_input_height: int | None = None
    preprocessing: str | None = None
    preprocessing_max_edge: int | None = None
    prompt_text: str | None = None
    used_fallback_prompt: bool = False
    temperature: float | None = None
    num_predict: int | None = None


class VisualProvider(Protocol):
    """Local visual inference backend."""

    @property
    def model_name(self) -> str: ...

    def is_available(self) -> bool: ...

    def observe(self, request: VisualObservationRequest) -> VisualObservationResult: ...


def _normalize_objective(objective: str) -> str:
    text = (objective or "").strip()
    if not text:
        raise VisualProviderError("visual objective must be non-empty")
    if len(text) > 500:
        raise VisualProviderError("visual objective exceeds 500 characters")
    return text


def _build_user_prompt(objective: str) -> str:
    # moondream:1.8b frequently returns empty content for yes/no or long
    # instruction wrappers. Descriptive prompts are reliable; claim/report
    # gates retain the non-fact semantics.
    return f"Describe the image. Focus on: {objective}"


def _jpeg_dimensions(payload: bytes) -> tuple[int | None, int | None]:
    """Best-effort SOF0/SOF2 dimension parse without external image libs."""
    if len(payload) < 4 or payload[:2] != b"\xff\xd8":
        return None, None
    i = 2
    while i + 9 < len(payload):
        if payload[i] != 0xFF:
            i += 1
            continue
        marker = payload[i + 1]
        if marker in {0xD8, 0xD9}:
            i += 2
            continue
        if i + 4 > len(payload):
            break
        length = int.from_bytes(payload[i + 2 : i + 4], "big")
        if length < 2 or i + 2 + length > len(payload):
            break
        # SOF0 / SOF2 baseline/progressive.
        if marker in {0xC0, 0xC2} and length >= 7:
            height = int.from_bytes(payload[i + 5 : i + 7], "big")
            width = int.from_bytes(payload[i + 7 : i + 9], "big")
            return width, height
        i += 2 + length
    return None, None


def normalize_snapshot_for_vlm(
    image_bytes: bytes,
    *,
    max_edge: int = DEFAULT_MAX_IMAGE_EDGE,
    timeout_s: float = 15.0,
) -> NormalizedSnapshot:
    """Bounded local JPEG re-encode for VLM compatibility.

    Some source JPEGs cause empty moondream responses; a controlled ffmpeg
    re-encode preserves visual content without modifying retained source bytes.
    """
    if not image_bytes:
        raise VisualProviderError("empty image bytes")
    if max_edge < 32 or max_edge > 1280:
        raise VisualProviderError("max_edge out of allowed range")
    with tempfile.TemporaryDirectory(prefix="nx-vlm-") as tmp:
        src = Path(tmp) / "src.jpg"
        dst = Path(tmp) / "norm.jpg"
        src.write_bytes(image_bytes)
        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-vf",
            f"scale='min({max_edge},iw)':-2",
            "-frames:v",
            "1",
            "-q:v",
            "5",
            str(dst),
        ]
        try:
            completed = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                timeout=timeout_s,
            )
        except FileNotFoundError as exc:
            raise VisualProviderError(
                "ffmpeg is required for bounded snapshot normalization"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise VisualProviderError("snapshot normalization timed out") from exc
        if completed.returncode != 0 or not dst.is_file():
            raise VisualProviderError("snapshot normalization failed")
        out = dst.read_bytes()
        if not out:
            raise VisualProviderError("normalized snapshot is empty")
        width, height = _jpeg_dimensions(out)
        digest = hashlib.sha256(out).hexdigest()
        return NormalizedSnapshot(
            bytes=out,
            content_sha256=digest,
            byte_count=len(out),
            width=width,
            height=height,
            preprocessing="ffmpeg_reencode_q5_max_edge",
            max_edge=max_edge,
        )


@dataclass
class StaticVisualProvider:
    """Deterministic fixture provider for tests. No network."""

    observation_text: str = (
        "The image appears to show a possible vehicle in a driveway; details are limited."
    )
    uncertainty: str | None = (
        "Local fixture interpretation only; not an independent physical-world fact."
    )
    outcome: VisualOutcome = "ok"
    model_name: str = "static-vlm:fixture"
    model_digest: str | None = "digest:static-fixture"
    available: bool = True
    raise_error: bool = False
    empty_response: bool = False

    def is_available(self) -> bool:
        return self.available

    def observe(self, request: VisualObservationRequest) -> VisualObservationResult:
        if not self.available:
            raise VisualProviderError("static visual provider is unavailable")
        if self.raise_error:
            raise VisualProviderError("static visual provider forced failure")
        _normalize_objective(request.objective)
        if not request.image_bytes:
            raise VisualProviderError("empty image bytes")
        if self.empty_response:
            raise VisualProviderError("empty visual model response")
        text = self.observation_text.strip()
        if len(text) > request.max_response_chars:
            text = text[: request.max_response_chars].rstrip() + "…"
        source_digest = request.content_sha256 or hashlib.sha256(
            request.image_bytes
        ).hexdigest()
        width, height = _jpeg_dimensions(request.image_bytes)
        return VisualObservationResult(
            outcome=self.outcome,
            observation_text=text,
            uncertainty=self.uncertainty,
            model_name=self.model_name,
            model_digest=self.model_digest,
            prompt_template_id=request.prompt_template_id,
            limitations=(
                "Static fixture observation; not a live VLM inference.",
                "Model interpretation is not an independently verified fact.",
            ),
            raw_response_chars=len(text),
            source_content_sha256=source_digest,
            inference_input_sha256=source_digest,
            inference_input_byte_count=len(request.image_bytes),
            inference_input_width=width,
            inference_input_height=height,
            preprocessing="none",
            preprocessing_max_edge=None,
            prompt_text=_build_user_prompt(request.objective.strip()),
            used_fallback_prompt=False,
            temperature=None,
            num_predict=None,
        )


@dataclass
class OllamaVisualProvider:
    """Local Ollama multimodal chat provider (image + text)."""

    model: str = "moondream:1.8b"
    base_url: str = "http://127.0.0.1:11434"
    timeout_s: float = 120.0
    temperature: float = 0.0
    num_predict: int = 256
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES
    _cached_digest: str | None = None
    _availability: bool | None = None

    @property
    def model_name(self) -> str:
        return self.model

    def is_available(self) -> bool:
        if self._availability is not None:
            return self._availability
        try:
            req = urllib.request.Request(
                f"{self.base_url.rstrip('/')}/api/tags",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=5.0) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            names = {
                m.get("name")
                for m in payload.get("models", [])
                if isinstance(m, dict)
            }
            # Accept exact or tag-family match (moondream:1.8b / moondream:latest).
            family = self.model.split(":", 1)[0]
            available = self.model in names or any(
                isinstance(n, str) and n.split(":", 1)[0] == family for n in names
            )
            self._availability = available
            return available
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            self._availability = False
            return False

    def refresh_model_digest(self) -> str | None:
        try:
            req = urllib.request.Request(
                f"{self.base_url.rstrip('/')}/api/tags",
                method="GET",
            )
            with urllib.request.urlopen(req, timeout=10.0) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            family = self.model.split(":", 1)[0]
            for item in payload.get("models", []):
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                digest = item.get("digest")
                if name == self.model and isinstance(digest, str) and digest.strip():
                    self._cached_digest = digest.strip()[:128]
                    return self._cached_digest
                if (
                    isinstance(name, str)
                    and name.split(":", 1)[0] == family
                    and isinstance(digest, str)
                    and digest.strip()
                    and self._cached_digest is None
                ):
                    self._cached_digest = digest.strip()[:128]
            return self._cached_digest
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return self._cached_digest

    def observe(self, request: VisualObservationRequest) -> VisualObservationResult:
        objective = _normalize_objective(request.objective)
        if not request.image_bytes:
            raise VisualProviderError("empty image bytes")
        if len(request.image_bytes) > self.max_image_bytes:
            raise VisualProviderError(
                f"image exceeds max_image_bytes={self.max_image_bytes}"
            )
        if not request.content_type.lower().startswith("image/"):
            raise VisualProviderError(
                f"unsupported visual content type: {request.content_type}"
            )
        if not self.is_available():
            raise VisualProviderError(
                f"local visual model is unavailable: {self.model}"
            )

        digest = self._cached_digest or self.refresh_model_digest()
        source_digest = request.content_sha256 or hashlib.sha256(
            request.image_bytes
        ).hexdigest()
        try:
            normalized = normalize_snapshot_for_vlm(request.image_bytes)
        except VisualProviderError:
            # Fall back to original bytes when normalization is unavailable.
            width, height = _jpeg_dimensions(request.image_bytes)
            normalized = NormalizedSnapshot(
                bytes=request.image_bytes,
                content_sha256=hashlib.sha256(request.image_bytes).hexdigest(),
                byte_count=len(request.image_bytes),
                width=width,
                height=height,
                preprocessing="none",
                max_edge=DEFAULT_MAX_IMAGE_EDGE,
            )
        if normalized.byte_count > self.max_image_bytes:
            raise VisualProviderError(
                f"normalized image exceeds max_image_bytes={self.max_image_bytes}"
            )
        encoded = base64.b64encode(normalized.bytes).decode("ascii")
        primary_prompt = _build_user_prompt(objective)
        # Prefer /api/generate for broad moondream compatibility.
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "prompt": primary_prompt,
            "images": [encoded],
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
        }
        body = json.dumps(payload).encode("utf-8")
        http_req = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(http_req, timeout=self.timeout_s) as resp:
                raw = resp.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            raise VisualProviderError(
                f"local VLM HTTP error: {exc.code}"
            ) from exc
        except urllib.error.URLError as exc:
            raise VisualProviderError(
                f"local VLM unavailable: {exc.reason}"
            ) from exc

        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise VisualProviderError("malformed local VLM JSON response") from exc

        content = decoded.get("response")
        if not isinstance(content, str) or not content.strip():
            # Fallback to chat schema if a newer model returns message.content.
            message = decoded.get("message") or {}
            content = message.get("content") if isinstance(message, dict) else None
        used_fallback_prompt = False
        prompt_text = primary_prompt
        if not isinstance(content, str) or not content.strip():
            # Single bounded fallback: generic describe-the-image prompt only.
            fallback_payload = dict(payload)
            fallback_payload["prompt"] = FALLBACK_PROMPT
            fallback_body = json.dumps(fallback_payload).encode("utf-8")
            fallback_req = urllib.request.Request(
                f"{self.base_url.rstrip('/')}/api/generate",
                data=fallback_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(
                    fallback_req, timeout=self.timeout_s
                ) as resp:
                    fallback_raw = resp.read().decode("utf-8")
                fallback_decoded = json.loads(fallback_raw)
                content = fallback_decoded.get("response")
                used_fallback_prompt = True
                prompt_text = FALLBACK_PROMPT
            except (
                urllib.error.URLError,
                urllib.error.HTTPError,
                json.JSONDecodeError,
                TimeoutError,
                OSError,
            ):
                content = None
        if not isinstance(content, str) or not content.strip():
            raise VisualProviderError("empty or malformed local VLM content")
        text = content.strip()
        raw_chars = len(text)
        if len(text) > request.max_response_chars:
            text = text[: request.max_response_chars].rstrip() + "…"

        lower = text.lower()
        uncertain_markers = (
            "unclear",
            "insufficient",
            "cannot determine",
            "can't determine",
            "unable to",
            "not sure",
            "uncertain",
            "too blurry",
            "low quality",
        )
        outcome: VisualOutcome = "ok"
        uncertainty: str | None = None
        if any(marker in lower for marker in uncertain_markers):
            outcome = "uncertain"
            uncertainty = "Model reported visual uncertainty or insufficient detail."

        limitations = [
            "Local VLM interpretation over retained media bytes only.",
            "Not an independently verified physical-world fact.",
            "Does not establish identity, ownership, causation, or interaction.",
            "Model statements about absence are not verified real-world absence.",
        ]
        if normalized.preprocessing != "none":
            limitations.append(
                "Snapshot was locally re-encoded for VLM input; EvidenceRef "
                "content_sha256 still refers to the retained source bytes."
            )
        if used_fallback_prompt:
            limitations.append(
                "Primary objective prompt returned empty model content; "
                "fell back once to a generic describe-the-image prompt."
            )
            if uncertainty is None:
                uncertainty = (
                    "Fallback descriptive prompt used after empty objective response."
                )
            if outcome == "ok":
                outcome = "uncertain"
        return VisualObservationResult(
            outcome=outcome,
            observation_text=text,
            uncertainty=uncertainty,
            model_name=self.model,
            model_digest=digest,
            prompt_template_id=request.prompt_template_id,
            limitations=tuple(limitations),
            raw_response_chars=raw_chars,
            source_content_sha256=source_digest,
            inference_input_sha256=normalized.content_sha256,
            inference_input_byte_count=normalized.byte_count,
            inference_input_width=normalized.width,
            inference_input_height=normalized.height,
            preprocessing=normalized.preprocessing,
            preprocessing_max_edge=normalized.max_edge,
            prompt_text=prompt_text,
            used_fallback_prompt=used_fallback_prompt,
            temperature=self.temperature,
            num_predict=self.num_predict,
        )


def detect_local_visual_provider(
    *,
    preferred_models: tuple[str, ...] = ("moondream:1.8b", "moondream", "llava:7b"),
    base_url: str = "http://127.0.0.1:11434",
) -> OllamaVisualProvider | None:
    """Return an available local vision model provider, or None."""
    for name in preferred_models:
        provider = OllamaVisualProvider(model=name, base_url=base_url)
        if provider.is_available():
            provider.refresh_model_digest()
            return provider
    return None
