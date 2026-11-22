"""Controlled local retention for acquired media bytes (T4).

Stores opaque media objects under a private root. EvidenceRef.retrieval_spec
references only ``retained_media_id`` — never filesystem paths or URLs.
SHA-256 verifies byte identity on reopen; it does not prove scene authenticity.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

_SAFE_ID = re.compile(r"^[a-zA-Z0-9._-]{1,128}$")
UTC = timezone.utc


class MediaStoreError(ValueError):
    """Local media retention or integrity failure."""


@dataclass(frozen=True, slots=True)
class RetainedMedia:
    """Integrity-checked retained media object."""

    retained_media_id: str
    case_id: str
    content_sha256: str
    content_type: str
    byte_count: int
    media_kind: str
    stored_at: datetime
    bytes: bytes


class LocalMediaStore:
    """Minimal case-scoped local media retention with SHA-256 integrity checks."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _assert_under(self, path: Path, parent: Path) -> Path:
        resolved = path.resolve()
        root = parent.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise MediaStoreError(
                "media path escapes configured private storage area"
            ) from exc
        return resolved

    def _case_dir(self, case_id: str) -> Path:
        if not _SAFE_ID.match(case_id):
            raise MediaStoreError("case_id is not a safe media-store key")
        path = self._assert_under(self.root / case_id, self.root)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _object_paths(
        self, *, case_id: str, retained_media_id: str
    ) -> tuple[Path, Path]:
        if not _SAFE_ID.match(retained_media_id):
            raise MediaStoreError("retained_media_id is not a safe media-store key")
        case_dir = self._case_dir(case_id)
        bin_path = self._assert_under(case_dir / f"{retained_media_id}.bin", case_dir)
        meta_path = self._assert_under(
            case_dir / f"{retained_media_id}.json", case_dir
        )
        return bin_path, meta_path

    def put(
        self,
        *,
        case_id: str,
        retained_media_id: str,
        media_kind: str,
        content_type: str,
        payload: bytes,
    ) -> RetainedMedia:
        if not payload:
            raise MediaStoreError("refusing to retain empty media payload")
        digest = hashlib.sha256(payload).hexdigest()
        bin_path, meta_path = self._object_paths(
            case_id=case_id, retained_media_id=retained_media_id
        )
        if bin_path.exists() or meta_path.exists():
            existing = self.get(case_id=case_id, retained_media_id=retained_media_id)
            if existing.content_sha256 != digest:
                raise MediaStoreError(
                    "retained_media_id already exists with different content hash"
                )
            return existing
        meta = {
            "retained_media_id": retained_media_id,
            "case_id": case_id,
            "content_sha256": digest,
            "content_type": content_type,
            "byte_count": len(payload),
            "media_kind": media_kind,
            "stored_at": datetime.now(tz=UTC).isoformat(),
        }
        # Write payload first, then metadata marker for reopen integrity.
        bin_path.write_bytes(payload)
        meta_path.write_text(json.dumps(meta, sort_keys=True), encoding="utf-8")
        return RetainedMedia(
            retained_media_id=retained_media_id,
            case_id=case_id,
            content_sha256=digest,
            content_type=content_type,
            byte_count=len(payload),
            media_kind=media_kind,
            stored_at=datetime.fromisoformat(meta["stored_at"]),
            bytes=payload,
        )

    def get(self, *, case_id: str, retained_media_id: str) -> RetainedMedia:
        if not _SAFE_ID.match(case_id) or not _SAFE_ID.match(retained_media_id):
            raise MediaStoreError("unsafe media-store lookup key")
        bin_path, meta_path = self._object_paths(
            case_id=case_id, retained_media_id=retained_media_id
        )
        if not bin_path.is_file() or not meta_path.is_file():
            raise MediaStoreError("retained media object is missing")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        payload = bin_path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        expected = str(meta.get("content_sha256") or "")
        if digest != expected:
            raise MediaStoreError(
                "retained media integrity check failed: content hash mismatch"
            )
        if int(meta.get("byte_count") or -1) != len(payload):
            raise MediaStoreError(
                "retained media integrity check failed: byte_count mismatch"
            )
        return RetainedMedia(
            retained_media_id=retained_media_id,
            case_id=case_id,
            content_sha256=digest,
            content_type=str(meta.get("content_type") or ""),
            byte_count=len(payload),
            media_kind=str(meta.get("media_kind") or ""),
            stored_at=datetime.fromisoformat(str(meta["stored_at"])),
            bytes=payload,
        )
