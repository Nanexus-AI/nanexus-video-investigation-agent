"""Private recorded Frigate HTTP response store for offline replay.

Real surveillance-derived payloads must remain Git-ignored. This module stores
raw HTTP bodies with integrity hashes for identical live/replay normalization.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nanexus_investigation.vms.frigate.client import (
    FrigateClientConfig,
    FrigateHttpResponse,
    FrigateReadOnlyClient,
)


@dataclass(frozen=True, slots=True)
class RecordedResponse:
    """On-disk envelope for one Frigate HTTP response."""

    path: str
    query: dict[str, str]
    status_code: int
    body: bytes
    retrieved_at: datetime
    frigate_version: str | None
    vms_instance_id: str
    content_sha256: str
    request_method: str = "GET"

    def to_http_response(self) -> FrigateHttpResponse:
        return FrigateHttpResponse(
            status_code=self.status_code,
            path=self.path,
            query=dict(self.query),
            body=self.body,
            headers={},
            retrieved_at=self.retrieved_at,
            frigate_version=self.frigate_version,
        )


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_recorded_response(
    directory: Path,
    name: str,
    response: FrigateHttpResponse,
    *,
    vms_instance_id: str,
) -> Path:
    """Write a recorded response envelope + raw body under a private directory."""
    directory.mkdir(parents=True, exist_ok=True)
    digest = _sha256(response.body)
    envelope = {
        "schema_version": "frigate-recorded-v1",
        "name": name,
        "request_method": "GET",
        "path": response.path,
        "query": response.query,
        "status_code": response.status_code,
        "retrieved_at": response.retrieved_at.isoformat(),
        "frigate_version": response.frigate_version,
        "vms_instance_id": vms_instance_id,
        "content_sha256": digest,
        "body_file": f"{name}.body",
    }
    body_path = directory / f"{name}.body"
    meta_path = directory / f"{name}.json"
    body_path.write_bytes(response.body)
    meta_path.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n")
    return meta_path


def load_recorded_response(meta_path: Path) -> RecordedResponse:
    envelope = json.loads(meta_path.read_text())
    body_path = meta_path.parent / str(envelope["body_file"])
    body = body_path.read_bytes()
    digest = _sha256(body)
    if digest != envelope["content_sha256"]:
        raise ValueError(
            f"recorded body integrity mismatch for {meta_path.name}: "
            f"expected {envelope['content_sha256']}, got {digest}"
        )
    return RecordedResponse(
        path=str(envelope["path"]),
        query=dict(envelope.get("query") or {}),
        status_code=int(envelope["status_code"]),
        body=body,
        retrieved_at=datetime.fromisoformat(envelope["retrieved_at"]),
        frigate_version=envelope.get("frigate_version"),
        vms_instance_id=str(envelope["vms_instance_id"]),
        content_sha256=digest,
        request_method=str(envelope.get("request_method") or "GET"),
    )


class FrigateRecordedTransport:
    """Replay recorded responses through the same normalizer inputs as live HTTP."""

    def __init__(self, recorded: RecordedResponse) -> None:
        self.recorded = recorded

    def as_http_response(self) -> FrigateHttpResponse:
        return self.recorded.to_http_response()


def load_frigate_env(path: Path | None = None) -> dict[str, str]:
    """Load private Git-ignored Frigate env without printing values."""
    env = dict(os.environ)
    candidates = []
    if path is not None:
        candidates.append(path)
    else:
        # record.py → frigate → vms → nanexus_investigation → src → repo
        repo = Path(__file__).resolve().parents[4]
        candidates.extend(
            [
                repo / ".local" / "frigate.env",
                repo / ".env.local",
                repo / ".env",
            ]
        )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        for line in candidate.read_text().splitlines():
            if not line or line.strip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            # Private file values win over empty/missing process env for Frigate keys.
            if key.startswith("NANEXUS_FRIGATE_"):
                env[key] = value  # keep exact password bytes aside from line ending
            elif key not in env:
                env[key] = value.strip().strip('"').strip("'")
        break
    return env


def config_from_env(env: dict[str, str] | None = None) -> FrigateClientConfig:
    """Build client config from NANEXUS_FRIGATE_* environment variables."""
    if env is None:
        env = load_frigate_env()
    cameras = tuple(
        camera.strip()
        for camera in (env.get("NANEXUS_FRIGATE_CAMERAS") or "").split(",")
        if camera.strip()
    )
    return FrigateClientConfig(
        base_url=(env.get("NANEXUS_FRIGATE_BASE_URL") or "").rstrip("/"),
        vms_instance_id=env.get("NANEXUS_FRIGATE_VMS_INSTANCE_ID") or "frigate-fixture",
        username=env.get("NANEXUS_FRIGATE_USERNAME") or None,
        password=env.get("NANEXUS_FRIGATE_PASSWORD") or None,
        bearer_token=env.get("NANEXUS_FRIGATE_BEARER_TOKEN") or None,
        session_cookie=env.get("NANEXUS_FRIGATE_SESSION_COOKIE") or None,
        ca_bundle=env.get("NANEXUS_FRIGATE_CA_BUNDLE") or None,
        disable_hostname_check=(
            (env.get("NANEXUS_FRIGATE_DISABLE_HOSTNAME_CHECK") or "").lower()
            in {"1", "true", "yes"}
        ),
        authorized_cameras=cameras,
        max_limit=int(env.get("NANEXUS_FRIGATE_MAX_LIMIT") or "50"),
    )


def live_client_ready(env: dict[str, str] | None = None) -> tuple[bool, str]:
    """Return whether authenticated live Frigate access is configured."""
    if env is None:
        env = load_frigate_env()
    base = env.get("NANEXUS_FRIGATE_BASE_URL") or ""
    if not base:
        return False, "NANEXUS_FRIGATE_BASE_URL is unset"
    if ":5000" in base:
        return False, "port 5000 is not permitted; configure authenticated :8971"
    has_bearer = bool(env.get("NANEXUS_FRIGATE_BEARER_TOKEN"))
    has_password = bool(env.get("NANEXUS_FRIGATE_USERNAME") and env.get("NANEXUS_FRIGATE_PASSWORD"))
    has_cookie = bool(env.get("NANEXUS_FRIGATE_SESSION_COOKIE"))
    if not (has_bearer or has_password or has_cookie):
        return (
            False,
            "authenticated credentials missing: set NANEXUS_FRIGATE_BEARER_TOKEN, "
            "NANEXUS_FRIGATE_USERNAME/PASSWORD, or NANEXUS_FRIGATE_SESSION_COOKIE "
            "for https://<host>:8971 (do not commit secrets)",
        )
    if base.startswith("https://") and not env.get("NANEXUS_FRIGATE_CA_BUNDLE"):
        return (
            False,
            "NANEXUS_FRIGATE_CA_BUNDLE is required for HTTPS Frigate trust "
            "(do not disable TLS verification)",
        )
    ca = env.get("NANEXUS_FRIGATE_CA_BUNDLE")
    if ca and not Path(ca).is_file():
        return False, f"NANEXUS_FRIGATE_CA_BUNDLE file not found: {ca}"
    if not (env.get("NANEXUS_FRIGATE_CAMERAS") or "").strip():
        return False, "NANEXUS_FRIGATE_CAMERAS must list at least one authorized camera"
    return True, "ok"


def acquire_and_record(
    client: FrigateReadOnlyClient,
    *,
    output_dir: Path,
    cameras: tuple[str, ...],
    after: float,
    before: float,
    label: str | None,
    limit: int,
) -> dict[str, Any]:
    """Live historical acquisition helper. Writes private recorded artifacts."""
    version = client.get_version()
    listed = client.list_events(
        cameras=cameras,
        after=after,
        before=before,
        label=label,
        limit=limit,
    )
    listed.frigate_version = version
    meta = save_recorded_response(
        output_dir,
        "events_list",
        listed,
        vms_instance_id=client.config.vms_instance_id,
    )
    detail_meta = None
    events = json.loads(listed.body.decode("utf-8"))
    if isinstance(events, list) and events:
        event_id = events[0].get("id")
        if isinstance(event_id, str):
            detail = client.get_event(event_id)
            detail.frigate_version = version
            detail_meta = save_recorded_response(
                output_dir,
                "event_detail",
                detail,
                vms_instance_id=client.config.vms_instance_id,
            )
    return {
        "frigate_version": version,
        "events_count": len(events) if isinstance(events, list) else 0,
        "list_meta": str(meta),
        "detail_meta": None if detail_meta is None else str(detail_meta),
        "acquired_at": datetime.now(tz=timezone.utc).isoformat(),
    }
