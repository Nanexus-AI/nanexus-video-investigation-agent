"""Frigate-backed tool dispatch for guarded ``search_events`` execution.

Stub, recorded-response, and live API modes share the B1 normalizer. This module
is invoked only after Tool Guard acceptance — never from planner or report nodes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from nanexus_investigation.contracts.enums import (
    SearchCoverage,
    ToolErrorCode,
    ToolResultStatus,
)
from nanexus_investigation.contracts.identity import Interval
from nanexus_investigation.contracts.tool import ToolError, ToolRequest, ToolResult
from nanexus_investigation.tools.registry import (
    ALLOWED_TOOLS,
    ANALYSIS_TOOLS,
    MEDIA_TOOLS,
    VISUAL_TOOLS,
)
from nanexus_investigation.vms.frigate.client import (
    FrigateAuthError,
    FrigateHttpResponse,
    FrigateNotFoundError,
    FrigateProtocolError,
    FrigateReadOnlyClient,
    FrigateScopeError,
    FrigateUnavailableError,
)
from nanexus_investigation.vms.frigate.normalize import (
    normalize_error_tool_result,
    normalize_events_response,
)
from nanexus_investigation.vms.frigate.record import (
    FrigateRecordedTransport,
    load_recorded_response,
)

UTC = timezone.utc
BackendMode = Literal["recorded", "live"]


def _parse_time_bounds(request: ToolRequest) -> Interval:
    raw = request.arguments.get("time_bounds")
    if not isinstance(raw, dict) or "start_utc" not in raw:
        raise FrigateScopeError("search_events requires time_bounds")
    start = raw["start_utc"]
    end = raw.get("end_utc")
    if isinstance(start, str):
        start = datetime.fromisoformat(start)
    if isinstance(end, str):
        end = datetime.fromisoformat(end)
    if not isinstance(start, datetime):
        raise FrigateScopeError("time_bounds.start_utc must be datetime")
    if end is not None and not isinstance(end, datetime):
        raise FrigateScopeError("time_bounds.end_utc must be datetime or null")
    if start.tzinfo is None or (end is not None and end.tzinfo is None):
        raise FrigateScopeError("time_bounds must be timezone-aware UTC intervals")
    return Interval(start_utc=start, end_utc=end)


def _http_from_events_body(
    body: bytes,
    *,
    path: str = "/api/events",
    query: dict[str, str] | None = None,
    frigate_version: str | None = None,
    retrieved_at: datetime | None = None,
) -> FrigateHttpResponse:
    return FrigateHttpResponse(
        status_code=200,
        path=path,
        query=dict(query or {}),
        body=body,
        headers={"Content-Type": "application/json"},
        retrieved_at=retrieved_at or datetime.now(tz=UTC),
        frigate_version=frigate_version,
    )


def load_events_http_response(source: Path) -> FrigateHttpResponse:
    """Load either a recorded envelope (``.json`` + ``.body``) or raw events JSON."""
    path = Path(source)
    if path.suffix == ".json":
        try:
            envelope = json.loads(path.read_text())
        except json.JSONDecodeError:
            envelope = None
        if isinstance(envelope, dict) and "body_file" in envelope:
            return FrigateRecordedTransport(load_recorded_response(path)).as_http_response()
        return _http_from_events_body(
            path.read_bytes(),
            frigate_version="fixture-sanitized",
            retrieved_at=datetime(2026, 11, 20, 18, 0, tzinfo=UTC),
        )
    return _http_from_events_body(path.read_bytes())


def _filter_events_payload(
    body: bytes,
    *,
    cameras: tuple[str, ...],
    bounds: Interval,
    label: str | None,
) -> bytes:
    """Apply request filters to a recorded payload without inventing fields.

    Events are kept when ``start_time`` falls in the half-open request interval
    ``[start, end)``, camera is requested, and label matches when provided.
    Events missing ``end_time`` are retained (no invented end).
    """
    payload = json.loads(body.decode("utf-8"))
    if not isinstance(payload, list):
        raise FrigateProtocolError("Frigate events response must be a JSON array")
    start_ts = bounds.start_utc.timestamp()
    end_ts = None if bounds.end_utc is None else bounds.end_utc.timestamp()
    camera_set = set(cameras)
    filtered: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for item in payload:
        if not isinstance(item, dict):
            raise FrigateProtocolError("Frigate event entries must be objects")
        native_id = item.get("id")
        if not isinstance(native_id, str) or not native_id.strip():
            raise FrigateProtocolError("event missing native id")
        if native_id in seen_ids:
            continue
        camera = item.get("camera")
        if camera_set and camera not in camera_set:
            continue
        start_time = item.get("start_time")
        if not isinstance(start_time, (int, float)):
            raise FrigateProtocolError("event missing start_time")
        if float(start_time) < start_ts:
            continue
        if end_ts is not None and float(start_time) >= end_ts:
            continue
        if label is not None and item.get("label") != label:
            continue
        seen_ids.add(native_id)
        filtered.append(item)
    return json.dumps(filtered).encode("utf-8")


@dataclass
class FrigateToolRegistry:
    """Real Frigate adapter dispatch for allowlisted investigation tools.

    ``search_events`` uses recorded or live transport. ``search_related_events``
    is explicitly UNSUPPORTED — never silently delegated to stub scripts.
    Local ``analyze_temporal_relations`` runs against the bound Case store.
    """

    mode: BackendMode
    vms_instance_id: str
    max_limit: int = 50
    recorded_source: Path | None = None
    client: FrigateReadOnlyClient | None = None
    call_counts: dict[str, int] = field(default_factory=dict)
    invocations: list[str] = field(default_factory=list)
    # Test/injection hook: return this HTTP response instead of I/O.
    injected_response: FrigateHttpResponse | None = None
    # Test/injection hook: raise this before any network/filesystem access.
    inject_error: Exception | None = None
    analysis_store: Any | None = None
    analysis_scope: Any | None = None
    analysis_max_pairs: int = 100
    analysis_max_window_seconds: float = 3600.0
    media_store: Any | None = None
    media_case_id: str | None = None
    media_transport: Any | None = None
    max_snapshot_bytes: int = 5 * 1024 * 1024
    max_clip_bytes: int = 25 * 1024 * 1024
    # Optional recorded media directory: {native_id}.{snapshot|clip}.{jpg|mp4}
    recorded_media_dir: Path | None = None
    visual_provider: Any | None = None

    def __post_init__(self) -> None:
        if self.mode == "recorded" and self.recorded_source is None and self.injected_response is None:
            raise ValueError("frigate_recorded backend requires recorded_source")
        if self.mode == "live" and self.client is None and self.injected_response is None:
            raise ValueError("frigate_live backend requires an authenticated client")

    def bind_case_store(self, store: Any, scope: Any) -> None:
        self.analysis_store = store
        self.analysis_scope = scope

    def bind_media_store(self, media_store: Any, *, case_id: str) -> None:
        self.media_store = media_store
        self.media_case_id = case_id

    def bind_visual_provider(self, visual_provider: Any | None) -> None:
        self.visual_provider = visual_provider

    def is_registered(self, tool_name: str) -> bool:
        return tool_name in ALLOWED_TOOLS

    def invoke(self, request: ToolRequest) -> ToolResult:
        name = request.tool_name
        if name not in ALLOWED_TOOLS:
            raise RuntimeError(f"unregistered tool body invoked: {name}")
        self.invocations.append(name)
        index = self.call_counts.get(name, 0)
        self.call_counts[name] = index + 1

        if name in ANALYSIS_TOOLS:
            return self._invoke_temporal(request)

        if name in MEDIA_TOOLS:
            return self._invoke_resolve_media(request)

        if name in VISUAL_TOOLS:
            return self._invoke_observe_media(request)

        if name == "search_related_events":
            return normalize_error_tool_result(
                status=ToolResultStatus.UNSUPPORTED,
                error=ToolError(
                    code=ToolErrorCode.DISABLED_CAPABILITY,
                    message=(
                        "search_related_events is not implemented for the Frigate "
                        "read-only adapter; refusing synthetic stub follow-up"
                    ),
                ),
                vms_instance_id=self.vms_instance_id,
            )

        if name != "search_events":
            return normalize_error_tool_result(
                status=ToolResultStatus.UNSUPPORTED,
                error=ToolError(
                    code=ToolErrorCode.DISABLED_CAPABILITY,
                    message=f"Frigate adapter does not implement {name}",
                ),
                vms_instance_id=self.vms_instance_id,
            )

        return self._invoke_search_events(request)

    def _invoke_temporal(self, request: ToolRequest) -> ToolResult:
        from nanexus_investigation.tools.temporal_analysis import (
            analyze_temporal_relations,
        )

        if self.analysis_store is None or self.analysis_scope is None:
            return normalize_error_tool_result(
                status=ToolResultStatus.ERROR,
                error=ToolError(
                    code=ToolErrorCode.DISABLED_CAPABILITY,
                    message="temporal analysis store is not bound for this investigation",
                ),
                vms_instance_id=self.vms_instance_id,
            )
        return analyze_temporal_relations(
            request,
            observations=self.analysis_store.observations,
            scope=self.analysis_scope,
            max_pairs=self.analysis_max_pairs,
            max_window_seconds=self.analysis_max_window_seconds,
        )

    def _media_transport(self) -> Any:
        if self.media_transport is not None:
            return self.media_transport
        if self.mode == "live":
            return self.client
        if self.mode == "recorded":
            return _RecordedMediaTransport(self.recorded_media_dir)
        return None

    def _invoke_resolve_media(self, request: ToolRequest) -> ToolResult:
        from nanexus_investigation.tools.media_resolve import resolve_media

        if self.analysis_store is None or self.analysis_scope is None:
            return normalize_error_tool_result(
                status=ToolResultStatus.ERROR,
                error=ToolError(
                    code=ToolErrorCode.DISABLED_CAPABILITY,
                    message="media resolve store is not bound for this investigation",
                ),
                vms_instance_id=self.vms_instance_id,
            )
        if self.media_store is None or not self.media_case_id:
            return normalize_error_tool_result(
                status=ToolResultStatus.ERROR,
                error=ToolError(
                    code=ToolErrorCode.DISABLED_CAPABILITY,
                    message="media retention store is not bound for this investigation",
                ),
                vms_instance_id=self.vms_instance_id,
            )
        return resolve_media(
            request,
            evidence_by_id=self.analysis_store.evidence,
            observations_by_id=self.analysis_store.observations,
            scope=self.analysis_scope,
            case_id=self.media_case_id,
            media_store=self.media_store,
            transport=self._media_transport(),
            max_snapshot_bytes=self.max_snapshot_bytes,
            max_clip_bytes=self.max_clip_bytes,
        )

    def _invoke_observe_media(self, request: ToolRequest) -> ToolResult:
        from nanexus_investigation.tools.media_observe import observe_media

        if self.analysis_store is None or self.analysis_scope is None:
            return normalize_error_tool_result(
                status=ToolResultStatus.ERROR,
                error=ToolError(
                    code=ToolErrorCode.DISABLED_CAPABILITY,
                    message="observe_media store is not bound for this investigation",
                ),
                vms_instance_id=self.vms_instance_id,
            )
        if self.media_store is None or not self.media_case_id:
            return normalize_error_tool_result(
                status=ToolResultStatus.ERROR,
                error=ToolError(
                    code=ToolErrorCode.DISABLED_CAPABILITY,
                    message="media retention store is not bound for this investigation",
                ),
                vms_instance_id=self.vms_instance_id,
            )
        return observe_media(
            request,
            evidence_by_id=self.analysis_store.evidence,
            scope=self.analysis_scope,
            case_id=self.media_case_id,
            media_store=self.media_store,
            visual_provider=self.visual_provider,
        )

    def _invoke_search_events(self, request: ToolRequest) -> ToolResult:
        source_id = request.source_id or self.vms_instance_id
        if source_id != self.vms_instance_id:
            return normalize_error_tool_result(
                status=ToolResultStatus.ERROR,
                error=ToolError(
                    code=ToolErrorCode.AUTHORIZATION_DENIED,
                    message="tool source_id does not match Frigate VMS instance",
                ),
                vms_instance_id=self.vms_instance_id,
            )

        try:
            bounds = _parse_time_bounds(request)
            cameras = request.camera_ids
            if not cameras:
                raise FrigateScopeError("at least one camera must be specified")
            label = request.arguments.get("label")
            if label is not None and not isinstance(label, str):
                raise FrigateScopeError("label must be a string when provided")
            limit_raw = request.arguments.get("limit", self.max_limit)
            limit = int(limit_raw)
            if limit < 1 or limit > self.max_limit:
                raise FrigateScopeError(
                    f"limit must be between 1 and {self.max_limit}"
                )

            if self.inject_error is not None:
                raise self.inject_error

            response = self._fetch_response(
                cameras=cameras,
                bounds=bounds,
                label=label if isinstance(label, str) else None,
                limit=limit,
            )
            if self.mode == "recorded" or self.injected_response is not None:
                # Recorded/injected bodies may be broader than the approved request;
                # filter to the Guard-approved cameras/time/label before normalize.
                filtered = _filter_events_payload(
                    response.body,
                    cameras=cameras,
                    bounds=bounds,
                    label=label if isinstance(label, str) else None,
                )
                # Respect page limit without inventing additional rows.
                payload = json.loads(filtered.decode("utf-8"))
                if isinstance(payload, list) and len(payload) > limit:
                    payload = payload[:limit]
                    filtered = json.dumps(payload).encode("utf-8")
                response = FrigateHttpResponse(
                    status_code=response.status_code,
                    path=response.path,
                    query={
                        "cameras": ",".join(cameras),
                        "limit": str(limit),
                        "after": str(bounds.start_utc.timestamp()),
                        **(
                            {"before": str(bounds.end_utc.timestamp())}
                            if bounds.end_utc is not None
                            else {}
                        ),
                        **({"labels": label} if isinstance(label, str) else {}),
                    },
                    body=filtered,
                    headers=response.headers,
                    retrieved_at=response.retrieved_at,
                    frigate_version=response.frigate_version,
                )

            normalized = normalize_events_response(
                response,
                vms_instance_id=self.vms_instance_id,
                cameras=cameras,
                time_bounds=bounds,
                label=label if isinstance(label, str) else None,
                page_limit=limit,
                tool_name="search_events",
            )
            result = normalized.tool_result
            usage = dict(result.usage or {})
            usage["page_limit"] = limit
            # Ensure registration sees durable source revision markers.
            data = result.data
            if isinstance(data, list):
                revision = f"frigate-read:{response.retrieved_at.isoformat()}"
                data = [
                    {
                        **row,
                        "source_revision": row.get("source_revision") or revision,
                    }
                    if isinstance(row, dict)
                    else row
                    for row in data
                ]
            # Preserve Frigate version on receipt via register_tool_artifacts
            # (ToolResult.usage is numeric-only under the canonical contract).
            return result.model_copy(update={"usage": usage, "data": data})
        except FrigateScopeError as exc:
            return normalize_error_tool_result(
                status=ToolResultStatus.ERROR,
                error=ToolError(
                    code=ToolErrorCode.INVALID_SCOPE,
                    message=str(exc),
                ),
                vms_instance_id=self.vms_instance_id,
            )
        except FrigateAuthError as exc:
            return normalize_error_tool_result(
                status=ToolResultStatus.ERROR,
                error=ToolError(
                    code=ToolErrorCode.AUTHORIZATION_DENIED,
                    message=str(exc),
                ),
                vms_instance_id=self.vms_instance_id,
            )
        except FrigateUnavailableError as exc:
            return normalize_error_tool_result(
                status=ToolResultStatus.ERROR,
                error=ToolError(
                    code=ToolErrorCode.SOURCE_UNAVAILABLE,
                    message=str(exc),
                ),
                vms_instance_id=self.vms_instance_id,
            )
        except FrigateNotFoundError as exc:
            return normalize_error_tool_result(
                status=ToolResultStatus.ERROR,
                error=ToolError(
                    code=ToolErrorCode.INVALID_RESPONSE,
                    message=str(exc),
                ),
                vms_instance_id=self.vms_instance_id,
            )
        except FrigateProtocolError as exc:
            return normalize_error_tool_result(
                status=ToolResultStatus.ERROR,
                error=ToolError(
                    code=ToolErrorCode.INVALID_RESPONSE,
                    message=str(exc),
                ),
                vms_instance_id=self.vms_instance_id,
            )
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError, ValueError) as exc:
            return normalize_error_tool_result(
                status=ToolResultStatus.ERROR,
                error=ToolError(
                    code=ToolErrorCode.INVALID_RESPONSE,
                    message=f"malformed Frigate payload: {exc}",
                ),
                vms_instance_id=self.vms_instance_id,
            )

    def _fetch_response(
        self,
        *,
        cameras: tuple[str, ...],
        bounds: Interval,
        label: str | None,
        limit: int,
    ) -> FrigateHttpResponse:
        if self.injected_response is not None:
            return self.injected_response
        if self.mode == "recorded":
            assert self.recorded_source is not None
            return load_events_http_response(self.recorded_source)
        assert self.client is not None
        # Live path: client enforces authorized cameras / limit before network I/O.
        return self.client.list_events(
            cameras=cameras,
            after=bounds.start_utc.timestamp(),
            before=None if bounds.end_utc is None else bounds.end_utc.timestamp(),
            label=label,
            limit=limit,
        )


@dataclass
class _RecordedMediaTransport:
    """Read previously captured media bytes from a private local directory.

    Files are named ``{native_id}.snapshot.jpg`` or ``{native_id}.clip.mp4``.
    Missing files surface as FrigateNotFoundError (unavailable media).
    """

    root: Path | None

    def _load(self, event_id: str, suffix: str, content_type: str) -> FrigateHttpResponse:
        if self.root is None:
            raise FrigateNotFoundError(f"Frigate media not found ({suffix})")
        if (
            not event_id
            or event_id.strip() != event_id
            or "/" in event_id
            or "\\" in event_id
            or ".." in event_id
            or "?" in event_id
        ):
            from nanexus_investigation.vms.frigate.client import FrigateScopeError

            raise FrigateScopeError("event_id contains forbidden characters")
        root = self.root.resolve()
        path = (root / f"{event_id}.{suffix}").resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            from nanexus_investigation.vms.frigate.client import FrigateScopeError

            raise FrigateScopeError(
                "recorded media path escapes private media root"
            ) from exc
        if not path.is_file():
            raise FrigateNotFoundError(f"Frigate media not found ({suffix})")
        body = path.read_bytes()
        if not body:
            from nanexus_investigation.vms.frigate.client import FrigateMediaError

            raise FrigateMediaError(f"empty media body for {suffix}")
        return FrigateHttpResponse(
            status_code=200,
            path=f"/api/events/{event_id}/{suffix}",
            query={},
            body=body,
            headers={"Content-Type": content_type},
            retrieved_at=datetime.now(tz=UTC),
            frigate_version="recorded",
        )

    def get_event_snapshot(self, event_id: str, *, max_bytes: int = 0) -> FrigateHttpResponse:
        response = self._load(event_id, "snapshot.jpg", "image/jpeg")
        if max_bytes > 0 and len(response.body) > max_bytes:
            from nanexus_investigation.vms.frigate.client import FrigateMediaTooLargeError

            raise FrigateMediaTooLargeError(
                f"media response exceeds max_bytes={max_bytes}"
            )
        return response

    def get_event_clip(self, event_id: str, *, max_bytes: int = 0) -> FrigateHttpResponse:
        response = self._load(event_id, "clip.mp4", "video/mp4")
        if max_bytes > 0 and len(response.body) > max_bytes:
            from nanexus_investigation.vms.frigate.client import FrigateMediaTooLargeError

            raise FrigateMediaTooLargeError(
                f"media response exceeds max_bytes={max_bytes}"
            )
        return response


def unsupported_related_events_result(*, vms_instance_id: str) -> ToolResult:
    """Explicit UNSUPPORTED ToolResult for tests (no stub fallthrough)."""
    return normalize_error_tool_result(
        status=ToolResultStatus.UNSUPPORTED,
        error=ToolError(
            code=ToolErrorCode.DISABLED_CAPABILITY,
            message="search_related_events is not implemented for Frigate",
        ),
        vms_instance_id=vms_instance_id,
    )


def empty_search_result(*, vms_instance_id: str, page_limit: int = 50) -> ToolResult:
    return ToolResult(
        status=ToolResultStatus.EMPTY,
        execution_id=f"exec-frigate-empty-{uuid4().hex[:8]}",
        data=[],
        source_id=vms_instance_id,
        retrieved_at=datetime.now(tz=UTC),
        coverage=SearchCoverage.EXHAUSTED,
        provenance="adapter:frigate-read/search_events",
        usage={"tool_attempts": 1, "page_limit": page_limit},
    )
