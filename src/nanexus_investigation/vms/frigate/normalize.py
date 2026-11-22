"""Shared Frigate response → Nanexus contract normalization.

Live HTTP responses and recorded responses use this same path. Missing fields
are not invented. Unrestricted raw payloads are not placed into Graph state.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from nanexus_investigation.contracts.enums import (
    EvidenceAvailability,
    EvidenceKind,
    ObservationType,
    RecordType,
    SearchCoverage,
    ToolResultStatus,
)
from nanexus_investigation.contracts.evidence import EvidenceRef, Observation
from nanexus_investigation.contracts.identity import Interval, SourceLocator
from nanexus_investigation.contracts.search import SearchExecutionReceipt
from nanexus_investigation.contracts.tool import ToolError, ToolResult
from nanexus_investigation.vms.frigate.client import (
    FrigateHttpResponse,
    FrigateProtocolError,
)


def _unix_to_utc(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    raise FrigateProtocolError("timestamp fields must be numeric Unix seconds")


def _parse_events_payload(body: bytes) -> list[dict[str, Any]]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrigateProtocolError(f"malformed Frigate JSON: {exc}") from exc
    if not isinstance(payload, list):
        raise FrigateProtocolError("Frigate events response must be a JSON array")
    events: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise FrigateProtocolError("Frigate event entries must be objects")
        events.append(item)
    return events


def _parse_event_object(body: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise FrigateProtocolError(f"malformed Frigate JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise FrigateProtocolError("Frigate event detail must be a JSON object")
    return payload


def _event_to_observation_evidence(
    event: dict[str, Any],
    *,
    vms_instance_id: str,
    retrieved_at: datetime,
    provenance: str,
) -> tuple[Observation, EvidenceRef]:
    native_id = event.get("id")
    camera = event.get("camera")
    label = event.get("label")
    start = _unix_to_utc(event.get("start_time"))
    end = _unix_to_utc(event.get("end_time"))
    if not isinstance(native_id, str) or not native_id.strip():
        raise FrigateProtocolError("event missing native id")
    if not isinstance(camera, str) or not camera.strip():
        raise FrigateProtocolError("event missing camera")
    if start is None:
        raise FrigateProtocolError("event missing start_time")
    # Incomplete events may omit end_time; do not invent an end.
    if end is not None and end <= start:
        raise FrigateProtocolError("event has inverted time interval")

    zones = event.get("zones") or ()
    if isinstance(zones, list):
        zone_tuple = tuple(str(z) for z in zones if isinstance(z, str) and z.strip())
    else:
        zone_tuple = ()

    labels = (str(label),) if isinstance(label, str) and label.strip() else ()
    # Frigate list/detail payloads do not expose a durable content revision.
    # Record retrieval provenance instead of inventing source revision bytes.
    source_revision = f"frigate-read:{retrieved_at.isoformat()}"

    locator = SourceLocator(
        vms_instance_id=vms_instance_id,
        record_type=RecordType.EVENT.value,
        native_record_id=native_id,
    )
    observation = Observation(
        observation_id=f"obs-{native_id}",
        source_locator=locator,
        source_revision=source_revision,
        observation_type=ObservationType.DETECTION_EVENT,
        camera_id=camera,
        start_utc=start,
        end_utc=end,
        labels=labels,
        zones=zone_tuple,
        attributes={
            "has_clip": bool(event.get("has_clip")) if "has_clip" in event else None,
            "has_snapshot": bool(event.get("has_snapshot"))
            if "has_snapshot" in event
            else None,
            "score": event.get("score"),
        },
        retrieved_at=retrieved_at,
        provenance=provenance,
    )
    # Drop None attribute values to keep payloads tight.
    observation = observation.model_copy(
        update={
            "attributes": {
                key: value
                for key, value in observation.attributes.items()
                if value is not None
            }
        }
    )
    evidence = EvidenceRef(
        evidence_id=f"ev-{native_id}",
        source_locator=locator,
        evidence_kind=EvidenceKind.EVENT_METADATA,
        camera_id=camera,
        interval=Interval(start_utc=start, end_utc=end),
        retrieval_spec={
            "operation": "get_event" if event.get("_detail") else "search_events",
            "native_id": native_id,
            "has_clip": bool(event["has_clip"]) if "has_clip" in event else None,
            "has_snapshot": bool(event["has_snapshot"])
            if "has_snapshot" in event
            else None,
        },
        availability=EvidenceAvailability.AVAILABLE,
        checked_at=retrieved_at,
        provenance=provenance,
    )
    evidence = evidence.model_copy(
        update={
            "retrieval_spec": {
                key: value
                for key, value in evidence.retrieval_spec.items()
                if value is not None
            }
        }
    )
    return observation, evidence


@dataclass(frozen=True, slots=True)
class NormalizedEventSearch:
    """Canonical Nanexus artifacts produced from one Frigate events response."""

    tool_result: ToolResult
    receipt: SearchExecutionReceipt
    observations: tuple[Observation, ...]
    evidence: tuple[EvidenceRef, ...]
    native_events: tuple[dict[str, Any], ...]


def normalize_events_response(
    response: FrigateHttpResponse,
    *,
    vms_instance_id: str,
    cameras: tuple[str, ...],
    time_bounds: Interval,
    label: str | None = None,
    page_limit: int | None = None,
    tool_name: str = "search_events",
) -> NormalizedEventSearch:
    """Normalize a list-events response into ToolResult + receipt + observations."""
    provenance = f"adapter:frigate-read/{tool_name}"
    events = _parse_events_payload(response.body)
    observations: list[Observation] = []
    evidence_refs: list[EvidenceRef] = []
    for event in events:
        observation, evidence = _event_to_observation_evidence(
            event,
            vms_instance_id=vms_instance_id,
            retrieved_at=response.retrieved_at,
            provenance=provenance,
        )
        observations.append(observation)
        evidence_refs.append(evidence)

    match_count = len(events)
    if match_count == 0:
        status = ToolResultStatus.EMPTY
        # Exhausted only when the query completed with an explicit limit and
        # returned fewer than that limit (including zero). Do not claim
        # exhausted when truncated at the requested page size.
        if page_limit is not None and match_count < page_limit:
            coverage = SearchCoverage.EXHAUSTED
        else:
            coverage = SearchCoverage.UNKNOWN
    elif page_limit is not None and match_count >= page_limit:
        status = ToolResultStatus.OK
        coverage = SearchCoverage.TRUNCATED
    else:
        status = ToolResultStatus.OK
        coverage = SearchCoverage.EXHAUSTED

    execution_id = f"exec-frigate-{uuid4().hex[:12]}"
    tool_result = ToolResult(
        status=status,
        execution_id=execution_id,
        data=[
            {
                "native_record_id": event["id"],
                "record_type": RecordType.EVENT.value,
                "camera_id": event.get("camera"),
                "label": event.get("label"),
                "start_utc": _unix_to_utc(event.get("start_time")),
                "end_utc": _unix_to_utc(event.get("end_time")),
                "has_clip": event.get("has_clip"),
                "has_snapshot": event.get("has_snapshot"),
                "source_revision": f"frigate-read:{response.retrieved_at.isoformat()}",
            }
            for event in events
        ],
        source_id=vms_instance_id,
        retrieved_at=response.retrieved_at,
        coverage=coverage,
        provenance=provenance,
        usage={
            "tool_attempts": 1,
            "http_status": response.status_code,
            **({"page_limit": page_limit} if page_limit is not None else {}),
        },
    )
    receipt = SearchExecutionReceipt(
        search_execution_id=f"receipt-{execution_id}",
        tool_name=tool_name,
        source_id=vms_instance_id,
        camera_ids=cameras,
        time_bounds=time_bounds,
        filters={"label": label, "path": response.path, **response.query},
        retrieved_at=response.retrieved_at,
        pages_fetched=1,
        page_limit=page_limit,
        match_count=match_count,
        coverage=coverage,
        source_capability=tool_name,
        consistency_limitations=(
            None
            if response.frigate_version is None
            else f"frigate_version={response.frigate_version}"
        ),
    )
    return NormalizedEventSearch(
        tool_result=tool_result,
        receipt=receipt,
        observations=tuple(observations),
        evidence=tuple(evidence_refs),
        native_events=tuple(events),
    )


def normalize_event_detail_response(
    response: FrigateHttpResponse,
    *,
    vms_instance_id: str,
) -> tuple[Observation, EvidenceRef, ToolResult]:
    """Normalize GET /api/events/{id} into observation/evidence/tool result."""
    provenance = "adapter:frigate-read/get_event"
    event = _parse_event_object(response.body)
    event = {**event, "_detail": True}
    observation, evidence = _event_to_observation_evidence(
        event,
        vms_instance_id=vms_instance_id,
        retrieved_at=response.retrieved_at,
        provenance=provenance,
    )
    tool_result = ToolResult(
        status=ToolResultStatus.OK,
        execution_id=f"exec-frigate-detail-{uuid4().hex[:12]}",
        data={
            "native_record_id": event["id"],
            "record_type": RecordType.EVENT.value,
            "camera_id": event.get("camera"),
            "label": event.get("label"),
            "start_utc": observation.start_utc,
            "end_utc": observation.end_utc,
            "has_clip": event.get("has_clip"),
            "has_snapshot": event.get("has_snapshot"),
        },
        source_id=vms_instance_id,
        retrieved_at=response.retrieved_at,
        coverage=None,
        provenance=provenance,
        usage={"tool_attempts": 1, "http_status": response.status_code},
    )
    return observation, evidence, tool_result


def normalize_error_tool_result(
    *,
    status: ToolResultStatus,
    error: ToolError,
    vms_instance_id: str,
    retrieved_at: datetime | None = None,
) -> ToolResult:
    return ToolResult(
        status=status,
        execution_id=f"exec-frigate-error-{uuid4().hex[:12]}",
        data=None,
        error=error,
        source_id=vms_instance_id,
        retrieved_at=retrieved_at or datetime.now(tz=timezone.utc),
        coverage=None,
        provenance="adapter:frigate-read/error",
        usage={"tool_attempts": 1},
    )
