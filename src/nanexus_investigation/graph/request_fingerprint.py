"""In-investigation tool-request equivalence for anti-repeat checks.

Not a persistent deduplication service. Fingerprints are scoped to one
InvestigationRuntime and compared before re-executing model-proposed tools.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Mapping

from nanexus_investigation.contracts.identity import Interval


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


def normalize_time_bounds(time_bounds: Interval | Mapping[str, Any] | None) -> dict[str, str | None] | None:
    if time_bounds is None:
        return None
    if isinstance(time_bounds, Interval):
        return {
            "start_utc": time_bounds.start_utc.isoformat(),
            "end_utc": None
            if time_bounds.end_utc is None
            else time_bounds.end_utc.isoformat(),
        }
    start = time_bounds.get("start_utc")
    end = time_bounds.get("end_utc")
    return {"start_utc": _iso(start), "end_utc": _iso(end)}


def tool_request_fingerprint(
    *,
    tool_name: str,
    source_id: str,
    camera_ids: tuple[str, ...] | list[str] | None,
    label: str | None,
    time_bounds: Interval | Mapping[str, Any] | None,
    extra_arguments: Mapping[str, Any] | None = None,
) -> str:
    """Stable fingerprint of an equivalent Guard-bound tool request."""
    cameras = tuple(sorted(str(c) for c in (camera_ids or ())))
    args: dict[str, Any] = {}
    if extra_arguments:
        for key, value in extra_arguments.items():
            if key in {"time_bounds", "label", "camera_ids"}:
                continue
            if value is not None:
                args[key] = value
    # Temporal analysis is keyed by structured arguments; decorative labels must
    # not create a second equivalent analyze_temporal_relations execution.
    effective_label = None if tool_name == "analyze_temporal_relations" else label
    payload = {
        "tool_name": tool_name,
        "source_id": source_id,
        "camera_ids": list(cameras),
        "label": effective_label,
        "time_bounds": normalize_time_bounds(time_bounds),
        "arguments": args,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
