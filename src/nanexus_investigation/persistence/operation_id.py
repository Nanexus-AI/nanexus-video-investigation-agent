"""Stable logical operation and attempt identities. G3-C1 §8.

Encoding is an implementation choice, not a new architectural contract.
Identities must survive process restart and remain stable across replay of the
same logical operation while distinguishing deliberate new operations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _digest(parts: Mapping[str, Any]) -> str:
    payload = _canonical_json(dict(parts))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class OperationIdentity:
    """Stable identity for one logical evidence-producing operation."""

    operation_id: str
    case_id: str
    plan_step_id: str
    tool_name: str
    content_fingerprint: str

    def as_key(self) -> str:
        return self.operation_id


@dataclass(frozen=True, slots=True)
class AttemptIdentity:
    """Physical attempt identity associated with a parent operation."""

    attempt_id: str
    operation_id: str
    attempt_index: int


def build_operation_identity(
    *,
    case_id: str,
    plan_step_id: str,
    tool_name: str,
    vms_instance_id: str,
    request_parameters: Mapping[str, Any],
) -> OperationIdentity:
    """Derive a replay-stable operation identity from Case/plan/request fields."""
    fingerprint = _digest(
        {
            "case_id": case_id,
            "plan_step_id": plan_step_id,
            "tool_name": tool_name,
            "vms_instance_id": vms_instance_id,
            "request_parameters": dict(request_parameters),
        }
    )
    operation_id = f"op:{case_id}:{plan_step_id}:{fingerprint[:24]}"
    return OperationIdentity(
        operation_id=operation_id,
        case_id=case_id,
        plan_step_id=plan_step_id,
        tool_name=tool_name,
        content_fingerprint=fingerprint,
    )


def build_attempt_identity(
    *,
    operation_id: str,
    attempt_index: int,
) -> AttemptIdentity:
    if attempt_index < 1:
        raise ValueError("attempt_index must be >= 1")
    attempt_id = f"att:{operation_id}:{attempt_index}"
    return AttemptIdentity(
        attempt_id=attempt_id,
        operation_id=operation_id,
        attempt_index=attempt_index,
    )


def payload_fingerprint(payload: Mapping[str, Any] | Any) -> str:
    """Content fingerprint for idempotent domain writes."""
    if hasattr(payload, "model_dump"):
        data = payload.model_dump(mode="json")
    elif isinstance(payload, Mapping):
        data = dict(payload)
    else:
        data = {"value": payload}
    return _digest(data)
