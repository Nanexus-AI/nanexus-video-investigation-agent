"""Deterministic local temporal analysis tool (not a VMS source adapter).

``analyze_temporal_relations`` evaluates approved predicates over already
registered Case observations. It never invents timestamps, fetches media, or
fabricates Frigate events.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from nanexus_investigation.contracts.enums import (
    SearchCoverage,
    TemporalEvaluationStatus,
    TemporalRelationType,
    ToolErrorCode,
    ToolResultStatus,
)
from nanexus_investigation.contracts.evidence import Observation
from nanexus_investigation.contracts.scope import InvestigationScope
from nanexus_investigation.contracts.tool import ToolError, ToolRequest, ToolResult
from nanexus_investigation.evidence.claims import observation_interval
from nanexus_investigation.evidence.relations import evaluate_predicate

UTC = timezone.utc
TEMPORAL_TOOL = "analyze_temporal_relations"
DEFAULT_MAX_PAIRS = 100
DEFAULT_MAX_WINDOW_SECONDS = 3600.0
ALLOWED_PREDICATES = frozenset({TemporalRelationType.NEAR.value})


class TemporalAnalysisError(ValueError):
    """Invalid temporal analysis request (authorization or argument failure)."""


def _parse_predicate(raw: Any) -> TemporalRelationType:
    if raw is None:
        return TemporalRelationType.NEAR
    if not isinstance(raw, str):
        raise TemporalAnalysisError("predicate must be a string")
    value = raw.strip().lower()
    if value not in ALLOWED_PREDICATES:
        raise TemporalAnalysisError(
            f"unsupported temporal predicate: {raw!r}; "
            f"allowed={sorted(ALLOWED_PREDICATES)}"
        )
    return TemporalRelationType(value)


def _parse_window_seconds(
    raw: Any,
    *,
    max_window_seconds: float,
) -> float:
    if raw is None:
        raise TemporalAnalysisError("window_seconds is required for NEAR analysis")
    try:
        window = float(raw)
    except (TypeError, ValueError) as exc:
        raise TemporalAnalysisError("window_seconds must be a number") from exc
    if window < 0:
        raise TemporalAnalysisError("window_seconds must be nonnegative")
    if window > max_window_seconds:
        raise TemporalAnalysisError(
            f"window_seconds {window} exceeds max_window_seconds={max_window_seconds}"
        )
    return window


def _parse_id_list(raw: Any, *, field: str) -> tuple[str, ...] | None:
    if raw is None:
        return None
    if not isinstance(raw, (list, tuple)):
        raise TemporalAnalysisError(f"{field} must be a list of strings when provided")
    ids: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip():
            raise TemporalAnalysisError(f"{field} entries must be non-empty strings")
        ids.append(item.strip())
    return tuple(ids)


def _obs_in_scope(observation: Observation, scope: InvestigationScope) -> bool:
    if observation.camera_id not in scope.camera_ids:
        return False
    if observation.source_locator.vms_instance_id != scope.vms_instance_id:
        return False
    # Observation start must fall within the authorized half-open envelope.
    bounds = scope.time_bounds
    start = observation.start_utc
    if start < bounds.start_utc:
        return False
    if bounds.end_utc is not None and start >= bounds.end_utc:
        return False
    return True


def _select_observations(
    *,
    store_observations: dict[str, Observation],
    scope: InvestigationScope,
    explicit_ids: tuple[str, ...] | None,
    label: str | None,
    role: str,
) -> list[Observation]:
    if explicit_ids is not None:
        selected: list[Observation] = []
        for obs_id in explicit_ids:
            observation = store_observations.get(obs_id)
            if observation is None:
                raise TemporalAnalysisError(
                    f"unknown or foreign {role} observation_id: {obs_id}"
                )
            if not _obs_in_scope(observation, scope):
                raise TemporalAnalysisError(
                    f"{role} observation out of authorized scope: {obs_id}"
                )
            selected.append(observation)
        return selected

    selected = []
    for observation in store_observations.values():
        if not _obs_in_scope(observation, scope):
            continue
        if label is not None:
            if label not in observation.labels:
                continue
        selected.append(observation)
    return selected


def _camera_relation(left: Observation, right: Observation) -> str:
    if left.camera_id == right.camera_id:
        return "same_camera"
    return "cross_camera_authorized_scope"


def analyze_temporal_relations(
    request: ToolRequest,
    *,
    observations: dict[str, Observation],
    scope: InvestigationScope,
    max_pairs: int = DEFAULT_MAX_PAIRS,
    max_window_seconds: float = DEFAULT_MAX_WINDOW_SECONDS,
) -> ToolResult:
    """Run bounded deterministic temporal analysis over registered observations."""
    execution_id = f"exec-temporal-{uuid4().hex[:12]}"
    args = dict(request.arguments or {})

    try:
        predicate = _parse_predicate(args.get("predicate"))
        window_seconds = _parse_window_seconds(
            args.get("window_seconds"),
            max_window_seconds=max_window_seconds,
        )
        requested_max = args.get("max_pairs", max_pairs)
        try:
            pair_limit = int(requested_max)
        except (TypeError, ValueError) as exc:
            raise TemporalAnalysisError("max_pairs must be an integer") from exc
        if pair_limit < 1:
            raise TemporalAnalysisError("max_pairs must be >= 1")
        if pair_limit > max_pairs:
            raise TemporalAnalysisError(
                f"max_pairs {pair_limit} exceeds configured maximum {max_pairs}"
            )

        anchor_ids = _parse_id_list(
            args.get("anchor_observation_ids"), field="anchor_observation_ids"
        )
        candidate_ids = _parse_id_list(
            args.get("candidate_observation_ids"),
            field="candidate_observation_ids",
        )
        anchor_label = args.get("anchor_label")
        candidate_label = args.get("candidate_label")
        if anchor_label is not None and not isinstance(anchor_label, str):
            raise TemporalAnalysisError("anchor_label must be a string when provided")
        if candidate_label is not None and not isinstance(candidate_label, str):
            raise TemporalAnalysisError("candidate_label must be a string when provided")
        if isinstance(anchor_label, str):
            anchor_label = anchor_label.strip() or None
        if isinstance(candidate_label, str):
            candidate_label = candidate_label.strip() or None

        anchors = _select_observations(
            store_observations=observations,
            scope=scope,
            explicit_ids=anchor_ids,
            label=anchor_label,
            role="anchor",
        )
        candidates = _select_observations(
            store_observations=observations,
            scope=scope,
            explicit_ids=candidate_ids,
            label=candidate_label,
            role="candidate",
        )
    except TemporalAnalysisError as exc:
        return ToolResult(
            status=ToolResultStatus.ERROR,
            execution_id=execution_id,
            error=ToolError(
                code=ToolErrorCode.INVALID_SCOPE,
                message=str(exc),
            ),
            source_id=request.source_id or scope.vms_instance_id,
            retrieved_at=datetime.now(tz=UTC),
            coverage=SearchCoverage.UNKNOWN,
            provenance="temporal-engine:analyze_temporal_relations",
            usage={"tool_attempts": 1, "pairs_evaluated": 0},
        )

    if not anchors or not candidates:
        return ToolResult(
            status=ToolResultStatus.EMPTY,
            execution_id=execution_id,
            data=[],
            source_id=request.source_id or scope.vms_instance_id,
            retrieved_at=datetime.now(tz=UTC),
            coverage=SearchCoverage.EXHAUSTED,
            provenance="temporal-engine:analyze_temporal_relations",
            usage={
                "tool_attempts": 1,
                "pairs_evaluated": 0,
                "anchor_count": len(anchors),
                "candidate_count": len(candidates),
            },
        )

    established: list[dict[str, Any]] = []
    indeterminate: list[dict[str, Any]] = []
    pairs_evaluated = 0
    truncated = False

    for left in anchors:
        for right in candidates:
            if left.observation_id == right.observation_id:
                continue
            if pairs_evaluated >= pair_limit:
                truncated = True
                break
            pairs_evaluated += 1
            left_interval = observation_interval(left)
            right_interval = observation_interval(right)
            result = evaluate_predicate(
                left_interval,
                right_interval,
                predicate,
                window_seconds=window_seconds,
            )
            payload = {
                "record_kind": "temporal_relation",
                "left_observation_id": left.observation_id,
                "right_observation_id": right.observation_id,
                "left_native_record_id": left.source_locator.native_record_id,
                "right_native_record_id": right.source_locator.native_record_id,
                "left_camera_id": left.camera_id,
                "right_camera_id": right.camera_id,
                "left_start_utc": left.start_utc.isoformat(),
                "left_end_utc": None
                if left.end_utc is None
                else left.end_utc.isoformat(),
                "right_start_utc": right.start_utc.isoformat(),
                "right_end_utc": None
                if right.end_utc is None
                else right.end_utc.isoformat(),
                "left_labels": list(left.labels),
                "right_labels": list(right.labels),
                "relation_type": predicate.value,
                "window_seconds": window_seconds,
                "gap_seconds": result.gap_seconds,
                "evaluation_status": result.evaluation_status.value,
                "holds": result.holds,
                "reason": result.reason,
                "camera_relation": _camera_relation(left, right),
                "time_basis": result.time_basis,
                "calculation_provenance": result.calculation_provenance,
                "vms_instance_id": scope.vms_instance_id,
            }
            if result.holds is True:
                payload["relation_id"] = (
                    f"rel-{left.observation_id}-{right.observation_id}-"
                    f"{predicate.value}-{int(window_seconds)}"
                )
                established.append(payload)
            elif result.holds is None:
                indeterminate.append(payload)
        if truncated:
            break

    coverage = SearchCoverage.TRUNCATED if truncated else SearchCoverage.EXHAUSTED
    if not established:
        return ToolResult(
            status=ToolResultStatus.EMPTY,
            execution_id=execution_id,
            data=[],
            source_id=request.source_id or scope.vms_instance_id,
            retrieved_at=datetime.now(tz=UTC),
            coverage=coverage,
            provenance="temporal-engine:analyze_temporal_relations",
            usage={
                "tool_attempts": 1,
                "pairs_evaluated": pairs_evaluated,
                "established_count": 0,
                "indeterminate_count": len(indeterminate),
                "truncated": int(truncated),
            },
        )

    return ToolResult(
        status=ToolResultStatus.OK,
        execution_id=execution_id,
        data=established,
        source_id=request.source_id or scope.vms_instance_id,
        retrieved_at=datetime.now(tz=UTC),
        coverage=coverage,
        provenance="temporal-engine:analyze_temporal_relations",
        usage={
            "tool_attempts": 1,
            "pairs_evaluated": pairs_evaluated,
            "established_count": len(established),
            "indeterminate_count": len(indeterminate),
            "truncated": int(truncated),
        },
    )


def materialize_relations_from_tool_data(
    data: list[dict[str, Any]],
) -> list[Any]:
    """Convert OK temporal tool payloads into TemporalRelation contracts."""
    from nanexus_investigation.contracts.temporal import TemporalRelation

    relations: list[TemporalRelation] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        if item.get("record_kind") != "temporal_relation":
            continue
        if item.get("holds") is not True:
            continue
        status = TemporalEvaluationStatus(str(item["evaluation_status"]))
        if status is not TemporalEvaluationStatus.ESTABLISHED:
            continue
        relations.append(
            TemporalRelation(
                relation_id=str(item["relation_id"]),
                left_observation_id=str(item["left_observation_id"]),
                right_observation_id=str(item["right_observation_id"]),
                relation_type=TemporalRelationType(str(item["relation_type"])),
                evaluation_status=status,
                window_seconds=float(item["window_seconds"]),
                gap_seconds=(
                    None
                    if item.get("gap_seconds") is None
                    else float(item["gap_seconds"])
                ),
                camera_relation=str(item["camera_relation"]),
                time_basis=str(item["time_basis"]),
                calculation_provenance=str(item["calculation_provenance"]),
            )
        )
    return relations
