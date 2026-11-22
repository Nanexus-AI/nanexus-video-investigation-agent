"""Deterministic temporal predicates. G3-B §7."""

from __future__ import annotations

from dataclasses import dataclass

from nanexus_investigation.contracts.enums import (
    TemporalEvaluationStatus,
    TemporalRelationType,
)
from nanexus_investigation.contracts.temporal import TemporalRelation
from nanexus_investigation.evidence.time import (
    CompletedIntervalMicros,
    IntervalMicros,
    gap_seconds,
)

CALC_PROVENANCE = "temporal-v1c-microseconds"
TIME_BASIS = "completed_half_open_utc_micros"


@dataclass(frozen=True, slots=True)
class TemporalPredicateResult:
    """Result of evaluating one approved temporal predicate.

    ``holds`` is the engine's three-way deterministic outcome and does not add a
    G3 TemporalRelation evaluation status:

    - ``holds=True``: predicate is decidably true.
    - ``holds=False``: predicate is decidably false.
    - ``holds=None``: predicate cannot be decided from available evidence.

    ``evaluation_status`` is used when materializing a ``TemporalRelation``:

    - ``ESTABLISHED`` only with ``holds=True``.
    - ``INDETERMINATE`` when endpoints are insufficient (``holds=None``).
    - ``UNSUPPORTED`` when the predicate cannot be applied (``holds=None``).

    A decidable-false result is never materialized as an established relation.
    Callers must inspect ``holds``; do not infer false from ``UNSUPPORTED``.
    """

    relation_type: TemporalRelationType
    evaluation_status: TemporalEvaluationStatus
    holds: bool | None
    gap_seconds: float | None = None
    window_seconds: float | None = None
    calculation_provenance: str = CALC_PROVENANCE
    time_basis: str = TIME_BASIS
    reason: str = ""

    @property
    def is_true(self) -> bool:
        return self.holds is True

    @property
    def is_false(self) -> bool:
        return self.holds is False

    @property
    def is_unknown(self) -> bool:
        return self.holds is None

    @property
    def is_inapplicable(self) -> bool:
        return (
            self.holds is None
            and self.evaluation_status is TemporalEvaluationStatus.UNSUPPORTED
        )

    def to_relation(
        self,
        *,
        relation_id: str,
        left_observation_id: str,
        right_observation_id: str,
        camera_relation: str,
    ) -> TemporalRelation | None:
        """Materialize only non-false outcomes.

        Decidable-false results return ``None``. Established true relations use
        ``ESTABLISHED``. Unknown outcomes keep ``INDETERMINATE`` or
        ``UNSUPPORTED`` as calculated.
        """
        if self.holds is False:
            return None
        status = self.evaluation_status
        if self.holds is True:
            status = TemporalEvaluationStatus.ESTABLISHED
        return TemporalRelation(
            relation_id=relation_id,
            left_observation_id=left_observation_id,
            right_observation_id=right_observation_id,
            relation_type=self.relation_type,
            evaluation_status=status,
            window_seconds=self.window_seconds,
            gap_seconds=self.gap_seconds,
            camera_relation=camera_relation,
            time_basis=self.time_basis,
            calculation_provenance=self.calculation_provenance,
        )


def _false(
    relation_type: TemporalRelationType,
    reason: str,
    **kwargs: float | None,
) -> TemporalPredicateResult:
    """Return a decidable-false predicate outcome.

    G3 has no ``false`` evaluation status. ``holds=False`` carries that
    meaning. ``evaluation_status`` stays non-``ESTABLISHED`` and must not be
    read as ``is_inapplicable``; inapplicable cases use ``holds=None``.
    """
    return TemporalPredicateResult(
        relation_type=relation_type,
        evaluation_status=TemporalEvaluationStatus.UNSUPPORTED,
        holds=False,
        reason=reason,
        gap_seconds=kwargs.get("gap_seconds"),
        window_seconds=kwargs.get("window_seconds"),
    )


def _completed_pair(
    left: IntervalMicros,
    right: IntervalMicros,
    relation_type: TemporalRelationType,
) -> tuple[CompletedIntervalMicros, CompletedIntervalMicros] | TemporalPredicateResult:
    if left.end is None or right.end is None:
        return TemporalPredicateResult(
            relation_type=relation_type,
            evaluation_status=TemporalEvaluationStatus.INDETERMINATE,
            holds=None,
            reason="one or both intervals have an open endpoint",
        )
    if left.end.utc_micros < left.start.utc_micros or right.end.utc_micros < right.start.utc_micros:
        return TemporalPredicateResult(
            relation_type=relation_type,
            evaluation_status=TemporalEvaluationStatus.UNSUPPORTED,
            holds=None,
            reason="invalid inverted interval",
        )
    if left.is_degenerate or right.is_degenerate:
        return TemporalPredicateResult(
            relation_type=relation_type,
            evaluation_status=TemporalEvaluationStatus.UNSUPPORTED,
            holds=None,
            reason=(
                "degenerate intervals must not generate fabricated positive "
                "overlap or containment; predicates require completed "
                "nondegenerate intervals"
            ),
        )
    return (
        CompletedIntervalMicros(start=left.start, end=left.end),
        CompletedIntervalMicros(start=right.start, end=right.end),
    )


def _positive_overlap(left: CompletedIntervalMicros, right: CompletedIntervalMicros) -> bool:
    # G3-B §7.2: max(a_start, b_start) < min(a_end, b_end)
    # Touching endpoints are not positive-duration overlap.
    start = max(left.start.utc_micros, right.start.utc_micros)
    end = min(left.end.utc_micros, right.end.utc_micros)
    return start < end


def evaluate_predicate(
    left: IntervalMicros,
    right: IntervalMicros,
    relation_type: TemporalRelationType,
    *,
    window_seconds: float | None = None,
) -> TemporalPredicateResult:
    """Evaluate one G3-B §7.2 predicate over half-open UTC intervals."""
    pair = _completed_pair(left, right, relation_type)
    if isinstance(pair, TemporalPredicateResult):
        return pair
    a, b = pair

    if relation_type is TemporalRelationType.BEFORE:
        if a.end.utc_micros <= b.start.utc_micros:
            return TemporalPredicateResult(
                relation_type=relation_type,
                evaluation_status=TemporalEvaluationStatus.ESTABLISHED,
                holds=True,
                gap_seconds=gap_seconds(a, b),
            )
        return _false(relation_type, "before predicate does not hold")

    if relation_type is TemporalRelationType.AFTER:
        if a.start.utc_micros >= b.end.utc_micros:
            return TemporalPredicateResult(
                relation_type=relation_type,
                evaluation_status=TemporalEvaluationStatus.ESTABLISHED,
                holds=True,
                gap_seconds=gap_seconds(a, b),
            )
        return _false(relation_type, "after predicate does not hold")

    if relation_type is TemporalRelationType.OVERLAPS:
        if _positive_overlap(a, b):
            return TemporalPredicateResult(
                relation_type=relation_type,
                evaluation_status=TemporalEvaluationStatus.ESTABLISHED,
                holds=True,
            )
        return _false(relation_type, "no positive-duration overlap")

    if relation_type is TemporalRelationType.CONTAINS:
        if (
            a.start.utc_micros <= b.start.utc_micros
            and a.end.utc_micros >= b.end.utc_micros
        ):
            return TemporalPredicateResult(
                relation_type=relation_type,
                evaluation_status=TemporalEvaluationStatus.ESTABLISHED,
                holds=True,
            )
        return _false(relation_type, "contains predicate does not hold")

    if relation_type is TemporalRelationType.NEAR:
        if window_seconds is None:
            return TemporalPredicateResult(
                relation_type=relation_type,
                evaluation_status=TemporalEvaluationStatus.UNSUPPORTED,
                holds=None,
                reason="near requires an explicit window_seconds; no default is defined",
            )
        if window_seconds < 0:
            raise ValueError("window_seconds must be nonnegative")
        if _positive_overlap(a, b):
            return _false(
                relation_type,
                "near requires no positive-duration overlap",
                window_seconds=window_seconds,
            )
        gap = gap_seconds(a, b)
        if gap <= window_seconds:
            return TemporalPredicateResult(
                relation_type=relation_type,
                evaluation_status=TemporalEvaluationStatus.ESTABLISHED,
                holds=True,
                gap_seconds=gap,
                window_seconds=window_seconds,
            )
        return _false(
            relation_type,
            "gap exceeds near window",
            gap_seconds=gap,
            window_seconds=window_seconds,
        )

    raise ValueError(f"unknown relation type: {relation_type}")


def evaluate_relation(
    left: IntervalMicros,
    right: IntervalMicros,
    relation_type: TemporalRelationType,
    *,
    window_seconds: float | None = None,
) -> TemporalPredicateResult:
    """Alias for ``evaluate_predicate``."""
    return evaluate_predicate(
        left,
        right,
        relation_type,
        window_seconds=window_seconds,
    )
