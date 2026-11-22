"""Temporal predicate tests. G3-B §7."""

from datetime import datetime, timedelta, timezone

import pytest

from nanexus_investigation.contracts.enums import (
    TemporalEvaluationStatus,
    TemporalRelationType,
)
from nanexus_investigation.evidence.relations import evaluate_predicate
from nanexus_investigation.evidence.time import build_interval

UTC = timezone.utc
T0 = datetime(2026, 11, 19, 20, 0, tzinfo=UTC)


def _iv(start_offset_s: int, end_offset_s: int | None):
    start = T0 + timedelta(seconds=start_offset_s)
    end = None if end_offset_s is None else T0 + timedelta(seconds=end_offset_s)
    return build_interval(start, end)


def test_before_and_after():
    left = _iv(0, 10)
    right = _iv(10, 20)
    before = evaluate_predicate(left, right, TemporalRelationType.BEFORE)
    after = evaluate_predicate(right, left, TemporalRelationType.AFTER)
    assert before.holds is True
    assert after.holds is True
    assert before.evaluation_status is TemporalEvaluationStatus.ESTABLISHED


def test_touching_endpoint_is_before_not_overlap():
    left = _iv(0, 10)
    right = _iv(10, 20)
    overlap = evaluate_predicate(left, right, TemporalRelationType.OVERLAPS)
    before = evaluate_predicate(left, right, TemporalRelationType.BEFORE)
    assert overlap.holds is False
    assert before.holds is True


def test_overlap_and_contains():
    outer = _iv(0, 30)
    inner = _iv(5, 15)
    overlap = evaluate_predicate(outer, inner, TemporalRelationType.OVERLAPS)
    contains = evaluate_predicate(outer, inner, TemporalRelationType.CONTAINS)
    assert overlap.holds is True
    assert contains.holds is True


def test_equal_boundaries_overlap():
    a = _iv(0, 10)
    b = _iv(0, 10)
    assert evaluate_predicate(a, b, TemporalRelationType.OVERLAPS).holds is True
    assert evaluate_predicate(a, b, TemporalRelationType.CONTAINS).holds is True


def test_near_requires_explicit_tolerance():
    left = _iv(0, 10)
    right = _iv(15, 25)
    missing = evaluate_predicate(left, right, TemporalRelationType.NEAR)
    assert missing.holds is None
    assert missing.evaluation_status is TemporalEvaluationStatus.UNSUPPORTED

    near = evaluate_predicate(
        left, right, TemporalRelationType.NEAR, window_seconds=5
    )
    assert near.holds is True
    assert near.gap_seconds == 5.0

    far = evaluate_predicate(
        left, right, TemporalRelationType.NEAR, window_seconds=4
    )
    assert far.holds is False


def test_before_and_near_can_both_hold():
    left = _iv(0, 10)
    right = _iv(12, 20)
    before = evaluate_predicate(left, right, TemporalRelationType.BEFORE)
    near = evaluate_predicate(
        left, right, TemporalRelationType.NEAR, window_seconds=5
    )
    assert before.holds is True
    assert near.holds is True


def test_open_endpoint_is_indeterminate():
    left = _iv(0, None)
    right = _iv(10, 20)
    result = evaluate_predicate(left, right, TemporalRelationType.BEFORE)
    assert result.holds is None
    assert result.evaluation_status is TemporalEvaluationStatus.INDETERMINATE


def test_near_with_open_endpoint_is_indeterminate():
    left = _iv(0, 10)
    right = _iv(12, None)
    result = evaluate_predicate(
        left, right, TemporalRelationType.NEAR, window_seconds=5
    )
    assert result.evaluation_status is TemporalEvaluationStatus.INDETERMINATE


def test_degenerate_interval_unsupported_for_overlap():
    point = build_interval(T0, T0)
    other = _iv(0, 10)
    result = evaluate_predicate(point, other, TemporalRelationType.OVERLAPS)
    assert result.evaluation_status is TemporalEvaluationStatus.UNSUPPORTED
    assert result.holds is None


def test_known_utc_offsets():
    left = build_interval(
        datetime(2026, 11, 19, 15, 0, tzinfo=timezone(timedelta(hours=-5))),
        datetime(2026, 11, 19, 15, 5, tzinfo=timezone(timedelta(hours=-5))),
    )
    right = build_interval(
        datetime(2026, 11, 19, 20, 5, tzinfo=UTC),
        datetime(2026, 11, 19, 20, 10, tzinfo=UTC),
    )
    assert evaluate_predicate(left, right, TemporalRelationType.BEFORE).holds is True


def test_near_does_not_imply_identity_or_causation():
    left = _iv(0, 10)
    right = _iv(12, 20)
    result = evaluate_predicate(
        left, right, TemporalRelationType.NEAR, window_seconds=5
    )
    assert result.holds is True
    # The engine returns a temporal fact only. Identity/causation are out of scope.
    assert "identity" not in result.reason
    assert "cause" not in result.reason


def test_true_false_and_unknown_are_distinct():
    left = _iv(0, 10)
    right = _iv(20, 30)
    open_right = _iv(20, None)

    true = evaluate_predicate(left, right, TemporalRelationType.BEFORE)
    false = evaluate_predicate(right, left, TemporalRelationType.BEFORE)
    unknown = evaluate_predicate(left, open_right, TemporalRelationType.BEFORE)
    inapplicable = evaluate_predicate(left, right, TemporalRelationType.NEAR)

    assert true.is_true and true.evaluation_status is TemporalEvaluationStatus.ESTABLISHED
    assert false.is_false and not false.is_inapplicable
    assert false.to_relation(
        relation_id="rel-false",
        left_observation_id="a",
        right_observation_id="b",
        camera_relation="same_camera",
    ) is None
    assert unknown.is_unknown
    assert unknown.evaluation_status is TemporalEvaluationStatus.INDETERMINATE
    assert inapplicable.is_inapplicable
    assert inapplicable.holds is None
    # Decidable false must not be confused with inapplicable unsupported.
    assert false.holds is False
    assert inapplicable.holds is None
