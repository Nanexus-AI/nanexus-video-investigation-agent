"""Read-only recovery boundary assessment for durable E0 investigations.

This module inspects Investigation Persistence and a native LangGraph Checkpointer.
It does not resume the Graph, re-dispatch tools, mutate domain records, or act as
a second workflow controller.

Cross-database inspection is not an atomic global snapshot: callers should assess
quiescent fixtures. Domain terminal status is never inferred solely from a
checkpoint candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from nanexus_investigation.contracts.case import TERMINAL_CASE_STATUSES
from nanexus_investigation.contracts.enums import CaseStatus
from nanexus_investigation.persistence.budget import (
    DurableBudgetLedger,
    ReservationRecord,
    ReservationStatus,
)
from nanexus_investigation.persistence.repository import InvestigationRepository

_KIND_EVIDENCE = "evidence_ref"
_KIND_OBSERVATION = "observation"
_KIND_RECEIPT = "search_receipt"


class RecoveryDiagnostic(StrEnum):
    """Assessment classification. Explicitly separate from Case status vocabulary."""

    CASE_NOT_FOUND = "case_not_found"
    CASE_THREAD_MISMATCH = "case_thread_mismatch"
    CREATED_NOT_RUNNING = "created_not_running"
    RUNNING_NO_CHECKPOINT = "running_no_checkpoint"
    RESERVATION_OUTCOME_UNKNOWN = "reservation_outcome_unknown"
    ATTEMPT_WITHOUT_REGISTRATION = "attempt_without_registration"
    ARTIFACT_AHEAD_OF_CHECKPOINT = "artifact_ahead_of_checkpoint"
    GRAPH_FINISHED_DOMAIN_RUNNING = "graph_finished_domain_running"
    DOMAIN_FINALIZED_CHECKPOINT_BEHIND = "domain_finalized_checkpoint_behind"
    CHECKPOINT_MISSING_DOMAIN_EVIDENCE = "checkpoint_missing_domain_evidence"
    CONSISTENT_FINALIZED = "consistent_finalized"
    CONSISTENT_FAILED = "consistent_failed"
    INCONSISTENT = "inconsistent"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class CheckpointView:
    """Compact read-only view of native Checkpointer channel values."""

    present: bool
    thread_id: str
    checkpoint_id: str | None = None
    case_id: str | None = None
    phase: str | None = None
    evidence_ids: tuple[str, ...] = ()
    observation_ids: tuple[str, ...] = ()
    search_receipt_ids: tuple[str, ...] = ()
    report_id: str | None = None
    last_error: str | None = None


@dataclass(frozen=True, slots=True)
class ReservationView:
    reservation_id: str
    operation_id: str
    attempt_id: str
    status: ReservationStatus
    execution_id: str | None
    is_confirmed_physical_attempt: bool
    outcome_unknown: bool


@dataclass(frozen=True, slots=True)
class DomainView:
    case_id: str
    case_status: CaseStatus | None
    report_id: str | None
    report_execution_status: CaseStatus | None
    registration_operation_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    observation_ids: tuple[str, ...]
    receipt_ids: tuple[str, ...]
    reservations: tuple[ReservationView, ...]
    operation_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RecoveryAssessment:
    """Deterministic recovery-boundary assessment for one Case/thread pair.

    ``diagnostic`` is not a Case status. ``domain_finalized`` is true only when
    Investigation Persistence holds an authoritative terminal Case record (and
    report when applicable). Graph candidates alone never set it.
    """

    case_id: str
    thread_id: str
    diagnostic: RecoveryDiagnostic
    domain: DomainView
    checkpoint: CheckpointView
    invariants_checked: tuple[str, ...]
    inconsistencies: tuple[str, ...]
    uncertainties: tuple[str, ...]
    secondary_diagnostics: tuple[RecoveryDiagnostic, ...] = ()
    requires_manual_review: bool = False
    automatic_tool_redispatch_prohibited: bool = True
    domain_finalized: bool = False
    authoritative_report_present: bool = False
    read_only: bool = True
    notes: tuple[str, ...] = ()


def _reservation_view(record: ReservationRecord) -> ReservationView:
    return ReservationView(
        reservation_id=record.reservation_id,
        operation_id=record.operation_id,
        attempt_id=record.attempt_id,
        status=record.status,
        execution_id=record.execution_id,
        is_confirmed_physical_attempt=record.is_confirmed_physical_attempt,
        outcome_unknown=record.status
        in {ReservationStatus.RESERVED, ReservationStatus.UNRESOLVED},
    )


def _ids_for_kind(refs: list[tuple[str, str]], kind: str) -> tuple[str, ...]:
    return tuple(artifact_id for artifact_kind, artifact_id in refs if artifact_kind == kind)


def read_checkpoint_view(checkpointer: Any, *, thread_id: str) -> CheckpointView:
    """Read native Checkpointer state for one thread. Read-only."""
    config = {"configurable": {"thread_id": thread_id}}
    tup = checkpointer.get_tuple(config)
    if tup is None:
        return CheckpointView(present=False, thread_id=thread_id)
    values = tup.checkpoint.get("channel_values") or {}
    checkpoint_id = None
    if isinstance(tup.config, dict):
        checkpoint_id = tup.config.get("configurable", {}).get("checkpoint_id")
    evidence = values.get("evidence_ids") or []
    observations = values.get("observation_ids") or []
    receipts = values.get("search_receipt_ids") or []
    return CheckpointView(
        present=True,
        thread_id=thread_id,
        checkpoint_id=str(checkpoint_id) if checkpoint_id is not None else None,
        case_id=values.get("case_id"),
        phase=values.get("phase"),
        evidence_ids=tuple(evidence),
        observation_ids=tuple(observations),
        search_receipt_ids=tuple(receipts),
        report_id=values.get("report_id"),
        last_error=values.get("last_error"),
    )


def read_domain_view(
    repo: InvestigationRepository,
    ledger: DurableBudgetLedger,
    *,
    case_id: str,
) -> DomainView:
    """Read Investigation Persistence facts for one Case. Read-only."""
    case = repo.get_case(case_id)
    report = repo.get_report(case_id)
    refs = repo.list_artifact_refs(case_id)
    # Use marker-table ids so incomplete groups remain visible for integrity checks.
    group_ops = repo.list_registration_group_operation_ids(case_id)
    reservations = ledger.list_reservations(case_id) if case is not None else []
    return DomainView(
        case_id=case_id,
        case_status=None if case is None else case.status,
        report_id=None if report is None else report.report_id,
        report_execution_status=None if report is None else report.execution_status,
        registration_operation_ids=tuple(group_ops),
        evidence_ids=_ids_for_kind(refs, _KIND_EVIDENCE),
        observation_ids=_ids_for_kind(refs, _KIND_OBSERVATION),
        receipt_ids=_ids_for_kind(refs, _KIND_RECEIPT),
        reservations=tuple(_reservation_view(item) for item in reservations),
        operation_ids=tuple(repo.list_operation_ids(case_id)),
    )


def _checkpoint_looks_finished(checkpoint: CheckpointView) -> bool:
    if not checkpoint.present:
        return False
    phase = (checkpoint.phase or "").lower()
    if checkpoint.report_id:
        return True
    return phase in {
        "completed",
        "terminated",
        "output_gate_rejected",
        "output_gate_failed",
    }


def assess_recovery(
    *,
    case_id: str,
    thread_id: str,
    repo: InvestigationRepository,
    ledger: DurableBudgetLedger,
    checkpointer: Any,
) -> RecoveryAssessment:
    """Assess recovery boundaries for one known Case and Graph thread.

    Read-only against already-open Investigation Persistence and Checkpointer
    handles. Never resumes the Graph, never invokes tools, never creates
    reservations, and never mutates Case/report/artifact rows. Callers must
    open databases separately; this function does not initialize schema.
    """
    if not case_id.strip() or not thread_id.strip():
        raise ValueError("case_id and thread_id are required")

    domain = read_domain_view(repo, ledger, case_id=case_id)
    checkpoint = read_checkpoint_view(checkpointer, thread_id=thread_id)

    invariants: list[str] = [
        "domain_case_status_authoritative",
        "checkpoint_not_authoritative_for_terminal_status",
        "automatic_tool_redispatch_prohibited",
        "cross_database_inspection_not_atomic",
        "registration_group_completeness_via_existing_loader",
        "thread_id_alone_is_not_case_binding",
    ]
    inconsistencies: list[str] = []
    uncertainties: list[str] = []
    secondary: list[RecoveryDiagnostic] = []
    notes: list[str] = [
        "Assessment uses separate Checkpointer and domain databases; "
        "not an atomic cross-file snapshot.",
        "Assessment is read-only over caller-supplied open stores.",
    ]

    # Validate registration groups are complete (EB2A integrity) — loader raises if not.
    incomplete_groups = 0
    for operation_id in domain.registration_operation_ids:
        try:
            group = repo.get_registration_group(operation_id)
            if group is None:
                incomplete_groups += 1
                inconsistencies.append(
                    f"registration group marker missing on load: {operation_id}"
                )
        except Exception as exc:  # noqa: BLE001 — surface integrity as inconsistency
            incomplete_groups += 1
            inconsistencies.append(
                f"registration group incomplete or inconsistent: {operation_id}: {exc}"
            )
    invariants.append("registration_groups_loadable")

    # Authoritative report citation integrity for this Case only.
    report_citation_failure = False
    report = repo.get_report(case_id)
    if report is not None:
        for evidence_id in report.evidence_index:
            if evidence_id not in domain.evidence_ids:
                report_citation_failure = True
                inconsistencies.append(
                    f"authoritative report cites unpersisted evidence: {evidence_id}"
                )

    if domain.case_status is None:
        return RecoveryAssessment(
            case_id=case_id,
            thread_id=thread_id,
            diagnostic=RecoveryDiagnostic.CASE_NOT_FOUND,
            domain=domain,
            checkpoint=checkpoint,
            invariants_checked=tuple(invariants),
            inconsistencies=tuple(inconsistencies),
            uncertainties=("case row absent in Investigation Persistence",),
            requires_manual_review=True,
            automatic_tool_redispatch_prohibited=True,
            domain_finalized=False,
            authoritative_report_present=False,
            notes=tuple(notes),
        )

    # Case/thread binding: distinguish missing, mismatched, and unbound checkpoints.
    checkpoint_bound = False
    if not checkpoint.present:
        notes.append(
            "no checkpoint for requested thread "
            "(missing checkpoint is distinct from a Case/thread mismatch)"
        )
    elif not (checkpoint.case_id and str(checkpoint.case_id).strip()):
        uncertainties.append(
            "checkpoint present for thread but lacks case_id; "
            "caller thread_id alone is not positive Case/thread binding evidence"
        )
        notes.append(
            "checkpoint artifact references are not treated as bound to the "
            "requested Case without case_id agreement"
        )
        invariants.append("checkpoint_binding_uncertain")
    elif checkpoint.case_id != case_id:
        inconsistencies.append(
            f"checkpoint case_id {checkpoint.case_id} does not match assessed "
            f"case_id {case_id}"
        )
        notes.append(
            "unrelated checkpoint is not applied as evidence for the requested Case"
        )
        return RecoveryAssessment(
            case_id=case_id,
            thread_id=thread_id,
            diagnostic=RecoveryDiagnostic.CASE_THREAD_MISMATCH,
            domain=domain,
            # Preserve checkpoint facts for diagnosis, but domain view stays
            # scoped to the requested Case only.
            checkpoint=checkpoint,
            invariants_checked=tuple(invariants),
            inconsistencies=tuple(inconsistencies),
            uncertainties=tuple(uncertainties),
            requires_manual_review=True,
            automatic_tool_redispatch_prohibited=True,
            domain_finalized=domain.case_status in TERMINAL_CASE_STATUSES,
            authoritative_report_present=domain.report_id is not None,
            notes=tuple(notes),
        )
    else:
        checkpoint_bound = True
        invariants.append("checkpoint_case_id_matches_requested_case")

    missing_evidence_refs: list[str] = []
    missing_observation_refs: list[str] = []
    missing_receipt_refs: list[str] = []
    if checkpoint_bound:
        missing_evidence_refs = [
            evidence_id
            for evidence_id in checkpoint.evidence_ids
            if evidence_id not in domain.evidence_ids
        ]
        missing_observation_refs = [
            observation_id
            for observation_id in checkpoint.observation_ids
            if observation_id not in domain.observation_ids
        ]
        missing_receipt_refs = [
            receipt_id
            for receipt_id in checkpoint.search_receipt_ids
            if receipt_id not in domain.receipt_ids
        ]
        if missing_evidence_refs or missing_observation_refs or missing_receipt_refs:
            inconsistencies.append(
                "checkpoint references domain artifacts that are not committed: "
                f"evidence={missing_evidence_refs}, "
                f"observations={missing_observation_refs}, "
                f"receipts={missing_receipt_refs}"
            )
            secondary.append(RecoveryDiagnostic.CHECKPOINT_MISSING_DOMAIN_EVIDENCE)

    unknown_reservations = [r for r in domain.reservations if r.outcome_unknown]
    # Confirmed/failed attempt accounting without a complete registration group:
    # do not infer a durable ToolResult.
    attempt_without_group = [
        r
        for r in domain.reservations
        if r.status
        in {
            ReservationStatus.CONFIRMED,
            ReservationStatus.FAILED,
        }
        and r.operation_id not in domain.registration_operation_ids
        and not str(r.operation_id).endswith(":__finalize__")
    ]

    # A tool registration group is "ahead" when no bound checkpoint exists or a
    # bound checkpoint does not reference the group's members. Finalize-only ops
    # are excluded.
    artifact_ahead = False
    tool_groups = [
        op
        for op in domain.registration_operation_ids
        if not str(op).endswith(":__finalize__")
    ]
    if tool_groups and not checkpoint_bound:
        artifact_ahead = True
    elif checkpoint_bound and tool_groups:
        if (
            not checkpoint.evidence_ids
            and not checkpoint.search_receipt_ids
            and not checkpoint.observation_ids
        ):
            artifact_ahead = True
        else:
            referenced = (
                set(checkpoint.evidence_ids)
                | set(checkpoint.observation_ids)
                | set(checkpoint.search_receipt_ids)
            )
            for group in repo.list_registration_groups(case_id):
                if str(group.operation_id).endswith(":__finalize__"):
                    continue
                member_ids = {
                    group.receipt_id,
                    *group.observation_ids,
                    *group.evidence_ids,
                }
                if member_ids and member_ids.isdisjoint(referenced):
                    artifact_ahead = True
                    notes.append(
                        f"registration group {group.operation_id} committed but not "
                        "referenced by checkpoint channel values"
                    )

    if unknown_reservations:
        uncertainties.append(
            "budget reservation(s) exist without confirmed execution outcome; "
            "reservation is not proof a physical VMS request occurred, and "
            "absence of a ToolResult is not proof the request never occurred"
        )
        secondary.append(RecoveryDiagnostic.RESERVATION_OUTCOME_UNKNOWN)
    if attempt_without_group:
        uncertainties.append(
            "attempt-accounting records exist without a committed registration "
            "group; durable ToolResult/artifacts are not inferred"
        )
        secondary.append(RecoveryDiagnostic.ATTEMPT_WITHOUT_REGISTRATION)
    if artifact_ahead:
        secondary.append(RecoveryDiagnostic.ARTIFACT_AHEAD_OF_CHECKPOINT)
        notes.append(
            "committed domain registration group may be reused by operation "
            "identity; no cross-database atomicity is claimed"
        )

    domain_finalized = domain.case_status in TERMINAL_CASE_STATUSES
    authoritative_report = domain.report_id is not None
    if domain_finalized and domain.case_status in {
        CaseStatus.COMPLETED,
        CaseStatus.PARTIAL,
    }:
        if domain.report_id is None:
            inconsistencies.append(
                f"terminal status {domain.case_status.value} lacks authoritative report"
            )
        elif (
            domain.report_execution_status is not None
            and domain.report_execution_status is not domain.case_status
        ):
            inconsistencies.append(
                "authoritative report.execution_status does not match Case status"
            )

    # Primary diagnostic selection (most specific recovery boundary).
    diagnostic = RecoveryDiagnostic.UNKNOWN
    requires_manual = bool(uncertainties) and not checkpoint_bound and checkpoint.present
    bound_finished = checkpoint_bound and _checkpoint_looks_finished(checkpoint)

    if domain.case_status is CaseStatus.CREATED:
        diagnostic = RecoveryDiagnostic.CREATED_NOT_RUNNING
        notes.append("Case creation committed; running transition not observed")
    elif domain.case_status is CaseStatus.FAILED and domain.report_id is None:
        diagnostic = RecoveryDiagnostic.CONSISTENT_FAILED
        if checkpoint_bound and not bound_finished:
            secondary.append(RecoveryDiagnostic.DOMAIN_FINALIZED_CHECKPOINT_BEHIND)
            notes.append(
                "domain failed terminal record is authoritative; stale checkpoint "
                "must not restart the Case"
            )
    elif domain.case_status in {CaseStatus.COMPLETED, CaseStatus.PARTIAL}:
        material_integrity_failure = (
            bool(missing_evidence_refs)
            or incomplete_groups > 0
            or report_citation_failure
        )
        if material_integrity_failure:
            diagnostic = RecoveryDiagnostic.INCONSISTENT
            requires_manual = True
            notes.append(
                "terminal Case cannot be classified consistent_finalized while "
                "required evidence or registration integrity is missing"
            )
        elif (
            not checkpoint_bound
            or not bound_finished
            or (
                domain.report_id is not None
                and checkpoint.report_id
                and checkpoint.report_id != domain.report_id
            )
            or (checkpoint_bound and checkpoint.report_id is None)
        ):
            diagnostic = RecoveryDiagnostic.DOMAIN_FINALIZED_CHECKPOINT_BEHIND
            notes.append(
                "authoritative domain Case/Report remain intact; do not restart "
                "or duplicate finalization"
            )
        else:
            diagnostic = RecoveryDiagnostic.CONSISTENT_FINALIZED
    elif domain.case_status is CaseStatus.RUNNING:
        if missing_evidence_refs or missing_observation_refs or missing_receipt_refs:
            diagnostic = RecoveryDiagnostic.CHECKPOINT_MISSING_DOMAIN_EVIDENCE
            requires_manual = True
            notes.append(
                "fail closed: unsupported claims must not be validated from "
                "unresolvable checkpoint references"
            )
        elif bound_finished:
            diagnostic = RecoveryDiagnostic.GRAPH_FINISHED_DOMAIN_RUNNING
            requires_manual = True
            notes.append(
                "Outcome B: Graph candidate/checkpoint is not an authoritative "
                "domain finalization"
            )
        elif not checkpoint.present:
            if attempt_without_group:
                diagnostic = RecoveryDiagnostic.ATTEMPT_WITHOUT_REGISTRATION
                requires_manual = True
            elif unknown_reservations:
                diagnostic = RecoveryDiagnostic.RESERVATION_OUTCOME_UNKNOWN
                requires_manual = True
            elif artifact_ahead:
                diagnostic = RecoveryDiagnostic.ARTIFACT_AHEAD_OF_CHECKPOINT
            else:
                diagnostic = RecoveryDiagnostic.RUNNING_NO_CHECKPOINT
                uncertainties.append(
                    "running Case with no checkpoint does not prove that external "
                    "work never occurred"
                )
        elif not checkpoint_bound:
            # Present but unbound: domain-only conclusions; keep uncertainties.
            if attempt_without_group:
                diagnostic = RecoveryDiagnostic.ATTEMPT_WITHOUT_REGISTRATION
                requires_manual = True
            elif unknown_reservations:
                diagnostic = RecoveryDiagnostic.RESERVATION_OUTCOME_UNKNOWN
                requires_manual = True
            elif artifact_ahead:
                diagnostic = RecoveryDiagnostic.ARTIFACT_AHEAD_OF_CHECKPOINT
            else:
                diagnostic = RecoveryDiagnostic.UNKNOWN
                requires_manual = True
        elif attempt_without_group:
            diagnostic = RecoveryDiagnostic.ATTEMPT_WITHOUT_REGISTRATION
            requires_manual = True
        elif unknown_reservations:
            diagnostic = RecoveryDiagnostic.RESERVATION_OUTCOME_UNKNOWN
            requires_manual = True
        elif artifact_ahead:
            diagnostic = RecoveryDiagnostic.ARTIFACT_AHEAD_OF_CHECKPOINT
        else:
            diagnostic = RecoveryDiagnostic.UNKNOWN
            uncertainties.append(
                "running Case with bound checkpoint present; no stronger boundary matched"
            )
    elif domain.case_status is CaseStatus.CANCELLED:
        diagnostic = RecoveryDiagnostic.UNKNOWN
        notes.append("cancelled Cases are terminal; no automatic restart")
        domain_finalized = True
    elif domain.case_status is CaseStatus.AWAITING_INPUT:
        diagnostic = RecoveryDiagnostic.UNKNOWN
        uncertainties.append("awaiting_input observed; resume not assessed in EB2C1")

    if inconsistencies and diagnostic not in {
        RecoveryDiagnostic.CHECKPOINT_MISSING_DOMAIN_EVIDENCE,
        RecoveryDiagnostic.INCONSISTENT,
        RecoveryDiagnostic.CASE_THREAD_MISMATCH,
    }:
        if diagnostic in {
            RecoveryDiagnostic.CONSISTENT_FINALIZED,
            RecoveryDiagnostic.CONSISTENT_FAILED,
        }:
            secondary.insert(0, diagnostic)
            diagnostic = RecoveryDiagnostic.INCONSISTENT
            requires_manual = True
        else:
            secondary.append(RecoveryDiagnostic.INCONSISTENT)
            requires_manual = True

    # Keep critical concurrent findings visible as secondary diagnostics.
    if unknown_reservations and (
        RecoveryDiagnostic.RESERVATION_OUTCOME_UNKNOWN not in secondary
        and diagnostic is not RecoveryDiagnostic.RESERVATION_OUTCOME_UNKNOWN
    ):
        secondary.append(RecoveryDiagnostic.RESERVATION_OUTCOME_UNKNOWN)
    if (
        (incomplete_groups or report_citation_failure)
        and diagnostic is not RecoveryDiagnostic.INCONSISTENT
    ):
        secondary.append(RecoveryDiagnostic.INCONSISTENT)

    secondary_unique = tuple(
        dict.fromkeys(item for item in secondary if item is not diagnostic)
    )

    return RecoveryAssessment(
        case_id=case_id,
        thread_id=thread_id,
        diagnostic=diagnostic,
        domain=domain,
        checkpoint=checkpoint,
        invariants_checked=tuple(dict.fromkeys(invariants)),
        inconsistencies=tuple(inconsistencies),
        uncertainties=tuple(uncertainties),
        secondary_diagnostics=secondary_unique,
        requires_manual_review=requires_manual
        or bool(inconsistencies)
        or bool(uncertainties),
        automatic_tool_redispatch_prohibited=True,
        domain_finalized=domain_finalized,
        authoritative_report_present=authoritative_report,
        notes=tuple(dict.fromkeys(notes)),
    )
