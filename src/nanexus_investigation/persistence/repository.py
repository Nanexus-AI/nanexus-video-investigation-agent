"""Idempotent Investigation Persistence writes for E0 domain artifacts."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, TypeVar

from pydantic import BaseModel

from nanexus_investigation.contracts.case import (
    InvestigationCase,
    TERMINAL_CASE_STATUSES,
    validate_case_transition,
)
from nanexus_investigation.contracts.claim import Claim, InvestigationReport
from nanexus_investigation.contracts.enums import CaseStatus, EvidenceKind
from nanexus_investigation.contracts.evidence import EvidenceRef, Observation
from nanexus_investigation.contracts.search import SearchExecutionReceipt
from nanexus_investigation.contracts.temporal import TemporalRelation
from nanexus_investigation.persistence.db import InvestigationDatabase
from nanexus_investigation.persistence.errors import (
    CaseLifecycleError,
    DomainFinalizationError,
    IdempotencyConflictError,
)
from nanexus_investigation.persistence.operation_id import (
    OperationIdentity,
    payload_fingerprint,
)

UTC = timezone.utc
T = TypeVar("T", bound=BaseModel)

_KIND_CASE = "case"
_KIND_OBSERVATION = "observation"
_KIND_EVIDENCE = "evidence_ref"
_KIND_RECEIPT = "search_receipt"
_KIND_TEMPORAL_RELATION = "temporal_relation"
_KIND_CLAIM = "claim"
_KIND_REPORT = "investigation_report"
_FINALIZE_PLAN_STEP = "__finalize__"
_FINALIZE_TOOL = "finalize_case"


@dataclass(frozen=True, slots=True)
class RegistrationGroupResult:
    """Committed logical registration group for one tool-result operation.

    A single tool result may register N observation/evidence pairs. The
    ``observation`` / ``evidence`` properties expose the first pair for
    backward-compatible single-event callers.

    Derived ``TemporalRelation`` members may share the same registration group
    as ``temporal_relation`` artifacts (additive kind; no schema bump). Source
    observations for those relations must already exist as durable evidence.
    """

    operation_id: str
    group_fingerprint: str
    receipt: SearchExecutionReceipt
    observations: tuple[Observation, ...]
    evidences: tuple[EvidenceRef, ...]
    relations: tuple[TemporalRelation, ...] = ()
    reused: bool = False

    @property
    def receipt_id(self) -> str:
        return self.receipt.search_execution_id

    @property
    def observation(self) -> Observation | None:
        return self.observations[0] if self.observations else None

    @property
    def evidence(self) -> EvidenceRef | None:
        return self.evidences[0] if self.evidences else None

    @property
    def observation_id(self) -> str | None:
        return None if self.observation is None else self.observation.observation_id

    @property
    def evidence_id(self) -> str | None:
        return None if self.evidence is None else self.evidence.evidence_id

    @property
    def observation_ids(self) -> tuple[str, ...]:
        return tuple(item.observation_id for item in self.observations)

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(item.evidence_id for item in self.evidences)


@dataclass(frozen=True, slots=True)
class FinalizationResult:
    """Authoritative domain terminal Case outcome.

    For ``failed`` (and cancelled without a report), no InvestigationReport is
    stored. Terminal failure is recorded solely as the Case status transition
    plus the finalize operation identity — not as a fabricated successful report.
    """

    case: InvestigationCase
    terminal_status: CaseStatus
    report: InvestigationReport | None
    finalization_fingerprint: str
    reused: bool


@dataclass(frozen=True, slots=True)
class AcceptedReportBinding:
    """Binds a specific InvestigationReport to Output Gate acceptance.

    A bare boolean is not sufficient. Domain finalization persists only the
    report fingerprint-bound here after deterministic gate acceptance.
    """

    case_id: str
    terminal_status: CaseStatus
    report: InvestigationReport
    report_fingerprint: str

    @classmethod
    def from_gate_report(
        cls,
        *,
        case_id: str,
        terminal_status: CaseStatus,
        report: InvestigationReport,
    ) -> "AcceptedReportBinding":
        if report.case_id != case_id:
            raise DomainFinalizationError(
                "accepted report case_id does not match finalization case_id"
            )
        if report.execution_status is not terminal_status:
            raise DomainFinalizationError(
                "accepted report.execution_status must match terminal_status"
            )
        if terminal_status not in {CaseStatus.COMPLETED, CaseStatus.PARTIAL}:
            raise DomainFinalizationError(
                "AcceptedReportBinding applies only to completed/partial reports"
            )
        payload = report.model_dump(mode="json")
        _forbidden_payload_keys(payload)
        return cls(
            case_id=case_id,
            terminal_status=terminal_status,
            report=report,
            report_fingerprint=payload_fingerprint(payload),
        )

    def assert_matches(self, report: InvestigationReport) -> None:
        payload = report.model_dump(mode="json")
        _forbidden_payload_keys(payload)
        if payload_fingerprint(payload) != self.report_fingerprint:
            raise DomainFinalizationError(
                "report content does not match Output Gate accepted binding"
            )
        if report.report_id != self.report.report_id:
            raise DomainFinalizationError(
                "report_id does not match Output Gate accepted binding"
            )


def _now() -> str:
    return datetime.now(tz=UTC).isoformat()


def _dump(model: BaseModel) -> str:
    return json.dumps(model.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))


def _forbidden_payload_keys(payload: dict[str, Any]) -> None:
    forbidden = {
        "password",
        "token",
        "api_key",
        "secret",
        "authorization",
        "credential",
        "credentials",
        "media_bytes",
        "raw_video",
        "image_bytes",
    }
    for key in payload:
        lower = key.lower()
        if lower in forbidden or lower.endswith("_token"):
            raise ValueError(f"persistence payload must not contain secrets or media: {key}")
        value = payload[key]
        if isinstance(value, (bytes, bytearray)):
            raise ValueError("persistence payload must not contain raw bytes")
        if isinstance(value, str) and value.startswith("data:"):
            raise ValueError("persistence payload must not contain embedded media data URIs")


def _normalize_artifact_pairs(
    *,
    observation: Observation | None,
    evidence: EvidenceRef | None,
    observations: tuple[Observation, ...] | list[Observation] | None,
    evidences: tuple[EvidenceRef, ...] | list[EvidenceRef] | None,
) -> tuple[tuple[Observation, ...], tuple[EvidenceRef, ...]]:
    has_singular = observation is not None or evidence is not None
    has_sequence = observations is not None or evidences is not None
    if has_singular and has_sequence:
        raise ValueError(
            "pass either observation/evidence or observations/evidences, not both"
        )
    if has_sequence:
        obs_list = tuple(observations or ())
        ev_list = tuple(evidences or ())
    elif has_singular:
        if (observation is None) != (evidence is None):
            raise ValueError(
                "observation and evidence must both be present or both absent"
            )
        obs_list = () if observation is None else (observation,)
        ev_list = () if evidence is None else (evidence,)
    else:
        obs_list = ()
        ev_list = ()

    if len(obs_list) != len(ev_list):
        raise ValueError("observations and evidences must have equal length")

    seen_native: set[str] = set()
    for obs_item, ev_item in zip(obs_list, ev_list, strict=True):
        if (
            obs_item.source_locator.native_record_id
            != ev_item.source_locator.native_record_id
        ):
            raise ValueError(
                "observation and evidence must share native source identity"
            )
        if (
            obs_item.source_locator.vms_instance_id
            != ev_item.source_locator.vms_instance_id
        ):
            raise ValueError("observation and evidence must share VMS instance")
        native = obs_item.source_locator.native_record_id
        if native in seen_native:
            raise ValueError(
                f"duplicate native_record_id in registration group: {native}"
            )
        seen_native.add(native)
    return obs_list, ev_list


def _pair_observations_and_evidence(
    observations: tuple[Observation, ...],
    evidences: tuple[EvidenceRef, ...],
) -> tuple[tuple[Observation, ...], tuple[EvidenceRef, ...]]:
    """Pair loaded artifacts by native id while preserving observation order."""
    if not observations:
        return (), ()
    by_native: dict[str, EvidenceRef] = {}
    for evidence_item in evidences:
        native = evidence_item.source_locator.native_record_id
        if native in by_native:
            raise IdempotencyConflictError(
                f"duplicate evidence native_record_id in group: {native}"
            )
        by_native[native] = evidence_item
    ordered_ev: list[EvidenceRef] = []
    for observation_item in observations:
        native = observation_item.source_locator.native_record_id
        evidence_item = by_native.get(native)
        if evidence_item is None:
            raise IdempotencyConflictError(
                f"observation native_record_id {native} missing evidence pair"
            )
        ordered_ev.append(evidence_item)
    if len(ordered_ev) != len(by_native):
        raise IdempotencyConflictError(
            "evidence artifacts remain unpaired after native-id matching"
        )
    return observations, tuple(ordered_ev)


def _artifact_group_fingerprint(
    *,
    operation_id: str,
    receipt: SearchExecutionReceipt,
    observations: tuple[Observation, ...],
    evidences: tuple[EvidenceRef, ...],
    relations: tuple[TemporalRelation, ...] = (),
) -> str:
    member_kinds = [_KIND_RECEIPT]
    payloads: dict[str, Any] = {_KIND_RECEIPT: receipt.model_dump(mode="json")}
    if observations:
        member_kinds.extend([_KIND_OBSERVATION, _KIND_EVIDENCE])
        payloads[_KIND_OBSERVATION] = [
            item.model_dump(mode="json") for item in observations
        ]
        payloads[_KIND_EVIDENCE] = [
            item.model_dump(mode="json") for item in evidences
        ]
    elif evidences:
        # Media-only registration: evidence_ref without new observations.
        member_kinds.append(_KIND_EVIDENCE)
        payloads[_KIND_EVIDENCE] = [
            item.model_dump(mode="json") for item in evidences
        ]
    if relations:
        member_kinds.append(_KIND_TEMPORAL_RELATION)
        payloads[_KIND_TEMPORAL_RELATION] = [
            item.model_dump(mode="json") for item in relations
        ]
    return payload_fingerprint(
        {
            "operation_id": operation_id,
            "member_kinds": member_kinds,
            "payloads": payloads,
        }
    )


def _legacy_single_artifact_group_fingerprint(
    *,
    operation_id: str,
    receipt: SearchExecutionReceipt,
    observation: Observation | None,
    evidence: EvidenceRef | None,
) -> str:
    """Pre-v3 fingerprint shape (singular observation/evidence payloads)."""
    member_kinds = [_KIND_RECEIPT]
    payloads: dict[str, Any] = {_KIND_RECEIPT: receipt.model_dump(mode="json")}
    if observation is not None and evidence is not None:
        member_kinds.extend([_KIND_OBSERVATION, _KIND_EVIDENCE])
        payloads[_KIND_OBSERVATION] = observation.model_dump(mode="json")
        payloads[_KIND_EVIDENCE] = evidence.model_dump(mode="json")
    return payload_fingerprint(
        {
            "operation_id": operation_id,
            "member_kinds": member_kinds,
            "payloads": payloads,
        }
    )


def _case_content_fingerprint(case: InvestigationCase) -> str:
    """Admission identity fingerprint: Case content excluding lifecycle status."""
    payload = case.model_dump(mode="json")
    payload.pop("status", None)
    _forbidden_payload_keys(payload)
    return payload_fingerprint(payload)


def _finalize_operation_identity(
    *, case_id: str, finalization_fingerprint: str
) -> OperationIdentity:
    return OperationIdentity(
        operation_id=f"op:{case_id}:{_FINALIZE_PLAN_STEP}",
        case_id=case_id,
        plan_step_id=_FINALIZE_PLAN_STEP,
        tool_name=_FINALIZE_TOOL,
        content_fingerprint=finalization_fingerprint,
    )


class InvestigationRepository:
    """Minimal domain persistence boundary. Not a second workflow controller."""

    def __init__(self, db: InvestigationDatabase) -> None:
        self._db = db
        # Test-local fault injection points for atomic registration boundaries.
        # Not a production chaos harness.
        self._fault_hooks: dict[str, Callable[[], None]] = {}
        self._txn_conn: sqlite3.Connection | None = None

    @property
    def db(self) -> InvestigationDatabase:
        return self._db

    def set_fault_hook(self, point: str, hook: Callable[[], None] | None) -> None:
        """Install or clear a deterministic test-only fault hook."""
        if hook is None:
            self._fault_hooks.pop(point, None)
        else:
            self._fault_hooks[point] = hook

    def clear_fault_hooks(self) -> None:
        self._fault_hooks.clear()

    def _fault(self, point: str) -> None:
        hook = self._fault_hooks.get(point)
        if hook is not None:
            hook()

    def upsert_case(self, case: InvestigationCase) -> InvestigationCase:
        """Persist Case row idempotently by case_id (legacy helper for fixtures).

        Prefer ``admit_case`` / ``transition_case_to_running`` / ``finalize_case``
        for lifecycle-correct durable boundaries.
        """
        payload = case.model_dump(mode="json")
        _forbidden_payload_keys(payload)
        fingerprint = payload_fingerprint(payload)
        with self._db.transaction() as conn:
            existing = conn.execute(
                "SELECT payload_json, payload_fingerprint FROM nx_cases WHERE case_id = ?",
                (case.case_id,),
            ).fetchone()
            if existing is not None:
                if existing["payload_fingerprint"] != fingerprint:
                    raise IdempotencyConflictError(
                        f"case_id {case.case_id} already exists with different content"
                    )
                return InvestigationCase.model_validate_json(existing["payload_json"])

            self._insert_case_conn(conn, case, fingerprint=fingerprint)
        return case

    def get_case(self, case_id: str) -> InvestigationCase | None:
        return self._get_case_conn(self._db.conn, case_id)

    def admit_case(self, case: InvestigationCase) -> InvestigationCase:
        """Commit Case admission with status ``created`` (one transaction).

        Effective scope must be explicitly present and satisfy the approved pure
        scope-containment rules. E0 treats that scope as trusted caller input —
        this method is not live VMS authorization. Duplicate equivalent admission
        is idempotent. Conflicting content and terminal Cases are rejected.
        """
        if case.status is not CaseStatus.CREATED:
            raise CaseLifecycleError("admit_case requires status=created")
        self._validate_admission_case(case)
        content_fp = _case_content_fingerprint(case)
        full_fp = payload_fingerprint(case.model_dump(mode="json"))

        with self._db.transaction() as conn:
            self._fault("before_case_admit")
            existing = self._get_case_conn(conn, case.case_id)
            if existing is not None:
                if existing.status in TERMINAL_CASE_STATUSES:
                    raise IdempotencyConflictError(
                        f"terminal case {case.case_id} cannot be re-admitted"
                    )
                if _case_content_fingerprint(existing) != content_fp:
                    raise IdempotencyConflictError(
                        f"case_id {case.case_id} already exists with different content"
                    )
                # Equivalent identity: return durable row (created or already running).
                return existing

            self._insert_case_conn(conn, case, fingerprint=full_fp)
            self._fault("after_case_admit")
        return case

    def transition_case_to_running(self, case_id: str) -> InvestigationCase:
        """Durably transition ``created`` → ``running`` before Graph invocation."""
        with self._db.transaction() as conn:
            self._fault("before_running_transition")
            current = self._get_case_conn(conn, case_id)
            if current is None:
                raise CaseLifecycleError(f"case {case_id} is not admitted")
            if current.status is CaseStatus.RUNNING:
                return current
            if current.status in TERMINAL_CASE_STATUSES:
                raise CaseLifecycleError(
                    f"terminal case {case_id} cannot return to running"
                )
            try:
                validate_case_transition(current.status, CaseStatus.RUNNING)
            except ValueError as exc:
                raise CaseLifecycleError(str(exc)) from exc
            updated = current.transition_to(CaseStatus.RUNNING)
            self._update_case_conn(conn, updated)
            self._fault("after_running_transition")
            return updated

    def admit_and_start_case(self, case: InvestigationCase) -> InvestigationCase:
        """Admit as ``created``, then separately commit ``running``.

        Two distinct transactions so ``created`` remains durable if the running
        transition fails. Graph invocation must not occur until this returns.
        """
        self.admit_case(case)
        return self.transition_case_to_running(case.case_id)

    def finalize_case(
        self,
        *,
        case_id: str,
        terminal_status: CaseStatus,
        accepted: AcceptedReportBinding | None = None,
        claims_by_id: dict[str, Claim] | None = None,
        relations_by_id: dict[str, TemporalRelation] | None = None,
    ) -> FinalizationResult:
        """Atomically persist terminal Case status and optional authoritative report.

        ``completed`` / ``partial`` require an ``AcceptedReportBinding`` produced
        from Output Gate acceptance of that exact report. A caller boolean is not
        accepted as authority. ``failed`` stores no InvestigationReport.
        """
        if terminal_status not in TERMINAL_CASE_STATUSES:
            raise DomainFinalizationError(
                f"finalize_case requires terminal status, got {terminal_status}"
            )
        report: InvestigationReport | None = None
        if terminal_status in {CaseStatus.COMPLETED, CaseStatus.PARTIAL}:
            if accepted is None:
                raise DomainFinalizationError(
                    "completed/partial finalization requires AcceptedReportBinding "
                    "from Output Gate acceptance"
                )
            if accepted.case_id != case_id:
                raise DomainFinalizationError(
                    "accepted binding case_id does not match finalization case_id"
                )
            if accepted.terminal_status is not terminal_status:
                raise DomainFinalizationError(
                    "accepted binding terminal_status does not match finalization"
                )
            accepted.assert_matches(accepted.report)
            report = accepted.report
            if report.case_id != case_id:
                raise DomainFinalizationError(
                    "report case_id does not match finalization case_id"
                )
            if report.execution_status is not terminal_status:
                raise DomainFinalizationError(
                    "report.execution_status must match terminal_status"
                )
        elif accepted is not None:
            raise DomainFinalizationError(
                f"{terminal_status.value} finalization must not attach an "
                "AcceptedReportBinding (failure is Case-status-only)"
            )

        if report is not None:
            report_payload = report.model_dump(mode="json")
            _forbidden_payload_keys(report_payload)
            finalization_fingerprint = payload_fingerprint(
                {
                    "case_id": case_id,
                    "terminal_status": terminal_status.value,
                    "report": report_payload,
                }
            )
        else:
            finalization_fingerprint = payload_fingerprint(
                {
                    "case_id": case_id,
                    "terminal_status": terminal_status.value,
                    "report": None,
                }
            )

        if self._txn_conn is not None:
            raise RuntimeError(
                "overlapping domain finalization is not supported "
                "(single-worker E0 transaction ownership)"
            )

        with self._db.transaction() as conn:
            self._txn_conn = conn
            try:
                self._fault("before_finalize")
                current = self._get_case_conn(conn, case_id)
                if current is None:
                    raise DomainFinalizationError(f"case {case_id} is not admitted")

                existing_report = self._get_report_conn(conn, case_id)
                if current.status in TERMINAL_CASE_STATUSES:
                    return self._reuse_finalization(
                        conn,
                        current=current,
                        terminal_status=terminal_status,
                        report=report,
                        finalization_fingerprint=finalization_fingerprint,
                        existing_report=existing_report,
                    )

                try:
                    validate_case_transition(current.status, terminal_status)
                except ValueError as exc:
                    raise DomainFinalizationError(str(exc)) from exc

                if report is not None:
                    claim_map = claims_by_id or {}
                    relation_map = relations_by_id or {}
                    self._fault("before_report_write")
                    operation = _finalize_operation_identity(
                        case_id=case_id,
                        finalization_fingerprint=finalization_fingerprint,
                    )
                    self.ensure_operation(operation)
                    # Persist any graph-session relations cited by accepted Claims
                    # that were not already written by a tool registration group.
                    self._persist_cited_relations(
                        case_id=case_id,
                        report=report,
                        claims_by_id=claim_map,
                        relations_by_id=relation_map,
                        operation=operation,
                    )
                    self._validate_report_prerequisites(
                        conn,
                        case_id=case_id,
                        report=report,
                        claims_by_id=claim_map,
                    )
                    # Persist accepted Claims before the Report so reopen can
                    # resolve report.claims without the in-memory graph store.
                    for claim_id in report.claims:
                        claim = claim_map[claim_id]
                        self.put_artifact(
                            kind=_KIND_CLAIM,
                            artifact_id=claim.claim_id,
                            operation=operation,
                            model=claim,
                        )
                    self.put_artifact(
                        kind=_KIND_REPORT,
                        artifact_id=report.report_id,
                        operation=operation,
                        model=report,
                    )
                    self._fault("after_report_write")
                else:
                    # Failure/cancelled: durable finalize operation without report body.
                    operation = _finalize_operation_identity(
                        case_id=case_id,
                        finalization_fingerprint=finalization_fingerprint,
                    )
                    self.ensure_operation(operation)

                self._fault("before_status_update")
                updated = current.transition_to(terminal_status)
                self._update_case_conn(conn, updated)
                self._fault("before_finalize_commit")
                return FinalizationResult(
                    case=updated,
                    terminal_status=terminal_status,
                    report=report,
                    finalization_fingerprint=finalization_fingerprint,
                    reused=False,
                )
            finally:
                self._txn_conn = None

    def get_report(self, case_id: str) -> InvestigationReport | None:
        return self._get_report_conn(self._db.conn, case_id)

    def get_claim(self, claim_id: str) -> Claim | None:
        return self.get_artifact(_KIND_CLAIM, claim_id, Claim)

    def list_claims_for_report(
        self, case_id: str, report: InvestigationReport
    ) -> list[Claim]:
        """Load durable Claims cited by an accepted Report, in report order."""
        claims: list[Claim] = []
        for claim_id in report.claims:
            claim = self.get_claim(claim_id)
            if claim is None:
                raise DomainFinalizationError(
                    f"accepted report {report.report_id} cites missing durable "
                    f"claim {claim_id}"
                )
            claims.append(claim)
        return claims

    def list_cases(
        self, *, limit: int = 50
    ) -> list[dict[str, str]]:
        """Recent Case summaries for operator reopen (read-only).

        Newest first by created_at; case_id DESC breaks ties deterministically.
        Callers choose the limit — this does not delete or truncate stored Cases.
        """
        capped = max(1, min(int(limit), 200))
        rows = self._db.conn.execute(
            """
            SELECT case_id, status, question, created_at, vms_instance_id
            FROM nx_cases
            ORDER BY created_at DESC, case_id DESC
            LIMIT ?
            """,
            (capped,),
        ).fetchall()
        return [
            {
                "case_id": row["case_id"],
                "status": row["status"],
                "question": row["question"],
                "created_at": row["created_at"],
                "vms_instance_id": row["vms_instance_id"],
            }
            for row in rows
        ]

    def _validate_admission_case(self, case: InvestigationCase) -> None:
        """Pure intake/scope checks before durable admission (not live VMS auth)."""
        if not case.case_id.strip() or not case.question.strip():
            raise CaseLifecycleError("admission requires non-empty case_id and question")
        if not case.policy_ref.strip():
            raise CaseLifecycleError("admission requires a policy_ref")
        if not case.effective_scope.camera_ids:
            raise CaseLifecycleError("admission requires an explicit effective camera scope")
        if (
            case.effective_scope.vms_instance_id
            != case.requested_scope.vms_instance_id
        ):
            raise CaseLifecycleError(
                "effective scope must keep the requested VMS instance"
            )
        unauthorized = [
            camera_id
            for camera_id in case.effective_scope.camera_ids
            if camera_id not in case.requested_scope.camera_ids
        ]
        if unauthorized:
            raise CaseLifecycleError(
                "effective scope cameras must be contained in requested scope"
            )

    def _get_case_conn(
        self, conn: sqlite3.Connection, case_id: str
    ) -> InvestigationCase | None:
        row = conn.execute(
            "SELECT payload_json FROM nx_cases WHERE case_id = ?",
            (case_id,),
        ).fetchone()
        if row is None:
            return None
        return InvestigationCase.model_validate_json(row["payload_json"])

    def _insert_case_conn(
        self,
        conn: sqlite3.Connection,
        case: InvestigationCase,
        *,
        fingerprint: str,
    ) -> None:
        conn.execute(
            """
            INSERT INTO nx_cases(
                case_id, status, question, policy_ref, created_at,
                vms_instance_id, payload_json, payload_fingerprint, schema_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (
                case.case_id,
                case.status.value,
                case.question,
                case.policy_ref,
                case.created_at.isoformat(),
                case.effective_scope.vms_instance_id,
                _dump(case),
                fingerprint,
            ),
        )

    def _update_case_conn(
        self, conn: sqlite3.Connection, case: InvestigationCase
    ) -> None:
        payload = case.model_dump(mode="json")
        _forbidden_payload_keys(payload)
        fingerprint = payload_fingerprint(payload)
        conn.execute(
            """
            UPDATE nx_cases SET
                status = ?,
                question = ?,
                policy_ref = ?,
                created_at = ?,
                vms_instance_id = ?,
                payload_json = ?,
                payload_fingerprint = ?
            WHERE case_id = ?
            """,
            (
                case.status.value,
                case.question,
                case.policy_ref,
                case.created_at.isoformat(),
                case.effective_scope.vms_instance_id,
                _dump(case),
                fingerprint,
                case.case_id,
            ),
        )

    def _get_report_conn(
        self, conn: sqlite3.Connection, case_id: str
    ) -> InvestigationReport | None:
        # Authoritative report is the one committed under the Case finalize operation.
        operation_id = f"op:{case_id}:{_FINALIZE_PLAN_STEP}"
        row = conn.execute(
            """
            SELECT payload_json FROM nx_artifacts
            WHERE case_id = ? AND artifact_kind = ? AND operation_id = ?
            """,
            (case_id, _KIND_REPORT, operation_id),
        ).fetchone()
        if row is None:
            return None
        return InvestigationReport.model_validate_json(row["payload_json"])

    def _reuse_finalization(
        self,
        conn: sqlite3.Connection,
        *,
        current: InvestigationCase,
        terminal_status: CaseStatus,
        report: InvestigationReport | None,
        finalization_fingerprint: str,
        existing_report: InvestigationReport | None,
    ) -> FinalizationResult:
        if current.status is not terminal_status:
            raise IdempotencyConflictError(
                f"case {current.case_id} already terminal as {current.status.value}; "
                f"cannot finalize as {terminal_status.value}"
            )
        operation = _finalize_operation_identity(
            case_id=current.case_id,
            finalization_fingerprint=finalization_fingerprint,
        )
        op_row = conn.execute(
            "SELECT content_fingerprint FROM nx_operations WHERE operation_id = ?",
            (operation.operation_id,),
        ).fetchone()
        if op_row is None:
            raise IdempotencyConflictError(
                f"terminal case {current.case_id} lacks finalize operation identity"
            )
        if op_row["content_fingerprint"] != finalization_fingerprint:
            raise IdempotencyConflictError(
                f"case {current.case_id} already finalized with different content"
            )
        if report is not None:
            if existing_report is None:
                raise IdempotencyConflictError(
                    f"case {current.case_id} terminal without stored report"
                )
            if existing_report.report_id != report.report_id:
                raise IdempotencyConflictError(
                    f"case {current.case_id} already has authoritative report "
                    f"{existing_report.report_id}"
                )
            existing_fp = payload_fingerprint(existing_report.model_dump(mode="json"))
            incoming_fp = payload_fingerprint(report.model_dump(mode="json"))
            if existing_fp != incoming_fp:
                raise IdempotencyConflictError(
                    f"case {current.case_id} authoritative report content conflict"
                )
        elif existing_report is not None:
            raise IdempotencyConflictError(
                f"case {current.case_id} already has an authoritative report"
            )
        return FinalizationResult(
            case=current,
            terminal_status=current.status,
            report=existing_report,
            finalization_fingerprint=finalization_fingerprint,
            reused=True,
        )

    def _persist_cited_relations(
        self,
        *,
        case_id: str,
        report: InvestigationReport,
        claims_by_id: dict[str, Claim],
        relations_by_id: dict[str, TemporalRelation],
        operation: OperationIdentity,
    ) -> None:
        """Persist graph-session temporal relations cited by accepted Claims.

        Some relations (for example deterministic BEFORE from evaluate_temporal)
        exist in the in-memory store but were not written by a tool registration
        group. Finalization must make them durable before Claim persistence.
        """
        conn = self._txn_conn
        if conn is None:
            raise RuntimeError("cited relation persistence requires an open transaction")
        needed: list[str] = []
        for claim_id in report.claims:
            claim = claims_by_id.get(claim_id)
            if claim is None:
                continue
            needed.extend(claim.relation_ids)
        for relation_id in dict.fromkeys(needed):
            row = conn.execute(
                """
                SELECT 1 FROM nx_artifacts
                WHERE case_id = ? AND artifact_kind = ? AND artifact_id = ?
                """,
                (case_id, _KIND_TEMPORAL_RELATION, relation_id),
            ).fetchone()
            if row is not None:
                continue
            relation = relations_by_id.get(relation_id)
            if relation is None:
                raise DomainFinalizationError(
                    f"accepted claim cites temporal relation {relation_id} that is "
                    "neither durable nor provided for finalization"
                )
            self.put_artifact(
                kind=_KIND_TEMPORAL_RELATION,
                artifact_id=relation.relation_id,
                operation=operation,
                model=relation,
            )

    def _validate_report_prerequisites(
        self,
        conn: sqlite3.Connection,
        *,
        case_id: str,
        report: InvestigationReport,
        claims_by_id: dict[str, Claim],
    ) -> None:
        if report.execution_status is CaseStatus.COMPLETED and not report.claims:
            raise DomainFinalizationError(
                "completed report requires evidence-grounded claims"
            )
        if (
            report.execution_status is CaseStatus.PARTIAL
            and not report.claims
            and not report.evidence_index
        ):
            raise DomainFinalizationError(
                "partial report requires evidence-grounded claims or evidence_index"
            )

        for evidence_id in report.evidence_index:
            row = conn.execute(
                """
                SELECT 1 FROM nx_artifacts
                WHERE case_id = ? AND artifact_kind = ? AND artifact_id = ?
                """,
                (case_id, _KIND_EVIDENCE, evidence_id),
            ).fetchone()
            if row is None:
                raise DomainFinalizationError(
                    f"report cites unpersisted evidence: {evidence_id}"
                )

        for claim_id in report.claims:
            claim = claims_by_id.get(claim_id)
            if claim is None:
                raise DomainFinalizationError(
                    f"report cites claim not provided for finalization: {claim_id}"
                )
            if claim.claim_id != claim_id:
                raise DomainFinalizationError(
                    f"claim map key {claim_id} does not match claim.claim_id "
                    f"{claim.claim_id}"
                )
            for evidence_id in claim.evidence_ids:
                row = conn.execute(
                    """
                    SELECT 1 FROM nx_artifacts
                    WHERE case_id = ? AND artifact_kind = ? AND artifact_id = ?
                    """,
                    (case_id, _KIND_EVIDENCE, evidence_id),
                ).fetchone()
                if row is None:
                    raise DomainFinalizationError(
                        f"claim {claim_id} cites unpersisted evidence: {evidence_id}"
                    )
            for receipt_id in claim.search_receipt_ids:
                row = conn.execute(
                    """
                    SELECT 1 FROM nx_artifacts
                    WHERE case_id = ? AND artifact_kind = ? AND artifact_id = ?
                    """,
                    (case_id, _KIND_RECEIPT, receipt_id),
                ).fetchone()
                if row is None:
                    raise DomainFinalizationError(
                        f"claim {claim_id} cites unpersisted search receipt: {receipt_id}"
                    )
            for relation_id in claim.relation_ids:
                row = conn.execute(
                    """
                    SELECT 1 FROM nx_artifacts
                    WHERE case_id = ? AND artifact_kind = ? AND artifact_id = ?
                    """,
                    (case_id, _KIND_TEMPORAL_RELATION, relation_id),
                ).fetchone()
                if row is None:
                    raise DomainFinalizationError(
                        f"claim {claim_id} cites unpersisted temporal relation: "
                        f"{relation_id}"
                    )
            # Producer "supported" labels are never treated as automatic authority.
            if claim.validation_status.value == "supported" and not claim.evidence_ids:
                raise DomainFinalizationError(
                    f"claim {claim_id}: syntactically supported claim is insufficient "
                    "without persisted evidence citations"
                )

    def ensure_operation(self, identity: OperationIdentity) -> OperationIdentity:
        if self._txn_conn is not None:
            self._ensure_operation_conn(self._txn_conn, identity)
            return identity
        with self._db.transaction() as conn:
            self._ensure_operation_conn(conn, identity)
        return identity

    def put_artifact(
        self,
        *,
        kind: str,
        artifact_id: str,
        operation: OperationIdentity,
        model: BaseModel,
        native_record_id: str | None = None,
        vms_instance_id: str | None = None,
    ) -> BaseModel:
        """Idempotent artifact write keyed by (artifact_kind, artifact_id).

        Multiple observation/evidence artifacts may share one operation_id
        (schema v3). Replaying the same id with different content conflicts.
        """
        self.ensure_operation(operation)
        payload = model.model_dump(mode="json")
        _forbidden_payload_keys(payload)
        fingerprint = payload_fingerprint(payload)

        def _write(conn: sqlite3.Connection) -> BaseModel:
            by_op_id = conn.execute(
                """
                SELECT artifact_id, payload_json, payload_fingerprint
                FROM nx_artifacts
                WHERE operation_id = ? AND artifact_kind = ? AND artifact_id = ?
                """,
                (operation.operation_id, kind, artifact_id),
            ).fetchone()
            if by_op_id is not None:
                if by_op_id["payload_fingerprint"] != fingerprint:
                    raise IdempotencyConflictError(
                        f"operation {operation.operation_id} already wrote {kind} "
                        f"id {artifact_id} with different content"
                    )
                return type(model).model_validate_json(by_op_id["payload_json"])

            by_id = conn.execute(
                """
                SELECT operation_id, payload_fingerprint, payload_json
                FROM nx_artifacts
                WHERE artifact_kind = ? AND artifact_id = ?
                """,
                (kind, artifact_id),
            ).fetchone()
            if by_id is not None:
                if by_id["operation_id"] != operation.operation_id:
                    raise IdempotencyConflictError(
                        f"{kind} id {artifact_id} already exists under another operation"
                    )
                if by_id["payload_fingerprint"] != fingerprint:
                    raise IdempotencyConflictError(
                        f"{kind} id {artifact_id} already exists with different content"
                    )
                return type(model).model_validate_json(by_id["payload_json"])

            # After a registration group is committed, refuse new artifact ids for
            # that operation so the durable group cannot silently grow.
            committed = conn.execute(
                """
                SELECT 1 FROM nx_registration_groups WHERE operation_id = ?
                """,
                (operation.operation_id,),
            ).fetchone()
            if committed is not None:
                raise IdempotencyConflictError(
                    f"operation {operation.operation_id} already has a committed "
                    f"registration group; refusing additional {kind} artifact "
                    f"{artifact_id}"
                )

            conn.execute(
                """
                INSERT INTO nx_artifacts(
                    artifact_kind, artifact_id, operation_id, case_id,
                    payload_json, payload_fingerprint, native_record_id,
                    vms_instance_id, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    kind,
                    artifact_id,
                    operation.operation_id,
                    operation.case_id,
                    _dump(model),
                    fingerprint,
                    native_record_id,
                    vms_instance_id,
                    _now(),
                ),
            )
            return model

        if self._txn_conn is not None:
            return _write(self._txn_conn)
        with self._db.transaction() as conn:
            return _write(conn)

    def get_artifact(self, kind: str, artifact_id: str, model_type: type[T]) -> T | None:
        row = self._db.conn.execute(
            """
            SELECT payload_json FROM nx_artifacts
            WHERE artifact_kind = ? AND artifact_id = ?
            """,
            (kind, artifact_id),
        ).fetchone()
        if row is None:
            return None
        return model_type.model_validate_json(row["payload_json"])

    def register_artifact_group(
        self,
        *,
        operation: OperationIdentity,
        receipt: SearchExecutionReceipt,
        observation: Observation | None = None,
        evidence: EvidenceRef | None = None,
        observations: tuple[Observation, ...] | list[Observation] | None = None,
        evidences: tuple[EvidenceRef, ...] | list[EvidenceRef] | None = None,
        relations: tuple[TemporalRelation, ...] | list[TemporalRelation] | None = None,
        media_evidences: tuple[EvidenceRef, ...] | list[EvidenceRef] | None = None,
    ) -> RegistrationGroupResult:
        """Atomically commit one logical tool-result registration group.

        Observation/EvidenceRef/SearchExecutionReceipt (as applicable), optional
        derived TemporalRelation members, optional media EvidenceRef members, and
        the registration-group completeness marker share one SQLite transaction.
        Budget reservations are intentionally outside this transaction.

        Prefer ``observations`` / ``evidences`` for multi-event tool results.
        The singular ``observation`` / ``evidence`` kwargs remain supported for
        single-event callers and are mutually exclusive with the sequence form.

        Relations may accompany a receipt without new observations when they cite
        already-persisted source observations. Media evidences (SNAPSHOT/CLIP)
        may accompany a receipt without new observations when they cite an
        already-persisted source EvidenceRef. Missing sources fail closed.
        """
        if media_evidences and (observations or evidences or observation or evidence):
            raise ValueError(
                "media_evidences cannot be combined with observation/evidence pairs"
            )
        if media_evidences:
            obs_list, ev_list = (), tuple(media_evidences)
            for item in ev_list:
                if item.evidence_kind not in {EvidenceKind.SNAPSHOT, EvidenceKind.CLIP}:
                    raise ValueError(
                        "media_evidences must use SNAPSHOT or CLIP evidence kinds"
                    )
                if not item.content_sha256 or not item.content_type:
                    raise ValueError(
                        "media evidence requires content_sha256 and content_type"
                    )
                source_ev_id = item.retrieval_spec.get("source_evidence_id")
                if not isinstance(source_ev_id, str) or not source_ev_id:
                    raise ValueError(
                        "media evidence requires retrieval_spec.source_evidence_id"
                    )
                if self.get_evidence(source_ev_id) is None:
                    raise ValueError(
                        "media evidence source EvidenceRef is not durable: "
                        f"{source_ev_id}"
                    )
        else:
            obs_list, ev_list = _normalize_artifact_pairs(
                observation=observation,
                evidence=evidence,
                observations=observations,
                evidences=evidences,
            )
        relation_list = tuple(relations or ())

        member_kinds = [_KIND_RECEIPT]
        if obs_list:
            member_kinds.extend([_KIND_OBSERVATION, _KIND_EVIDENCE])
        elif ev_list:
            # Media-only group: evidence_ref members without new observations.
            member_kinds.append(_KIND_EVIDENCE)
        if relation_list:
            member_kinds.append(_KIND_TEMPORAL_RELATION)
        for item in obs_list:
            _forbidden_payload_keys(item.model_dump(mode="json"))
        for item in ev_list:
            _forbidden_payload_keys(item.model_dump(mode="json"))
        for item in relation_list:
            _forbidden_payload_keys(item.model_dump(mode="json"))
        _forbidden_payload_keys(receipt.model_dump(mode="json"))

        new_obs_ids = {item.observation_id for item in obs_list}
        seen_relation_ids: set[str] = set()
        for rel in relation_list:
            if rel.relation_id in seen_relation_ids:
                raise ValueError(
                    "duplicate TemporalRelation id in registration group: "
                    f"{rel.relation_id}"
                )
            seen_relation_ids.add(rel.relation_id)
            for source_obs_id in (rel.left_observation_id, rel.right_observation_id):
                if source_obs_id in new_obs_ids:
                    continue
                if self.get_observation(source_obs_id) is None:
                    raise ValueError(
                        "TemporalRelation source observation is not durable: "
                        f"{source_obs_id} (relation {rel.relation_id})"
                    )

        group_fingerprint = _artifact_group_fingerprint(
            operation_id=operation.operation_id,
            receipt=receipt,
            observations=obs_list,
            evidences=ev_list,
            relations=relation_list,
        )
        member_kinds_json = json.dumps(member_kinds, separators=(",", ":"))

        # Single-worker E0 assumption: one registration transaction at a time per
        # repository instance. Nested/overlapping groups are not supported.
        if self._txn_conn is not None:
            raise RuntimeError(
                "overlapping domain registration is not supported "
                "(single-worker E0 transaction ownership)"
            )

        with self._db.transaction() as conn:
            self._txn_conn = conn
            try:
                self._fault("before_first_write")
                self.ensure_operation(operation)
                self._fault("after_operation")

                existing_group = conn.execute(
                    """
                    SELECT * FROM nx_registration_groups WHERE operation_id = ?
                    """,
                    (operation.operation_id,),
                ).fetchone()
                if existing_group is not None:
                    if existing_group["group_fingerprint"] != group_fingerprint:
                        raise IdempotencyConflictError(
                            f"operation {operation.operation_id} already has a committed "
                            "registration group with different content"
                        )
                    return self._load_group_result(
                        conn,
                        operation_id=operation.operation_id,
                        reused=True,
                    )

                # Artifacts without a committed group are incomplete and must not
                # be treated as a successful registration.
                orphan_count = conn.execute(
                    """
                    SELECT COUNT(*) AS c FROM nx_artifacts WHERE operation_id = ?
                    """,
                    (operation.operation_id,),
                ).fetchone()["c"]
                if orphan_count:
                    raise IdempotencyConflictError(
                        f"operation {operation.operation_id} has incomplete artifacts "
                        "without a committed registration group"
                    )

                # Public put_* methods participate in this outer transaction via
                # ``_txn_conn``, so monkeypatched put_evidence (EB1B F1) still
                # fails closed inside the same atomic group.
                self.put_search_receipt(receipt, operation=operation)
                self._fault("after_receipt")

                if obs_list:
                    for observation_item, evidence_item in zip(
                        obs_list, ev_list, strict=True
                    ):
                        self.put_observation(observation_item, operation=operation)
                        self._fault("after_observation")
                        self.put_evidence(evidence_item, operation=operation)
                        self._fault("after_evidence")
                else:
                    for evidence_item in ev_list:
                        self.put_evidence(evidence_item, operation=operation)
                        self._fault("after_evidence")

                for relation_item in relation_list:
                    self.put_temporal_relation(relation_item, operation=operation)
                    self._fault("after_temporal_relation")

                self._fault("before_commit")
                conn.execute(
                    """
                    INSERT INTO nx_registration_groups(
                        operation_id, case_id, group_fingerprint, member_kinds,
                        receipt_id, observation_id, evidence_id, status, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 'committed', ?)
                    """,
                    (
                        operation.operation_id,
                        operation.case_id,
                        group_fingerprint,
                        member_kinds_json,
                        receipt.search_execution_id,
                        None if not obs_list else obs_list[0].observation_id,
                        None if not ev_list else ev_list[0].evidence_id,
                        _now(),
                    ),
                )
                return self._load_group_result(
                    conn,
                    operation_id=operation.operation_id,
                    reused=False,
                )
            finally:
                self._txn_conn = None

    def get_registration_group(
        self, operation_id: str
    ) -> RegistrationGroupResult | None:
        row = self._db.conn.execute(
            "SELECT operation_id FROM nx_registration_groups WHERE operation_id = ?",
            (operation_id,),
        ).fetchone()
        if row is None:
            return None
        return self._load_group_result(
            self._db.conn, operation_id=operation_id, reused=True
        )

    def list_registration_group_operation_ids(self, case_id: str) -> list[str]:
        """Read-only listing of registration-group operation ids for one Case."""
        rows = self._db.conn.execute(
            """
            SELECT operation_id FROM nx_registration_groups
            WHERE case_id = ?
            ORDER BY created_at ASC, operation_id ASC
            """,
            (case_id,),
        ).fetchall()
        return [row["operation_id"] for row in rows]

    def list_registration_groups(self, case_id: str) -> list[RegistrationGroupResult]:
        """Read-only listing of loadable committed registration groups for one Case.

        Incomplete groups are skipped so read-only recovery inspection can
        continue and report integrity failures via ``get_registration_group``.
        """
        results: list[RegistrationGroupResult] = []
        for operation_id in self.list_registration_group_operation_ids(case_id):
            try:
                group = self.get_registration_group(operation_id)
            except Exception:  # noqa: BLE001 — incomplete group surfaced by caller
                continue
            if group is not None:
                results.append(group)
        return results

    def list_artifact_refs(
        self, case_id: str, *, kind: str | None = None
    ) -> list[tuple[str, str]]:
        """Read-only (artifact_kind, artifact_id) pairs for one Case."""
        if kind is None:
            rows = self._db.conn.execute(
                """
                SELECT artifact_kind, artifact_id FROM nx_artifacts
                WHERE case_id = ?
                ORDER BY artifact_kind ASC, artifact_id ASC
                """,
                (case_id,),
            ).fetchall()
        else:
            rows = self._db.conn.execute(
                """
                SELECT artifact_kind, artifact_id FROM nx_artifacts
                WHERE case_id = ? AND artifact_kind = ?
                ORDER BY artifact_id ASC
                """,
                (case_id, kind),
            ).fetchall()
        return [(row["artifact_kind"], row["artifact_id"]) for row in rows]

    def list_operation_ids(self, case_id: str) -> list[str]:
        """Read-only operation identities recorded for one Case."""
        rows = self._db.conn.execute(
            """
            SELECT operation_id FROM nx_operations
            WHERE case_id = ?
            ORDER BY created_at ASC, operation_id ASC
            """,
            (case_id,),
        ).fetchall()
        return [row["operation_id"] for row in rows]

    def _ensure_operation_conn(
        self, conn: sqlite3.Connection, identity: OperationIdentity
    ) -> None:
        existing = conn.execute(
            "SELECT content_fingerprint FROM nx_operations WHERE operation_id = ?",
            (identity.operation_id,),
        ).fetchone()
        if existing is not None:
            if existing["content_fingerprint"] != identity.content_fingerprint:
                raise IdempotencyConflictError(
                    f"operation_id {identity.operation_id} fingerprint conflict"
                )
            return
        conn.execute(
            """
            INSERT INTO nx_operations(
                operation_id, case_id, plan_step_id, tool_name,
                content_fingerprint, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                identity.operation_id,
                identity.case_id,
                identity.plan_step_id,
                identity.tool_name,
                identity.content_fingerprint,
                _now(),
            ),
        )

    def _insert_artifact_conn(
        self,
        conn: sqlite3.Connection,
        *,
        kind: str,
        artifact_id: str,
        operation: OperationIdentity,
        model: BaseModel,
        native_record_id: str | None,
        vms_instance_id: str | None,
    ) -> None:
        payload = model.model_dump(mode="json")
        _forbidden_payload_keys(payload)
        fingerprint = payload_fingerprint(payload)
        by_id = conn.execute(
            """
            SELECT operation_id, payload_fingerprint FROM nx_artifacts
            WHERE artifact_kind = ? AND artifact_id = ?
            """,
            (kind, artifact_id),
        ).fetchone()
        if by_id is not None:
            raise IdempotencyConflictError(
                f"{kind} id {artifact_id} already exists outside this registration group"
            )
        conn.execute(
            """
            INSERT INTO nx_artifacts(
                artifact_kind, artifact_id, operation_id, case_id,
                payload_json, payload_fingerprint, native_record_id,
                vms_instance_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                kind,
                artifact_id,
                operation.operation_id,
                operation.case_id,
                _dump(model),
                fingerprint,
                native_record_id,
                vms_instance_id,
                _now(),
            ),
        )

    def _load_group_result(
        self,
        conn: sqlite3.Connection,
        *,
        operation_id: str,
        reused: bool,
    ) -> RegistrationGroupResult:
        group = conn.execute(
            "SELECT * FROM nx_registration_groups WHERE operation_id = ?",
            (operation_id,),
        ).fetchone()
        if group is None or group["status"] != "committed":
            raise IdempotencyConflictError(
                f"operation {operation_id} registration group is not committed"
            )
        receipt_row = conn.execute(
            """
            SELECT payload_json FROM nx_artifacts
            WHERE operation_id = ? AND artifact_kind = ?
            """,
            (operation_id, _KIND_RECEIPT),
        ).fetchone()
        if receipt_row is None:
            raise IdempotencyConflictError(
                f"committed group {operation_id} missing search receipt"
            )
        receipt = SearchExecutionReceipt.model_validate_json(receipt_row["payload_json"])
        if receipt.search_execution_id != group["receipt_id"]:
            raise IdempotencyConflictError(
                f"committed group {operation_id} receipt id mismatch"
            )

        # created_at preserves registration write order for multi-event groups.
        obs_rows = conn.execute(
            """
            SELECT artifact_id, payload_json FROM nx_artifacts
            WHERE operation_id = ? AND artifact_kind = ?
            ORDER BY created_at ASC, artifact_id ASC
            """,
            (operation_id, _KIND_OBSERVATION),
        ).fetchall()
        ev_rows = conn.execute(
            """
            SELECT artifact_id, payload_json FROM nx_artifacts
            WHERE operation_id = ? AND artifact_kind = ?
            ORDER BY created_at ASC, artifact_id ASC
            """,
            (operation_id, _KIND_EVIDENCE),
        ).fetchall()
        observations_loaded = tuple(
            Observation.model_validate_json(row["payload_json"]) for row in obs_rows
        )
        evidences_loaded = tuple(
            EvidenceRef.model_validate_json(row["payload_json"]) for row in ev_rows
        )
        rel_rows = conn.execute(
            """
            SELECT artifact_id, payload_json FROM nx_artifacts
            WHERE operation_id = ? AND artifact_kind = ?
            ORDER BY created_at ASC, artifact_id ASC
            """,
            (operation_id, _KIND_TEMPORAL_RELATION),
        ).fetchall()
        relations_loaded = tuple(
            TemporalRelation.model_validate_json(row["payload_json"]) for row in rel_rows
        )
        # Paired search groups require equal counts. Media-only groups may have
        # evidences without observations. Observation-only groups are invalid.
        if observations_loaded and len(observations_loaded) != len(evidences_loaded):
            raise IdempotencyConflictError(
                f"committed group {operation_id} has inconsistent observation/evidence counts"
            )
        if observations_loaded and not evidences_loaded:
            raise IdempotencyConflictError(
                f"committed group {operation_id} has observations without evidence"
            )
        if (group["observation_id"] is None) != (len(observations_loaded) == 0):
            raise IdempotencyConflictError(
                f"committed group {operation_id} observation membership mismatch"
            )
        if (group["evidence_id"] is None) != (len(evidences_loaded) == 0):
            raise IdempotencyConflictError(
                f"committed group {operation_id} evidence membership mismatch"
            )
        if group["observation_id"] is not None:
            obs_ids = {item.observation_id for item in observations_loaded}
            if group["observation_id"] not in obs_ids:
                raise IdempotencyConflictError(
                    f"committed group {operation_id} missing primary observation"
                )
        if group["evidence_id"] is not None:
            ev_ids = {item.evidence_id for item in evidences_loaded}
            if group["evidence_id"] not in ev_ids:
                raise IdempotencyConflictError(
                    f"committed group {operation_id} missing primary evidence"
                )

        if observations_loaded:
            ordered_obs, ordered_ev = _pair_observations_and_evidence(
                observations_loaded, evidences_loaded
            )
        else:
            # Receipt-only or media-only (evidence without new observations).
            ordered_obs, ordered_ev = (), evidences_loaded
        expected_fingerprint = _artifact_group_fingerprint(
            operation_id=operation_id,
            receipt=receipt,
            observations=ordered_obs,
            evidences=ordered_ev,
            relations=relations_loaded,
        )
        legacy_fingerprint = None
        if len(ordered_obs) <= 1 and not relations_loaded and not (
            not ordered_obs and ordered_ev
        ):
            legacy_fingerprint = _legacy_single_artifact_group_fingerprint(
                operation_id=operation_id,
                receipt=receipt,
                observation=ordered_obs[0] if ordered_obs else None,
                evidence=ordered_ev[0] if ordered_ev else None,
            )
        if group["group_fingerprint"] not in {
            expected_fingerprint,
            legacy_fingerprint,
        }:
            raise IdempotencyConflictError(
                f"committed group {operation_id} fingerprint does not match members"
            )
        return RegistrationGroupResult(
            operation_id=operation_id,
            group_fingerprint=group["group_fingerprint"],
            receipt=receipt,
            observations=ordered_obs,
            evidences=ordered_ev,
            relations=relations_loaded,
            reused=reused,
        )

    def put_observation(
        self,
        observation: Observation,
        *,
        operation: OperationIdentity,
    ) -> Observation:
        return self.put_artifact(  # type: ignore[return-value]
            kind=_KIND_OBSERVATION,
            artifact_id=observation.observation_id,
            operation=operation,
            model=observation,
            native_record_id=observation.source_locator.native_record_id,
            vms_instance_id=observation.source_locator.vms_instance_id,
        )

    def put_evidence(
        self,
        evidence: EvidenceRef,
        *,
        operation: OperationIdentity,
    ) -> EvidenceRef:
        return self.put_artifact(  # type: ignore[return-value]
            kind=_KIND_EVIDENCE,
            artifact_id=evidence.evidence_id,
            operation=operation,
            model=evidence,
            native_record_id=evidence.source_locator.native_record_id,
            vms_instance_id=evidence.source_locator.vms_instance_id,
        )

    def put_search_receipt(
        self,
        receipt: SearchExecutionReceipt,
        *,
        operation: OperationIdentity,
    ) -> SearchExecutionReceipt:
        return self.put_artifact(  # type: ignore[return-value]
            kind=_KIND_RECEIPT,
            artifact_id=receipt.search_execution_id,
            operation=operation,
            model=receipt,
            vms_instance_id=receipt.source_id,
        )

    def put_temporal_relation(
        self,
        relation: TemporalRelation,
        *,
        operation: OperationIdentity,
    ) -> TemporalRelation:
        return self.put_artifact(  # type: ignore[return-value]
            kind=_KIND_TEMPORAL_RELATION,
            artifact_id=relation.relation_id,
            operation=operation,
            model=relation,
        )

    def get_observation(self, observation_id: str) -> Observation | None:
        return self.get_artifact(_KIND_OBSERVATION, observation_id, Observation)

    def get_evidence(self, evidence_id: str) -> EvidenceRef | None:
        return self.get_artifact(_KIND_EVIDENCE, evidence_id, EvidenceRef)

    def get_search_receipt(self, receipt_id: str) -> SearchExecutionReceipt | None:
        return self.get_artifact(_KIND_RECEIPT, receipt_id, SearchExecutionReceipt)

    def get_temporal_relation(self, relation_id: str) -> TemporalRelation | None:
        return self.get_artifact(
            _KIND_TEMPORAL_RELATION, relation_id, TemporalRelation
        )
