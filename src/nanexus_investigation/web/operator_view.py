"""Deterministic OperatorView derived from accepted Report + durable Claims."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from nanexus_investigation.contracts.claim import Claim, InvestigationReport
from nanexus_investigation.contracts.enums import ClaimType, EvidenceKind
from nanexus_investigation.contracts.evidence import EvidenceRef, Observation
from nanexus_investigation.contracts.temporal import TemporalRelation
from nanexus_investigation.persistence.repository import InvestigationRepository

_TIMELINE_CAP = 12
_FINDING_CAP = 10
_NEAR_QUALIFICATION = (
    "Nearby-activity (NEAR) means temporal proximity only — not interaction, "
    "identity, ownership, or causation."
)
# Canonical MODEL_INTERPRETATION claim wording from draft_candidate_claims:
# "... interpreted retained snapshot {id} as: {observation_text}"
_VISUAL_CLAIM_OBS_MARKER = " as: "
_MEDIA_NONE_RETRIEVED = (
    "No snapshots were retrieved for this investigation."
)
_MEDIA_RETRIEVAL_FAILED = (
    "Snapshot retrieval was attempted, but no snapshot bytes were retained."
)
_ANTI_REPEAT_OPERATOR = (
    "Useful evidence was already obtained for this Case. Further execution "
    "stopped as PARTIAL to avoid repeating an earlier tool operation — not "
    "because prior detections, snapshots, or visual interpretations were discarded."
)


def build_operator_view(
    *,
    repo: InvestigationRepository,
    case_id: str,
    status_override: str | None = None,
    interrupted: bool = False,
) -> dict[str, Any]:
    """Assemble a concise operator presentation from durable artifacts only."""
    case = repo.get_case(case_id)
    if case is None:
        return {
            "case_id": case_id,
            "status": "unsupported",
            "error": "case not found",
        }

    if interrupted and case.status.value == "running":
        return {
            "case_id": case_id,
            "question": case.question,
            "status": "failed",
            "interrupted": True,
            "summary": (
                "Investigation was interrupted before durable finalization "
                "(for example, the Web process restarted). Results are incomplete; "
                "start a new investigation."
            ),
            "scope": _scope_dict(case),
            "findings": [],
            "findings_total": 0,
            "timeline": [],
            "timeline_full": [],
            "timeline_total": 0,
            "relations": [],
            "media": [],
            "limitations": [
                "Process interruption: no automatic recovery was performed."
            ],
            "unresolved_questions": [
                "Re-run the investigation to obtain a finalized report."
            ],
            "technical": {"case_status": case.status.value},
        }

    report = repo.get_report(case_id)
    status = status_override or case.status.value
    if report is None:
        return {
            "case_id": case_id,
            "question": case.question,
            "status": status if status != "running" else "failed",
            "summary": (
                "No accepted investigation report is available for this Case."
            ),
            "scope": _scope_dict(case),
            "findings": [],
            "findings_total": 0,
            "timeline": [],
            "timeline_full": [],
            "timeline_total": 0,
            "relations": [],
            "media": [],
            "limitations": ["Case has no durable InvestigationReport."],
            "unresolved_questions": [],
            "technical": {"case_status": case.status.value},
        }

    claims = repo.list_claims_for_report(case_id, report)
    observations = _load_observations(repo, report)
    relations = _load_relations(repo, report, claims)
    evidence_by_id = _load_evidence(repo, report, claims)

    findings_full = _select_findings(
        claims,
        observations=observations,
        evidence_by_id=evidence_by_id,
        relations=relations,
        cap=None,
        repo=repo,
    )
    findings = findings_full[:_FINDING_CAP]
    timeline_full = _select_timeline(observations, cap=None)
    timeline = timeline_full[:_TIMELINE_CAP]
    media = _select_media(claims, evidence_by_id, repo=repo)
    relation_rows = _format_relations(relations, observations)
    media_empty_message = None
    if not media:
        media_empty_message = _media_absence_message(
            repo=repo,
            claims=claims,
            evidence_by_id=evidence_by_id,
            limitations=list(report.limitations),
        )

    summary = _build_summary(
        status=report.execution_status.value,
        question=case.question,
        findings=findings,
        timeline=timeline_full,
        relations=relation_rows,
        media=media,
        limitations=list(report.limitations),
        unresolved=list(report.unresolved_questions),
        observation_count=len(observations),
        scope=_scope_dict(case),
    )

    return {
        "case_id": case_id,
        "question": case.question,
        "status": report.execution_status.value,
        "scope": _scope_dict(case),
        "scope_summary": report.scope_summary,
        "summary": summary,
        "findings": findings,
        "findings_full": findings_full,
        "findings_total": len(findings_full),
        "timeline": timeline,
        "timeline_full": timeline_full,
        "timeline_total": len(observations),
        "relations": relation_rows,
        "relation_qualification": _NEAR_QUALIFICATION if relation_rows else None,
        "media": media,
        "media_empty_message": media_empty_message,
        "limitations": [
            _operator_limitation_text(item) for item in report.limitations
        ],
        "unresolved_questions": [
            _operator_limitation_text(item) for item in report.unresolved_questions
        ],
        "technical": {
            "report_id": report.report_id,
            "claim_ids": list(report.claims),
            "evidence_index": list(report.evidence_index),
            "validation_summary": report.validation_summary,
            "case_status": case.status.value,
            "durable_limitations": list(report.limitations),
            "claim_statements": [
                {
                    "claim_id": claim.claim_id,
                    "claim_type": claim.claim_type.value,
                    "statement": claim.statement,
                    "producer": claim.producer,
                    "validation_status": claim.validation_status.value,
                    "evidence_ids": list(claim.evidence_ids),
                    "relation_ids": list(claim.relation_ids),
                }
                for claim in claims
            ],
        },
    }


def build_operator_view_from_runtime(
    *,
    case_id: str,
    question: str,
    status: str,
    report: InvestigationReport | None,
    claims: dict[str, Claim],
    observations: dict[str, Observation],
    relations: dict[str, TemporalRelation],
    evidence: dict[str, EvidenceRef],
    scope: dict[str, Any],
    error: str | None = None,
) -> dict[str, Any]:
    """In-process view immediately after a graph run (before/without reopen)."""
    if report is None:
        return {
            "case_id": case_id,
            "question": question,
            "status": status,
            "summary": error or "Investigation ended without an accepted report.",
            "scope": scope,
            "findings": [],
            "findings_total": 0,
            "timeline": [],
            "timeline_full": [],
            "timeline_total": 0,
            "relations": [],
            "media": [],
            "limitations": [error] if error else [],
            "unresolved_questions": [],
            "technical": {},
        }
    ordered_claims = [claims[cid] for cid in report.claims if cid in claims]
    obs_list = list(observations.values())
    rel_list = list(relations.values())
    findings_full = _select_findings(
        ordered_claims,
        observations=obs_list,
        evidence_by_id=evidence,
        relations=rel_list,
        cap=None,
        repo=None,
    )
    findings = findings_full[:_FINDING_CAP]
    timeline_full = _select_timeline(obs_list, cap=None)
    timeline = timeline_full[:_TIMELINE_CAP]
    media = _select_media(ordered_claims, evidence, repo=None)
    relation_rows = _format_relations(rel_list, obs_list)
    media_empty_message = None
    if not media:
        media_empty_message = _media_absence_message(
            repo=None,
            claims=ordered_claims,
            evidence_by_id=evidence,
            limitations=list(report.limitations),
        )
    return {
        "case_id": case_id,
        "question": question,
        "status": report.execution_status.value,
        "scope": scope,
        "scope_summary": report.scope_summary,
        "summary": _build_summary(
            status=report.execution_status.value,
            question=question,
            findings=findings,
            timeline=timeline_full,
            relations=relation_rows,
            media=media,
            limitations=list(report.limitations),
            unresolved=list(report.unresolved_questions),
            observation_count=len(observations),
            scope=scope,
        ),
        "findings": findings,
        "findings_full": findings_full,
        "findings_total": len(findings_full),
        "timeline": timeline,
        "timeline_full": timeline_full,
        "timeline_total": len(observations),
        "relations": relation_rows,
        "relation_qualification": _NEAR_QUALIFICATION if relation_rows else None,
        "media": media,
        "media_empty_message": media_empty_message,
        "limitations": [
            _operator_limitation_text(item) for item in report.limitations
        ],
        "unresolved_questions": [
            _operator_limitation_text(item) for item in report.unresolved_questions
        ],
        "technical": {
            "report_id": report.report_id,
            "claim_ids": list(report.claims),
            "evidence_index": list(report.evidence_index),
            "validation_summary": report.validation_summary,
            "durable_limitations": list(report.limitations),
            "claim_statements": [
                {
                    "claim_id": claim.claim_id,
                    "claim_type": claim.claim_type.value,
                    "statement": claim.statement,
                    "producer": claim.producer,
                    "validation_status": claim.validation_status.value,
                    "evidence_ids": list(claim.evidence_ids),
                    "relation_ids": list(claim.relation_ids),
                }
                for claim in ordered_claims
            ],
        },
    }


def _operator_limitation_text(item: str) -> str:
    """Operator-facing limitation wording; durable text remains in technical."""
    text = item.strip()
    lower = text.lower()
    if "timeline temporal relations are deterministic" in lower:
        return _NEAR_QUALIFICATION
    if "identical tool request already executed" in lower:
        return _ANTI_REPEAT_OPERATOR
    if lower.startswith("partial investigation coverage:"):
        text = text.replace(
            "partial investigation coverage:",
            "Coverage incomplete:",
            1,
        )
    if "search_events" in text:
        text = text.replace("search_events", "detection search")
    if lower.startswith("partial investigation stop:"):
        text = text.replace(
            "partial investigation stop:",
            "Investigation stopped early:",
            1,
        )
    return text


def _media_absence_message(
    *,
    repo: InvestigationRepository | None,
    claims: list[Claim],
    evidence_by_id: dict[str, EvidenceRef],
    limitations: list[str],
) -> str:
    """Explain empty Selected Visual Evidence without implying VMS has no images."""
    if any(
        evidence.evidence_kind is EvidenceKind.SNAPSHOT
        for evidence in evidence_by_id.values()
    ):
        # Snapshot EvidenceRefs exist but no displayable media row — unusual.
        return (
            "Snapshot evidence is registered for this Case, but no displayable "
            "retained media was selected. See Evidence Details."
        )
    attempted = False
    for lim in limitations:
        lower = lim.lower()
        if (
            "without acquiring the requested snapshot" in lower
            or "unavailable media" in lower
            or ("resolve_media" in lower and "error" in lower)
        ):
            attempted = True
            break
    if repo is not None:
        for claim in claims:
            for receipt_id in claim.search_receipt_ids:
                receipt = repo.get_search_receipt(receipt_id)
                if receipt is not None and receipt.tool_name == "resolve_media":
                    attempted = True
                    break
            if attempted:
                break
    if attempted:
        return _MEDIA_RETRIEVAL_FAILED
    return _MEDIA_NONE_RETRIEVED


def _scope_dict(case: Any) -> dict[str, Any]:
    bounds = case.effective_scope.time_bounds
    return {
        "vms": case.effective_scope.vms_instance_id,
        "cameras": list(case.effective_scope.camera_ids),
        "start_utc": bounds.start_utc.isoformat(),
        "end_utc": None if bounds.end_utc is None else bounds.end_utc.isoformat(),
    }


def _load_observations(
    repo: InvestigationRepository, report: InvestigationReport
) -> list[Observation]:
    obs: list[Observation] = []
    seen: set[str] = set()
    for entry in report.timeline:
        if entry.startswith("rel-"):
            continue
        if entry in seen:
            continue
        item = repo.get_observation(entry)
        if item is not None:
            seen.add(entry)
            obs.append(item)
    return obs


def _load_relations(
    repo: InvestigationRepository,
    report: InvestigationReport,
    claims: list[Claim],
) -> list[TemporalRelation]:
    ids: list[str] = []
    for entry in report.timeline:
        if entry.startswith("rel-"):
            ids.append(entry)
    for claim in claims:
        ids.extend(claim.relation_ids)
    out: list[TemporalRelation] = []
    seen: set[str] = set()
    for relation_id in ids:
        if relation_id in seen:
            continue
        rel = repo.get_temporal_relation(relation_id)
        if rel is not None:
            seen.add(relation_id)
            out.append(rel)
    return out


def _load_evidence(
    repo: InvestigationRepository,
    report: InvestigationReport,
    claims: list[Claim],
) -> dict[str, EvidenceRef]:
    ids = set(report.evidence_index)
    for claim in claims:
        ids.update(claim.evidence_ids)
    out: dict[str, EvidenceRef] = {}
    for evidence_id in ids:
        item = repo.get_evidence(evidence_id)
        if item is not None:
            out[evidence_id] = item
    return out


def _obs_by_native(observations: list[Observation]) -> dict[str, Observation]:
    return {item.source_locator.native_record_id: item for item in observations}


def _observation_for_evidence(
    evidence: EvidenceRef | None, observations: list[Observation]
) -> Observation | None:
    if evidence is None:
        return None
    return _obs_by_native(observations).get(evidence.source_locator.native_record_id)


def _fmt_time(value: datetime | None) -> str:
    if value is None:
        return "unknown time"
    return value.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _label_text(observation: Observation | None) -> str:
    if observation is None or not observation.labels:
        return "detection"
    return ", ".join(observation.labels)


def _select_findings(
    claims: list[Claim],
    *,
    observations: list[Observation],
    evidence_by_id: dict[str, EvidenceRef],
    relations: list[TemporalRelation],
    cap: int | None,
    repo: InvestigationRepository | None = None,
) -> list[dict[str, Any]]:
    ranked: list[Claim] = []
    # Prefer the longest visual interpretation per evidence set (drop garbled duplicates).
    visual_by_key: dict[str, Claim] = {}
    for claim in claims:
        if claim.claim_type is not ClaimType.MODEL_INTERPRETATION:
            continue
        key = ",".join(claim.evidence_ids) or claim.claim_id
        prior = visual_by_key.get(key)
        if prior is None or len(claim.statement) > len(prior.statement):
            visual_by_key[key] = claim
    ranked.extend(visual_by_key.values())
    for claim in claims:
        if claim.claim_type is ClaimType.COMPUTED_RELATION:
            ranked.append(claim)
    # Prefer event-metadata observations; media-byte claims are secondary.
    source_events: list[Claim] = []
    source_media: list[Claim] = []
    for claim in claims:
        if claim.claim_type is not ClaimType.SOURCE_OBSERVATION:
            continue
        evidence = (
            evidence_by_id.get(claim.evidence_ids[0]) if claim.evidence_ids else None
        )
        if evidence is not None and evidence.evidence_kind in {
            EvidenceKind.SNAPSHOT,
            EvidenceKind.CLIP,
        }:
            source_media.append(claim)
        else:
            source_events.append(claim)
    ranked.extend(source_events[:5])
    # Only surface media-byte claims when no visual interpretation exists.
    if not any(c.claim_type is ClaimType.MODEL_INTERPRETATION for c in claims):
        ranked.extend(source_media[:2])
    for claim in claims:
        if claim.claim_type is ClaimType.QUALIFIED_SEARCH_RESULT:
            ranked.append(claim)

    relation_by_id = {rel.relation_id: rel for rel in relations}
    seen: set[str] = set()
    findings: list[dict[str, Any]] = []
    for claim in ranked:
        if claim.claim_id in seen:
            continue
        seen.add(claim.claim_id)
        finding = _format_finding(
            claim,
            observations=observations,
            evidence_by_id=evidence_by_id,
            relation_by_id=relation_by_id,
            repo=repo,
        )
        # Avoid repeating the shared temporal non-causation disclaimer on every
        # NEAR finding card; it is surfaced once via relation_qualification.
        if claim.claim_type is ClaimType.COMPUTED_RELATION:
            finding["limitations"] = [
                item
                for item in finding["limitations"]
                if "proximity alone" not in item.lower()
                and "does not prove" not in item.lower()
                and "not interaction" not in item.lower()
            ]
        findings.append(finding)
        if cap is not None and len(findings) >= cap:
            break
        if cap is None and len(findings) >= 50:
            break
    return findings


def _format_finding(
    claim: Claim,
    *,
    observations: list[Observation],
    evidence_by_id: dict[str, EvidenceRef],
    relation_by_id: dict[str, TemporalRelation],
    repo: InvestigationRepository | None = None,
) -> dict[str, Any]:
    headline, detail = _operator_text_for_claim(
        claim,
        observations=observations,
        evidence_by_id=evidence_by_id,
        relation_by_id=relation_by_id,
        repo=repo,
    )
    return {
        "kind": claim.claim_type.value,
        "headline": headline,
        "detail": detail,
        "limitations": list(claim.limitations),
        "refs": {
            "claim_id": claim.claim_id,
            "evidence_ids": list(claim.evidence_ids),
            "relation_ids": list(claim.relation_ids),
            "search_receipt_ids": list(claim.search_receipt_ids),
            "producer": claim.producer,
            "validation_status": claim.validation_status.value,
            "canonical_statement": claim.statement,
        },
    }


def _operator_text_for_claim(
    claim: Claim,
    *,
    observations: list[Observation],
    evidence_by_id: dict[str, EvidenceRef],
    relation_by_id: dict[str, TemporalRelation],
    repo: InvestigationRepository | None = None,
) -> tuple[str, str]:
    """Human-readable headline/detail; canonical statement stays in refs."""
    if claim.claim_type is ClaimType.MODEL_INTERPRETATION:
        evidence = (
            evidence_by_id.get(claim.evidence_ids[0]) if claim.evidence_ids else None
        )
        obs = _observation_for_evidence(evidence, observations)
        where = ""
        if obs is not None:
            where = (
                f" for {_label_text(obs)} on {obs.camera_id} "
                f"at {_fmt_time(obs.start_utc)}"
            )
        elif evidence is not None:
            where = f" for camera {evidence.camera_id}"
        visual = _faithful_observation_text(claim, repo=repo)
        return (
            "Qualified local visual interpretation",
            f"Local VLM interpretation{where}: {visual}",
        )

    if claim.claim_type is ClaimType.COMPUTED_RELATION:
        rel = None
        if claim.relation_ids:
            rel = relation_by_id.get(claim.relation_ids[0])
        labels = {
            item.observation_id: _label_text(item) for item in observations
        }
        if rel is not None:
            left = labels.get(rel.left_observation_id, "an event")
            right = labels.get(rel.right_observation_id, "an event")
            gap = (
                "unknown gap"
                if rel.gap_seconds is None
                else f"{rel.gap_seconds:.0f}s apart"
            )
            # Shared non-causation disclaimer is shown once under Limitations /
            # relation_qualification — not repeated on every NEAR finding.
            return (
                "Nearby activity",
                (
                    f"{left} near {right} — {gap} "
                    f"(within {rel.window_seconds:.0f}s)."
                ),
            )
        return (
            "Nearby activity",
            "A nearby-activity (NEAR) relation was established between detections.",
        )

    if claim.claim_type is ClaimType.SOURCE_OBSERVATION:
        evidence = (
            evidence_by_id.get(claim.evidence_ids[0]) if claim.evidence_ids else None
        )
        obs = _observation_for_evidence(evidence, observations)
        if evidence is not None and evidence.evidence_kind in {
            EvidenceKind.SNAPSHOT,
            EvidenceKind.CLIP,
        }:
            kind = evidence.evidence_kind.value
            if obs is not None:
                return (
                    f"Retained {kind}",
                    (
                        f"{kind.title()} retained for {_label_text(obs)} on "
                        f"{obs.camera_id} at {_fmt_time(obs.start_utc)}. "
                        "Byte retention is not visual interpretation."
                    ),
                )
            camera = evidence.camera_id if evidence is not None else "authorized camera"
            return (
                f"Retained {kind}",
                f"{kind.title()} retained for camera {camera}. "
                "Byte retention is not visual interpretation.",
            )
        if obs is not None:
            label = _label_text(obs)
            return (
                f"{label.title()} on {obs.camera_id}",
                (
                    f"Authorized {label} detection on camera {obs.camera_id} "
                    f"at {_fmt_time(obs.start_utc)}."
                ),
            )
        return (
            "Authorized VMS detection",
            "Authorized detection metadata was registered for this case.",
        )

    if claim.claim_type is ClaimType.QUALIFIED_SEARCH_RESULT:
        return (
            "Qualified search coverage",
            "Search coverage for the authorized scope was recorded with "
            "explicit qualification (not an exhaustive real-world absence claim).",
        )

    return (
        claim.claim_type.value.replace("_", " ").title(),
        claim.statement,
    )


def _observation_text_from_claim_statement(statement: str) -> str | None:
    """Extract observation_text using the canonical Claim ``as:`` boundary.

    Drafted MODEL_INTERPRETATION statements are:
    ``The local visual model (…) interpreted retained snapshot {id} as: {text}``.
    Embedded quotation marks inside ``{text}`` must be preserved.
    """
    marker = _VISUAL_CLAIM_OBS_MARKER
    idx = statement.find(marker)
    if idx < 0:
        return None
    # Require the drafted interpretation preamble so we do not split on other "as:".
    head = statement[:idx].lower()
    if "interpreted retained snapshot" not in head and "interpreted" not in head:
        return None
    text = statement[idx + len(marker) :].strip()
    return text or None


def _observation_text_from_receipt(
    claim: Claim, *, repo: InvestigationRepository | None
) -> str | None:
    """Prefer durable observe_media receipt.filters['observation_text'] when present."""
    if repo is None:
        return None
    for receipt_id in claim.search_receipt_ids:
        receipt = repo.get_search_receipt(receipt_id)
        if receipt is None or receipt.tool_name != "observe_media":
            continue
        raw = (receipt.filters or {}).get("observation_text")
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
    return None


def _faithful_observation_text(
    claim: Claim, *, repo: InvestigationRepository | None = None
) -> str:
    """Return the exact registered VLM observation for operator display."""
    from_receipt = _observation_text_from_receipt(claim, repo=repo)
    if from_receipt:
        return from_receipt
    from_claim = _observation_text_from_claim_statement(claim.statement)
    if from_claim:
        return from_claim
    return (
        "Qualified visual interpretation text is unavailable for display; "
        "see Evidence Details for the canonical claim statement."
    )


def _operator_visual_text(statement: str) -> str:
    """Derive observation prose from a MODEL_INTERPRETATION claim statement only."""
    extracted = _observation_text_from_claim_statement(statement)
    if extracted:
        return extracted
    return (
        "Qualified visual interpretation text is unavailable for display; "
        "see Evidence Details for the canonical claim statement."
    )


def _select_timeline(
    observations: list[Observation], *, cap: int | None
) -> list[dict[str, Any]]:
    ordered = sorted(observations, key=lambda item: (item.start_utc, item.observation_id))
    rows: list[dict[str, Any]] = []
    limit = len(ordered) if cap is None else min(len(ordered), cap)
    for item in ordered[:limit]:
        rows.append(
            {
                "t": item.start_utc.isoformat(),
                "t_display": _fmt_time(item.start_utc),
                "end": None if item.end_utc is None else item.end_utc.isoformat(),
                "label": ", ".join(item.labels) if item.labels else "(unlabeled)",
                "camera": item.camera_id,
                "observation_id": item.observation_id,
                "native_record_id": item.source_locator.native_record_id,
            }
        )
    return rows


def _format_relations(
    relations: list[TemporalRelation], observations: list[Observation]
) -> list[dict[str, Any]]:
    labels = {
        item.observation_id: ", ".join(item.labels) if item.labels else item.observation_id
        for item in observations
    }
    rows: list[dict[str, Any]] = []
    for rel in relations:
        if rel.relation_type.value != "near":
            continue
        gap = rel.gap_seconds
        rows.append(
            {
                "relation_id": rel.relation_id,
                "left": rel.left_observation_id,
                "right": rel.right_observation_id,
                "left_label": labels.get(rel.left_observation_id, rel.left_observation_id),
                "right_label": labels.get(
                    rel.right_observation_id, rel.right_observation_id
                ),
                "gap_seconds": gap,
                "window_seconds": rel.window_seconds,
                "detail": (
                    f"{labels.get(rel.left_observation_id, 'event')} near "
                    f"{labels.get(rel.right_observation_id, 'event')} — "
                    + (
                        "unknown gap"
                        if gap is None
                        else f"{gap:.0f}s apart"
                    )
                    + f" (within {rel.window_seconds:.0f}s)"
                ),
            }
        )
    return rows


def _select_media(
    claims: list[Claim],
    evidence_by_id: dict[str, EvidenceRef],
    *,
    repo: InvestigationRepository | None = None,
) -> list[dict[str, Any]]:
    visual_claims = [
        c for c in claims if c.claim_type is ClaimType.MODEL_INTERPRETATION
    ]
    rows: list[dict[str, Any]] = []
    for claim in visual_claims:
        for evidence_id in claim.evidence_ids:
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None or evidence.evidence_kind is not EvidenceKind.SNAPSHOT:
                continue
            retained = evidence.retrieval_spec.get("retained_media_id")
            rows.append(
                {
                    "evidence_id": evidence_id,
                    "retained_media_id": retained,
                    "content_sha256": evidence.content_sha256,
                    "sha256_prefix": (
                        None
                        if not evidence.content_sha256
                        else evidence.content_sha256[:16]
                    ),
                    "visual_text": _faithful_observation_text(claim, repo=repo),
                    "limitations": list(claim.limitations),
                    "producer": claim.producer,
                    "claim_id": claim.claim_id,
                }
            )
    # One card per retained snapshot.
    deduped: list[dict[str, Any]] = []
    seen_media: set[str] = set()
    for row in rows:
        key = str(row.get("retained_media_id") or row.get("evidence_id"))
        if key in seen_media:
            continue
        seen_media.add(key)
        deduped.append(row)
    rows = deduped
    # Snapshots without visual claims still listed briefly.
    if not rows:
        for evidence in evidence_by_id.values():
            if evidence.evidence_kind is not EvidenceKind.SNAPSHOT:
                continue
            retained = evidence.retrieval_spec.get("retained_media_id")
            rows.append(
                {
                    "evidence_id": evidence.evidence_id,
                    "retained_media_id": retained,
                    "content_sha256": evidence.content_sha256,
                    "sha256_prefix": (
                        None
                        if not evidence.content_sha256
                        else evidence.content_sha256[:16]
                    ),
                    "visual_text": None,
                    "limitations": [
                        "Snapshot retained; no qualified visual interpretation "
                        "was registered."
                    ],
                    "producer": None,
                    "claim_id": None,
                }
            )
            if len(rows) >= 3:
                break
    return rows


def _build_summary(
    *,
    status: str,
    question: str,
    findings: list[dict[str, Any]],
    timeline: list[dict[str, Any]],
    relations: list[dict[str, Any]],
    media: list[dict[str, Any]],
    limitations: list[str],
    unresolved: list[str],
    observation_count: int,
    scope: dict[str, Any],
) -> str:
    cameras = ", ".join(scope.get("cameras") or []) or "authorized cameras"
    parts: list[str] = []

    if observation_count:
        label_counts: dict[str, int] = {}
        for row in timeline:
            label = row.get("label") or "detection"
            label_counts[label] = label_counts.get(label, 0) + 1
        label_bits = ", ".join(
            f"{count} {label}" for label, count in sorted(label_counts.items())
        )
        parts.append(
            f"Outcome ({status}): found {observation_count} authorized detection(s) "
            f"on {cameras} in the requested window"
            + (f" — {label_bits}" if label_bits else "")
            + "."
        )
    elif status in {"completed", "partial"}:
        parts.append(
            f"Outcome ({status}): no authorized detections were registered for "
            f"{cameras} in the requested window."
        )
    else:
        parts.append(f"Outcome ({status}).")

    if relations:
        parts.append(
            f"{len(relations)} nearby-activity (NEAR) relation(s) were established "
            "(temporal proximity only)."
        )
    visual = [f for f in findings if f["kind"] == "model_interpretation"]
    if visual:
        parts.append(
            "A qualified local visual interpretation is available; it is model "
            "opinion, not a verified physical-world fact."
        )
    elif media:
        parts.append("Snapshot media was retained without a visual interpretation.")
    if any("partial investigation coverage" in item.lower() for item in limitations):
        parts.append(
            "Some requested investigation objectives were not fully executed."
        )
    if unresolved:
        parts.append(f"{len(unresolved)} unresolved question(s) remain.")
    if question and observation_count == 0 and not findings:
        parts.append("No evidence-grounded findings answered the question.")
    return " ".join(parts)
