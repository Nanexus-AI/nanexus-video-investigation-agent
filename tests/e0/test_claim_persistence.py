"""Durable Claim persistence at domain finalization (T7B D1)."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from nanexus_investigation.contracts.enums import CaseStatus, ClaimType
from nanexus_investigation.graph.runtime import ScenarioConfig, make_case, ok_event_result
from nanexus_investigation.graph.workflow import InvestigationRuntime, run_investigation
from nanexus_investigation.persistence.db import InvestigationDatabase
from nanexus_investigation.persistence.repository import InvestigationRepository

UTC = timezone.utc
T0 = datetime(2026, 11, 19, 20, 0, tzinfo=UTC)


def _open_saver(path: Path) -> tuple[sqlite3.Connection, SqliteSaver]:
    conn = sqlite3.connect(str(path), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    return conn, saver


def test_t7b_accepted_claims_persist_and_reopen_without_models(tmp_path: Path):
    domain_db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    runtime = InvestigationRuntime.from_durable_fixture(
        case=make_case(case_id="case-t7b-claims"),
        scenario=ScenarioConfig(
            name="t7b_claims",
            tool_scripts={
                "search_events": [
                    ok_event_result(
                        execution_id="exec-t7b-c",
                        native_id="evt-t7b-c",
                        label="person",
                        start=T0 + timedelta(minutes=1),
                        end=T0 + timedelta(minutes=2),
                    )
                ]
            },
        ),
        domain_db=domain_db,
    )
    conn, saver = _open_saver(tmp_path / "ckpt.sqlite")
    result = run_investigation(
        runtime, thread_id="t-t7b-claims", checkpointer=saver, durability="sync"
    )
    assert result["domain_finalized"] is True
    report = result["domain_report"]
    assert report is not None
    assert report.claims
    assert runtime.case.status in {CaseStatus.COMPLETED, CaseStatus.PARTIAL}

    # Fresh repository — no Agent, no models.
    domain_db.close()
    reopened_db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    repo = InvestigationRepository(reopened_db)
    loaded_report = repo.get_report("case-t7b-claims")
    assert loaded_report is not None
    assert loaded_report.report_id == report.report_id
    claims = repo.list_claims_for_report("case-t7b-claims", loaded_report)
    assert len(claims) == len(loaded_report.claims)
    assert all(c.claim_id in loaded_report.claims for c in claims)
    assert any(c.claim_type is ClaimType.SOURCE_OBSERVATION for c in claims)
    for claim in claims:
        for evidence_id in claim.evidence_ids:
            assert repo.get_evidence(evidence_id) is not None
    conn.close()
    reopened_db.close()
