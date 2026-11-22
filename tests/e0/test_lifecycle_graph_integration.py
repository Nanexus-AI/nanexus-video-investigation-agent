"""G4-V1-EB2B L15: durable Graph integration with admission + finalization."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from nanexus_investigation.contracts.enums import CaseStatus
from nanexus_investigation.graph.runtime import (
    ScenarioConfig,
    make_case,
    ok_event_result,
)
from nanexus_investigation.graph.workflow import InvestigationRuntime, run_investigation
from nanexus_investigation.persistence.db import InvestigationDatabase

UTC = timezone.utc
T0 = datetime(2026, 11, 19, 20, 0, tzinfo=UTC)


def test_l15_durable_trace_admission_and_finalization(tmp_path: Path):
    domain_db = InvestigationDatabase.open(tmp_path / "domain.sqlite")
    runtime = InvestigationRuntime.from_durable_fixture(
        case=make_case(case_id="case-l15"),
        scenario=ScenarioConfig(
            name="l15",
            include_follow_up=True,
            tool_scripts={
                "search_events": [
                    ok_event_result(
                        execution_id="l15-p",
                        native_id="evt-l15-p",
                        label="person",
                        start=T0 + timedelta(minutes=1),
                        end=T0 + timedelta(minutes=2),
                    )
                ],
                "search_related_events": [
                    ok_event_result(
                        execution_id="l15-v",
                        native_id="evt-l15-v",
                        label="vehicle",
                        start=T0 + timedelta(minutes=5),
                        end=T0 + timedelta(minutes=6),
                    )
                ],
            },
        ),
        domain_db=domain_db,
    )
    conn = sqlite3.connect(str(tmp_path / "ckpt.sqlite"), check_same_thread=False)
    saver = SqliteSaver(conn)
    saver.setup()
    result = run_investigation(
        runtime, thread_id="t-l15", checkpointer=saver, durability="sync"
    )

    print("\n=== G4-V1-EB2B EXECUTION TRACE ===")
    print("case_id:", runtime.case.case_id)
    print("admission:", runtime.trace.of_kind("domain_case_admitted"))
    print("registration_groups:", runtime.trace.of_kind("domain_registration_group_committed"))
    print("output_gate:", runtime.trace.of_kind("output_gate"))
    print("finalization:", runtime.trace.of_kind("domain_case_finalized"))
    print("domain_finalized:", result["domain_finalized"])
    print("domain_status:", runtime.domain_repo.get_case("case-l15").status.value)
    print(
        "domain_report_id:",
        None
        if result["domain_report"] is None
        else result["domain_report"].report_id,
    )
    print("=== END TRACE ===\n")

    assert result["domain_finalized"] is True
    assert result["finalization_error"] is None
    assert runtime.trace.of_kind("domain_case_admitted")
    assert runtime.trace.of_kind("domain_case_finalized")
    assert runtime.domain_repo.get_case("case-l15").status in {
        CaseStatus.COMPLETED,
        CaseStatus.PARTIAL,
    }
    assert runtime.domain_repo.get_report("case-l15") is not None
    assert len(runtime.trace.of_kind("domain_registration_group_committed")) == 2

    # In-memory path still independent.
    mem = InvestigationRuntime.from_fixture(
        case=make_case(case_id="case-l15-mem"),
        scenario=ScenarioConfig(
            name="l15_mem",
            tool_scripts={
                "search_events": [
                    ok_event_result(
                        execution_id="mem-l15",
                        native_id="evt-mem",
                        label="person",
                        start=T0 + timedelta(minutes=1),
                        end=T0 + timedelta(minutes=2),
                    )
                ]
            },
        ),
    )
    mem_result = run_investigation(mem, thread_id="t-l15-mem")
    assert mem_result["domain_finalized"] is None
    assert mem_result["report"] is not None
    assert not mem.uses_durable_persistence

    conn.close()
    domain_db.close()
