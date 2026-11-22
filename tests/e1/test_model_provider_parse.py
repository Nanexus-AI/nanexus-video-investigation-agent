"""Deterministic Model Provider parsing tests (no live inference)."""

from __future__ import annotations

import pytest

from nanexus_investigation.model.provider import (
    ProposalParseError,
    StaticModelProvider,
    parse_investigation_proposal,
)


def test_parse_invoke_tool_proposal():
    proposal = parse_investigation_proposal(
        '{"action":"invoke_tool","tool_name":"search_related_events",'
        '"label":"vehicle","camera_ids":["front_yard"],"rationale":"person found"}'
    )
    assert proposal.action == "invoke_tool"
    assert proposal.tool_name == "search_related_events"
    assert proposal.label == "vehicle"
    assert proposal.camera_ids == ("front_yard",)


def test_parse_proceed_clears_tool_name():
    proposal = parse_investigation_proposal(
        '{"action":"proceed_to_report","tool_name":"search_events",'
        '"label":null,"rationale":"no matches"}'
    )
    assert proposal.action == "proceed_to_report"
    assert proposal.tool_name is None
    assert proposal.time_bounds is None


def test_parse_optional_time_bounds():
    proposal = parse_investigation_proposal(
        '{"action":"invoke_tool","tool_name":"search_events","label":"car",'
        '"time_bounds":{"start_utc":"2026-10-10T12:00:00+00:00",'
        '"end_utc":"2026-10-10T13:00:00+00:00"},'
        '"rationale":"narrow vehicle window"}'
    )
    assert proposal.time_bounds == {
        "start_utc": "2026-10-10T12:00:00+00:00",
        "end_utc": "2026-10-10T13:00:00+00:00",
    }


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not-json",
        '{"action":"hack","tool_name":"x","rationale":"no"}',
        '{"action":"invoke_tool","tool_name":null,"rationale":"missing tool"}',
        '{"action":"proceed_to_report","rationale":""}',
    ],
)
def test_parse_rejects_malformed(raw: str):
    with pytest.raises(ProposalParseError):
        parse_investigation_proposal(raw)


def test_parse_resolve_media_defaults_media_kind_to_snapshot():
    proposal = parse_investigation_proposal(
        '{"action":"invoke_tool","tool_name":"resolve_media","label":null,'
        '"media_kind":null,"evidence_id":"ev-1","rationale":"need snapshot"}'
    )
    assert proposal.media_kind == "snapshot"
    assert proposal.media_arguments()["media_kind"] == "snapshot"


def test_parse_recovers_tool_name_placed_in_action():
    proposal = parse_investigation_proposal(
        '{"action":"observe_media","tool_name":null,"label":null,'
        '"evidence_id":"ev-media-1","objective":"Describe the image.",'
        '"rationale":"visual check"}'
    )
    assert proposal.action == "invoke_tool"
    assert proposal.tool_name == "observe_media"
    assert proposal.evidence_id == "ev-media-1"


def test_static_provider_records_context():
    provider = StaticModelProvider(
        raw_text='{"action":"proceed_to_report","tool_name":null,'
        '"label":null,"rationale":"stop"}'
    )
    from nanexus_investigation.model.provider import ProposalContext

    ctx = ProposalContext(
        question="q",
        case_id="c",
        authorized_tools=("search_events",),
        authorized_cameras=("front_yard",),
        vms_instance_id="vms-local-1",
        time_bounds={"start_utc": "t0", "end_utc": "t1"},
        prior_tool_name="search_events",
        prior_tool_status="empty",
        prior_match_count=0,
        prior_result_summary="none",
    )
    proposal = provider.propose_follow_up(ctx)
    assert proposal.action == "proceed_to_report"
    assert len(provider.calls) == 1
