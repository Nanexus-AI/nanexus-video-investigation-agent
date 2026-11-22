"""T7B Web console API, OperatorView, and claim reopen tests."""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from nanexus_investigation.web.app import create_app
from nanexus_investigation.web.config import WebConfig
from nanexus_investigation.web.operator_view import build_operator_view

UTC = timezone.utc


@pytest.fixture()
def web_client(tmp_path: Path):
    cfg = WebConfig(
        host="127.0.0.1",
        port=8787,
        domain_db_path=tmp_path / "domain.sqlite",
        media_store_root=tmp_path / "media",
        tool_backend="stub",
        stub_cameras=("front_yard",),
        timezone_label="UTC",
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        yield client, cfg


def test_t7b_health_and_cameras(web_client):
    client, _ = web_client
    health = client.get("/api/health").json()
    assert health["ok"] is True
    assert health["backend"] == "stub"
    assert "front_yard" in health["cameras"]
    cams = client.get("/api/cameras").json()
    assert cams["cameras"] == ["front_yard"]


def test_r2c_live_health_without_frigate_config_is_not_http_500(tmp_path: Path):
    """Missing Frigate env must not crash /api/health (app liveness vs readiness)."""
    cfg = WebConfig(
        host="127.0.0.1",
        port=8787,
        domain_db_path=tmp_path / "domain.sqlite",
        media_store_root=tmp_path / "media",
        frigate_env_path=tmp_path / "missing-frigate.env",
        tool_backend="live",
        stub_cameras=("front_yard",),
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        resp = client.get("/api/health")
        assert resp.status_code == 200
        health = resp.json()
        assert health["ok"] is True
        assert health["backend"] == "live"
        assert health["frigate"]["ready"] is False
        assert "NANEXUS_FRIGATE_BASE_URL" in health["frigate"]["detail"]
        assert health["cameras"] == []
        cams = client.get("/api/cameras").json()
        assert cams["cameras"] == []
        # Fail-closed: no configured allowlist → any camera rejected.
        start = datetime(2026, 11, 19, 20, 0, tzinfo=UTC)
        end = start + timedelta(hours=1)
        denied = client.post(
            "/api/investigations",
            json={
                "question": "Find person detections.",
                "camera_ids": ["front_yard"],
                "start_utc": start.isoformat(),
                "end_utc": end.isoformat(),
            },
        )
        assert denied.status_code == 400
        assert "unauthorized" in denied.json()["detail"].lower()


def test_t7b_reject_unauthorized_camera(web_client):
    client, _ = web_client
    start = datetime(2026, 11, 19, 20, 0, tzinfo=UTC)
    end = start + timedelta(hours=1)
    resp = client.post(
        "/api/investigations",
        json={
            "question": "Find person detections.",
            "camera_ids": ["back_alley"],
            "start_utc": start.isoformat(),
            "end_utc": end.isoformat(),
        },
    )
    assert resp.status_code == 400
    assert "unauthorized" in resp.json()["detail"].lower()


def test_t7b_stub_investigation_poll_and_reopen(web_client):
    client, cfg = web_client
    start = datetime(2026, 11, 19, 20, 0, tzinfo=UTC)
    end = start + timedelta(hours=1)
    created = client.post(
        "/api/investigations",
        json={
            "question": "Find person detections in front_yard.",
            "camera_ids": ["front_yard"],
            "start_utc": start.isoformat(),
            "end_utc": end.isoformat(),
        },
    )
    assert created.status_code == 200
    case_id = created.json()["case_id"]
    assert created.json()["status"] == "running"

    deadline = time.time() + 30
    payload = None
    while time.time() < deadline:
        payload = client.get(f"/api/investigations/{case_id}").json()
        if payload["status"] != "running":
            break
        time.sleep(0.05)
    assert payload is not None
    assert payload["status"] in {"completed", "partial"}
    assert payload["result"]["summary"]
    assert payload["result"]["findings"]
    assert payload["result"]["timeline"]
    # Operator-facing findings must not lead with raw internal evidence IDs.
    for finding in payload["result"]["findings"]:
        detail = finding["detail"].lower()
        assert "ev-exec-" not in detail
        assert "authorized source returned event metadata" not in detail
        assert finding["refs"]["canonical_statement"]
    assert "Outcome" in payload["result"]["summary"] or "outcome" in payload["result"]["summary"].lower()

    listed = client.get("/api/investigations").json()
    assert any(item["case_id"] == case_id for item in listed["cases"])

    # Fresh app/repo reopen — no models.
    app2 = create_app(
        WebConfig(
            host="127.0.0.1",
            port=8787,
            domain_db_path=cfg.domain_db_path,
            media_store_root=cfg.media_store_root,
            tool_backend="stub",
            stub_cameras=("front_yard",),
        )
    )
    with TestClient(app2) as client2:
        reopened = client2.get(f"/api/investigations/{case_id}").json()
        assert reopened["status"] in {"completed", "partial"}
        assert reopened["result"]["findings"]
        tech = reopened["result"]["technical"]
        assert tech["claim_ids"]
        # Durable claims resolve through OperatorView rebuild.
        from nanexus_investigation.persistence.db import InvestigationDatabase
        from nanexus_investigation.persistence.repository import InvestigationRepository

        db = InvestigationDatabase.open(cfg.domain_db_path)
        repo = InvestigationRepository(db)
        view = build_operator_view(repo=repo, case_id=case_id)
        assert view["status"] in {"completed", "partial"}
        assert view["findings"]
        report = repo.get_report(case_id)
        assert report is not None
        claims = repo.list_claims_for_report(case_id, report)
        assert len(claims) == len(report.claims)
        db.close()


def test_t7b_index_page_served(web_client):
    client, _ = web_client
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Nanexus" in resp.text
    assert "Local Video Investigation Agent" in resp.text
    css = client.get("/static/styles.css")
    assert css.status_code == 200
    js = client.get("/static/app.js")
    assert js.status_code == 200


def test_t7b_foreign_origin_post_rejected(web_client):
    client, _ = web_client
    start = datetime(2026, 11, 19, 20, 0, tzinfo=UTC)
    end = start + timedelta(hours=1)
    resp = client.post(
        "/api/investigations",
        headers={"Origin": "https://evil.example"},
        json={
            "question": "Find person detections.",
            "camera_ids": ["front_yard"],
            "start_utc": start.isoformat(),
            "end_utc": end.isoformat(),
        },
    )
    assert resp.status_code == 403


def test_t7c_lan_host_and_client_allowlist(tmp_path: Path):
    """Temporary LAN review: Host must match bind; clients can be allowlisted."""
    cfg = WebConfig(
        host="10.0.0.10",
        port=8787,
        domain_db_path=tmp_path / "domain.sqlite",
        media_store_root=tmp_path / "media",
        tool_backend="stub",
        stub_cameras=("front_yard",),
        allowed_clients=("10.0.0.20", "testclient"),
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        ok = client.get("/api/health", headers={"Host": "10.0.0.10:8787"})
        assert ok.status_code == 200
        bad_host = client.get("/api/health", headers={"Host": "evil.example"})
        assert bad_host.status_code == 400
        # Starlette TestClient presents as testclient / 127.0.0.1 depending on version.
        # Reject an explicitly disallowed peer by temporarily narrowing the allowlist.
    cfg2 = WebConfig(
        host="10.0.0.10",
        port=8787,
        domain_db_path=tmp_path / "domain2.sqlite",
        media_store_root=tmp_path / "media2",
        tool_backend="stub",
        stub_cameras=("front_yard",),
        allowed_clients=("203.0.113.9",),
    )
    app2 = create_app(cfg2)
    with TestClient(app2) as client2:
        denied = client2.get("/api/health", headers={"Host": "10.0.0.10:8787"})
        assert denied.status_code == 403
        assert denied.json()["detail"] == "client not allowed"


def test_t7c_near_findings_do_not_repeat_causation_disclaimer(tmp_path: Path):
    """NEAR findings stay concise; shared qualification appears once."""
    from nanexus_investigation.contracts.claim import Claim
    from nanexus_investigation.contracts.enums import (
        ClaimType,
        ClaimValidationStatus,
        ObservationType,
        RecordType,
        TemporalEvaluationStatus,
        TemporalRelationType,
    )
    from nanexus_investigation.contracts.evidence import Observation
    from nanexus_investigation.contracts.identity import SourceLocator
    from nanexus_investigation.contracts.temporal import TemporalRelation
    from nanexus_investigation.web.operator_view import (
        _NEAR_QUALIFICATION,
        _format_relations,
        _select_findings,
    )

    loc = SourceLocator(
        vms_instance_id="vms-1",
        native_record_id="evt-1",
        record_type=RecordType.EVENT,
    )
    obs_a = Observation(
        observation_id="obs-a",
        source_locator=loc,
        source_revision="r1",
        observation_type=ObservationType.DETECTION_EVENT,
        camera_id="front_yard",
        start_utc=datetime(2026, 10, 10, 12, 0, tzinfo=UTC),
        end_utc=datetime(2026, 10, 10, 12, 1, tzinfo=UTC),
        labels=("person",),
        retrieved_at=datetime(2026, 10, 10, 12, 2, tzinfo=UTC),
        provenance="test",
    )
    loc_b = SourceLocator(
        vms_instance_id="vms-1",
        native_record_id="evt-2",
        record_type=RecordType.EVENT,
    )
    obs_b = Observation(
        observation_id="obs-b",
        source_locator=loc_b,
        source_revision="r1",
        observation_type=ObservationType.DETECTION_EVENT,
        camera_id="front_yard",
        start_utc=datetime(2026, 10, 10, 12, 1, tzinfo=UTC),
        end_utc=datetime(2026, 10, 10, 12, 2, tzinfo=UTC),
        labels=("car",),
        retrieved_at=datetime(2026, 10, 10, 12, 2, tzinfo=UTC),
        provenance="test",
    )
    rel = TemporalRelation(
        relation_id="rel-1",
        relation_type=TemporalRelationType.NEAR,
        left_observation_id="obs-a",
        right_observation_id="obs-b",
        evaluation_status=TemporalEvaluationStatus.ESTABLISHED,
        gap_seconds=69.0,
        window_seconds=180.0,
        camera_relation="same_camera",
        time_basis="observation_intervals",
        calculation_provenance="test",
    )
    claim = Claim(
        claim_id="claim-rel-1",
        statement=(
            "Deterministic near relation established between observations "
            "obs-a and obs-b (69.0s gap, window=180s). Temporal proximity alone "
            "does not prove physical interaction, common identity, ownership, "
            "or causation."
        ),
        claim_type=ClaimType.COMPUTED_RELATION,
        evidence_ids=(),
        relation_ids=("rel-1",),
        producer="temporal-engine:analyze_temporal_relations",
        validation_status=ClaimValidationStatus.QUALIFIED,
        limitations=(
            "Temporal proximity alone does not prove physical interaction.",
        ),
        validated_at=None,
    )
    findings = _select_findings(
        [claim],
        observations=[obs_a, obs_b],
        evidence_by_id={},
        relations=[rel],
        cap=None,
    )
    assert len(findings) == 1
    detail = findings[0]["detail"]
    assert "69s" in detail
    assert "causation" not in detail.lower()
    assert findings[0]["limitations"] == []
    rows = _format_relations([rel], [obs_a, obs_b])
    assert "69" in rows[0]["detail"]
    assert "note" not in rows[0]
    assert "interaction" in _NEAR_QUALIFICATION.lower()


def test_t7c_operator_view_prefers_readable_event_findings(web_client):
    """default findings describe camera/label/time, not raw evidence IDs."""
    client, _ = web_client
    start = datetime(2026, 11, 19, 20, 0, tzinfo=UTC)
    end = start + timedelta(hours=1)
    created = client.post(
        "/api/investigations",
        json={
            "question": "Find person detections in front_yard.",
            "camera_ids": ["front_yard"],
            "start_utc": start.isoformat(),
            "end_utc": end.isoformat(),
        },
    )
    case_id = created.json()["case_id"]
    deadline = time.time() + 30
    payload = None
    while time.time() < deadline:
        payload = client.get(f"/api/investigations/{case_id}").json()
        if payload["status"] != "running":
            break
        time.sleep(0.05)
    assert payload is not None
    result = payload["result"]
    assert result["timeline"]
    first = result["timeline"][0]
    assert first["label"]
    assert first["camera"] == "front_yard"
    assert "t_display" in first
    finding = result["findings"][0]
    assert "front_yard" in finding["detail"] or "front_yard" in finding["headline"]
    assert "person" in finding["detail"].lower() or "person" in finding["headline"].lower()
    assert "canonical_statement" in finding["refs"]
    page = client.get("/").text
    assert 'id="findings-note"' in page


def test_t7b_busy_worker_rejects_new_request(web_client):
    client, _ = web_client
    worker = client.app.state.worker
    with worker._lock:
        worker._busy = True
    start = datetime(2026, 11, 19, 20, 0, tzinfo=UTC)
    end = start + timedelta(hours=1)
    try:
        resp = client.post(
            "/api/investigations",
            json={
                "question": "Find person detections.",
                "camera_ids": ["front_yard"],
                "start_utc": start.isoformat(),
                "end_utc": end.isoformat(),
            },
        )
        assert resp.status_code == 409
    finally:
        with worker._lock:
            worker._busy = False


def test_t7v_faithful_observation_preserves_embedded_quotes():
    """P0: do not use greedy/innermost quote extraction for VLM text."""
    from nanexus_investigation.contracts.claim import Claim
    from nanexus_investigation.contracts.enums import ClaimType, ClaimValidationStatus
    from nanexus_investigation.web.operator_view import (
        _ANTI_REPEAT_OPERATOR,
        _MEDIA_NONE_RETRIEVED,
        _faithful_observation_text,
        _media_absence_message,
        _observation_text_from_claim_statement,
        _operator_limitation_text,
    )

    full = (
        'The image also contains some text, which reads "DETERMINIST: 727.232", '
        "possibly indicating the name or address of the person who took this photo."
    )
    statement = (
        "The local visual model (moondream:1.8b) interpreted retained snapshot "
        f"ev-media-1 as: {full}"
    )
    assert _observation_text_from_claim_statement(statement) == full
    multi = 'Sees "person" near "car" and a sign saying "OPEN".'
    multi_statement = (
        "The local visual model (x) interpreted retained snapshot ev-1 as: " + multi
    )
    assert _observation_text_from_claim_statement(multi_statement) == multi
    plain = "A vehicle is visible on the street."
    plain_statement = (
        "The local visual model (x) interpreted retained snapshot ev-1 as: " + plain
    )
    assert _observation_text_from_claim_statement(plain_statement) == plain
    assert _observation_text_from_claim_statement("no marker here") is None
    assert (
        _observation_text_from_claim_statement(
            "The local visual model (x) interpreted retained snapshot ev-1 as: "
        )
        is None
    )
    # Malformed claim: missing as: boundary → truthful unavailable, not a quote fragment.
    empty_claim = Claim(
        claim_id="claim-empty-visual",
        statement=(
            "The local visual model (x) interpreted retained snapshot ev-1 "
            'with embedded "DETERMINIST: 727.232" but no canonical boundary.'
        ),
        claim_type=ClaimType.MODEL_INTERPRETATION,
        evidence_ids=("ev-1",),
        relation_ids=(),
        producer="local-vlm:test",
        validation_status=ClaimValidationStatus.QUALIFIED,
        limitations=(),
        validated_at=None,
    )
    unavailable = _faithful_observation_text(empty_claim, repo=None)
    assert "unavailable" in unavailable.lower()
    assert unavailable != "DETERMINIST: 727.232"

    good = Claim(
        claim_id="claim-visual-quotes",
        statement=statement,
        claim_type=ClaimType.MODEL_INTERPRETATION,
        evidence_ids=("ev-1",),
        relation_ids=(),
        producer="local-vlm:moondream:1.8b@digest",
        validation_status=ClaimValidationStatus.QUALIFIED,
        limitations=(),
        validated_at=None,
        search_receipt_ids=(),
    )
    text = _faithful_observation_text(good, repo=None)
    assert text == full
    assert text.startswith("The image also contains")
    assert '"DETERMINIST: 727.232"' in text

    # P1: empty media without resolve_media attempt
    assert (
        _media_absence_message(
            repo=None, claims=[], evidence_by_id={}, limitations=[]
        )
        == _MEDIA_NONE_RETRIEVED
    )
    assert "No snapshots were retrieved" in _MEDIA_NONE_RETRIEVED
    assert "No snapshot evidence." not in _MEDIA_NONE_RETRIEVED

    # P2: anti-repeat operator wording
    durable = (
        "partial investigation stop: identical tool request already executed; "
        "duplicate retrieval blocked before Tool Guard reservation"
    )
    op = _operator_limitation_text(durable)
    assert op == _ANTI_REPEAT_OPERATOR
    assert "Useful evidence was already obtained" in op
    assert "PARTIAL" in op
    assert "identical tool request" not in op


def test_t7v_stub_investigation_empty_media_wording(web_client):
    """P1 acceptance A: ordinary temporal/stub search shows truthful empty-media copy."""
    client, _ = web_client
    start = datetime(2026, 11, 19, 20, 0, tzinfo=UTC)
    end = start + timedelta(hours=1)
    created = client.post(
        "/api/investigations",
        json={
            "question": "Find person detections in front_yard.",
            "camera_ids": ["front_yard"],
            "start_utc": start.isoformat(),
            "end_utc": end.isoformat(),
        },
    )
    case_id = created.json()["case_id"]
    deadline = time.time() + 30
    payload = None
    while time.time() < deadline:
        payload = client.get(f"/api/investigations/{case_id}").json()
        if payload["status"] != "running":
            break
        time.sleep(0.05)
    assert payload is not None
    result = payload["result"]
    assert result["media"] == []
    assert (
        result["media_empty_message"]
        == "No snapshots were retrieved for this investigation."
    )
    js = client.get("/static/app.js").text
    assert "media_empty_message" in js
    assert "No snapshots were retrieved for this investigation." in js
    assert "No snapshot evidence." not in js


def test_t7v_reopen_diagnostic_case_preserves_full_observation():
    """P0 acceptance C: reopen case-web-ae5d8acd6a8c without Frigate/LLM/VLM calls."""
    db_path = Path(".local/web/domain.sqlite")
    if not db_path.is_file():
        pytest.skip("diagnostic web domain DB not present")
    from nanexus_investigation.persistence.db import InvestigationDatabase
    from nanexus_investigation.persistence.repository import InvestigationRepository

    case_id = "case-web-ae5d8acd6a8c"
    db = InvestigationDatabase.open(db_path)
    try:
        repo = InvestigationRepository(db)
        if repo.get_case(case_id) is None:
            pytest.skip("diagnostic case not present")
        report = repo.get_report(case_id)
        assert report is not None
        claims_before = repo.list_claims_for_report(case_id, report)
        visual = [
            c
            for c in claims_before
            if c.claim_type.value == "model_interpretation"
        ]
        assert visual
        stmt_before = visual[0].statement
        receipt = repo.get_search_receipt(visual[0].search_receipt_ids[0])
        assert receipt is not None
        obs_text = receipt.filters["observation_text"]
        assert "DETERMINIST: 727.232" in obs_text
        assert obs_text.startswith("The image shows")

        view = build_operator_view(repo=repo, case_id=case_id)
        assert view["status"] == "partial"
        assert view["media"]
        shown = view["media"][0]["visual_text"]
        assert shown == obs_text
        assert shown != "DETERMINIST: 727.232"
        assert '"DETERMINIST: 727.232"' in shown
        finding = next(
            f
            for f in view["findings"]
            if f["headline"] == "Qualified local visual interpretation"
        )
        assert obs_text[:80] in finding["detail"]
        assert any(
            "Useful evidence was already obtained" in lim for lim in view["limitations"]
        )
        # Fidelity: durable claim statement unchanged by presentation.
        claims_after = repo.list_claims_for_report(case_id, report)
        assert claims_after[0].statement == stmt_before or any(
            c.statement == stmt_before for c in claims_after
        )
        assert visual[0].validation_status.value == "qualified"
    finally:
        db.close()


def test_t7_ui_reopen_restores_request_scope_fields(web_client):
    """Historical reopen must expose question+scope so the form can match Results."""
    client, _ = web_client
    page = client.get("/").text
    assert 'id="viewing-hint"' in page
    assert 'id="new-investigation"' in page
    js = client.get("/static/app.js").text
    assert "applyRequestForm" in js
    assert "utcIsoToLocalInput" in js
    assert "Viewing saved investigation" in js
    assert "setViewingSaved" in js

    start = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
    end = datetime(2026, 10, 10, 13, 0, tzinfo=UTC)
    created = client.post(
        "/api/investigations",
        json={
            "question": "Find person detections in front_yard for form-restore test.",
            "camera_ids": ["front_yard"],
            "start_utc": start.isoformat(),
            "end_utc": end.isoformat(),
        },
    )
    case_id = created.json()["case_id"]
    deadline = time.time() + 30
    payload = None
    while time.time() < deadline:
        payload = client.get(f"/api/investigations/{case_id}").json()
        if payload["status"] != "running":
            break
        time.sleep(0.05)
    assert payload is not None
    assert payload["status"] in {"completed", "partial"}
    assert payload["question"].startswith("Find person detections")
    scope = payload["result"]["scope"]
    assert scope["cameras"] == ["front_yard"]
    assert scope["start_utc"].startswith("2026-10-10T12:00:00")
    assert scope["end_utc"].startswith("2026-10-10T13:00:00")

    # Reopen is durable read — same scope, no mutation of Case identity.
    reopened = client.get(f"/api/investigations/{case_id}").json()
    assert reopened["case_id"] == case_id
    assert reopened["result"]["scope"] == scope
    assert reopened["result"]["question"] == payload["result"]["question"]


def test_t7_ui_utc_to_local_input_respects_date_specific_offset():
    """Mirror app.js utcIsoToLocalInput: Date instant → local datetime-local wall time."""
    from datetime import datetime as dt
    from zoneinfo import ZoneInfo

    def utc_iso_to_local_input(iso: str, tz_name: str) -> str:
        instant = dt.fromisoformat(iso.replace("Z", "+00:00"))
        local = instant.astimezone(ZoneInfo(tz_name))
        return local.strftime("%Y-%m-%dT%H:%M")

    # America/Toronto: Oct 10 2026 is EDT (UTC-4); Jan 15 2026 is EST (UTC-5).
    edt = utc_iso_to_local_input("2026-10-10T12:00:00+00:00", "America/Toronto")
    est = utc_iso_to_local_input("2026-01-15T13:00:00+00:00", "America/Toronto")
    assert edt == "2026-10-10T08:00"
    assert est == "2026-01-15T08:00"
    end_edt = utc_iso_to_local_input("2026-10-10T13:00:00+00:00", "America/Toronto")
    assert end_edt == "2026-10-10T09:00"


def test_t7_ui_recent_cases_capped_at_ten_newest_first(tmp_path: Path):
    """Display/query limit is 10; older Cases remain in durable storage."""
    from nanexus_investigation.contracts.case import InvestigationCase
    from nanexus_investigation.contracts.enums import CaseStatus
    from nanexus_investigation.contracts.identity import Interval, TimeContext
    from nanexus_investigation.contracts.scope import InvestigationScope
    from nanexus_investigation.persistence.db import InvestigationDatabase
    from nanexus_investigation.persistence.repository import InvestigationRepository
    from nanexus_investigation.web.app import create_app
    from nanexus_investigation.web.config import WebConfig

    db_path = tmp_path / "domain.sqlite"
    db = InvestigationDatabase.open(db_path)
    repo = InvestigationRepository(db)
    base = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
    bounds = Interval(start_utc=base, end_utc=base + timedelta(hours=1))
    scope = InvestigationScope(
        vms_instance_id="vms-test",
        camera_ids=("front_yard",),
        time_bounds=bounds,
    )
    # 12 Cases; two newest share created_at to exercise case_id secondary sort.
    for i in range(12):
        case_id = f"case-web-limit-{i:02d}"
        created = base + timedelta(hours=1) if i >= 10 else base + timedelta(minutes=i)
        case = InvestigationCase(
            case_id=case_id,
            question=f"Q{i}",
            status=CaseStatus.CREATED,
            created_at=created,
            requested_scope=scope,
            effective_scope=scope,
            time_context=TimeContext(
                timezone="America/Toronto",
                start_utc=bounds.start_utc,
                end_utc=bounds.end_utc,
            ),
            policy_ref="policy-test",
        )
        repo.upsert_case(case)
    all_rows = repo.list_cases(limit=200)
    assert len(all_rows) == 12

    cfg = WebConfig(
        host="127.0.0.1",
        port=8787,
        domain_db_path=db_path,
        media_store_root=tmp_path / "media",
        tool_backend="stub",
        stub_cameras=("front_yard",),
    )
    app = create_app(cfg)
    with TestClient(app) as client:
        listed = client.get("/api/investigations").json()["cases"]
        assert len(listed) == 10
        ids = [c["case_id"] for c in listed]
        # Newest first; tie at top breaks as case_id DESC → 11 then 10.
        assert ids == [
            "case-web-limit-11",
            "case-web-limit-10",
            "case-web-limit-09",
            "case-web-limit-08",
            "case-web-limit-07",
            "case-web-limit-06",
            "case-web-limit-05",
            "case-web-limit-04",
            "case-web-limit-03",
            "case-web-limit-02",
        ]
        over = client.get("/api/investigations?limit=50").json()["cases"]
        assert len(over) == 10
        js = client.get("/static/app.js").text
        assert "/api/investigations?limit=10" in js
        page = client.get("/").text
        assert "Show More" not in page
        assert "pagination" not in page.lower()

    # Durable store still holds all 12 after list cap (not a retention policy).
    assert len(repo.list_cases(limit=200)) == 12
    assert {r["case_id"] for r in repo.list_cases(limit=200)} >= {
        "case-web-limit-00",
        "case-web-limit-01",
    }
    db.close()
