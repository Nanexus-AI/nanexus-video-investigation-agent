"""FastAPI application for the local Investigation Console."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from nanexus_investigation.contracts.enums import EvidenceKind
from nanexus_investigation.evidence.media_store import LocalMediaStore, MediaStoreError
from nanexus_investigation.web.config import WebConfig
from nanexus_investigation.web.worker import (
    InvestigationRequestError,
    InvestigationWorker,
    WorkerBusyError,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"


class InvestigateBody(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    camera_ids: list[str] = Field(min_length=1, max_length=32)
    start_utc: datetime
    end_utc: datetime

    @field_validator("question")
    @classmethod
    def _trim_question(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("question must be non-empty")
        return text

    @field_validator("camera_ids")
    @classmethod
    def _cameras(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value if item.strip()]
        if not cleaned:
            raise ValueError("camera_ids required")
        return cleaned

    @field_validator("start_utc", "end_utc")
    @classmethod
    def _aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamps must include timezone offset")
        return value.astimezone(timezone.utc)


def _allowed_browser_origins(*, host: str, port: int) -> set[str]:
    """Same-origin browser Origins for the configured bind address."""
    origins = {
        f"http://{host}:{port}",
        f"http://127.0.0.1:{port}",
        f"http://localhost:{port}",
    }
    return origins


def _local_origin_ok(request: Request, *, host: str, port: int) -> bool:
    """Allow same-origin browser traffic; reject foreign sites."""
    origin = request.headers.get("origin")
    if origin is None:
        return True  # non-browser / same-origin navigations
    return origin in _allowed_browser_origins(host=host, port=port)


def _host_header_ok(*, host_header: str, bind_host: str) -> bool:
    """Accept only the configured bind host (plus local/test aliases)."""
    allowed = {
        bind_host.lower(),
        "127.0.0.1",
        "localhost",
        "testserver",
        "",
    }
    return host_header in allowed


def create_app(config: WebConfig | None = None) -> FastAPI:
    cfg = config or WebConfig.from_env()
    worker = InvestigationWorker(config=cfg)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        worker.open()
        try:
            yield
        finally:
            worker.close()

    app = FastAPI(
        title="Nanexus Local Video Investigation Agent",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.worker = worker
    app.state.config = cfg

    @app.middleware("http")
    async def _security_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
        # Host must match the configured bind (plus local/test aliases).
        host_header = (request.headers.get("host") or "").split(":")[0].lower()
        if not _host_header_ok(host_header=host_header, bind_host=cfg.host):
            return JSONResponse({"detail": "invalid host"}, status_code=400)
        # Optional temporary LAN client allowlist (set via NANEXUS_WEB_ALLOWED_CLIENTS).
        if cfg.allowed_clients:
            client = request.client.host if request.client else ""
            if client not in cfg.allowed_clients:
                return JSONResponse({"detail": "client not allowed"}, status_code=403)
        if request.method in {"POST", "PUT", "PATCH", "DELETE"}:
            if not _local_origin_ok(request, host=cfg.host, port=cfg.port):
                return JSONResponse({"detail": "origin not allowed"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        # No unrestricted CORS.
        return response

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return worker.health()

    @app.get("/api/cameras")
    def cameras() -> dict[str, Any]:
        return {
            "cameras": list(worker.authorized_cameras()),
            "timezone": cfg.timezone_label,
        }

    @app.post("/api/investigations")
    def create_investigation(body: InvestigateBody) -> dict[str, Any]:
        try:
            job = worker.submit(
                question=body.question,
                camera_ids=body.camera_ids,
                start_utc=body.start_utc,
                end_utc=body.end_utc,
            )
        except WorkerBusyError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except InvestigationRequestError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"case_id": job.case_id, "status": job.status}

    @app.get("/api/investigations")
    def list_investigations(limit: int = 10) -> dict[str, Any]:
        # Display/query cap for Recent Cases only — not a retention policy.
        capped = max(1, min(int(limit), 10))
        return {"cases": worker.list_jobs(limit=capped)}

    @app.get("/api/investigations/{case_id}")
    def get_investigation(case_id: str) -> dict[str, Any]:
        job = worker.get_job(case_id)
        if job is None:
            raise HTTPException(status_code=404, detail="case not found")
        payload: dict[str, Any] = {
            "case_id": job.case_id,
            "status": job.status,
            "question": job.question,
            "error": job.error,
            "interrupted": job.interrupted,
        }
        if job.view is not None:
            # Escape model-origin strings already handled in UI; keep raw for JSON.
            payload["result"] = job.view
        return payload

    @app.get("/api/investigations/{case_id}/evidence/{evidence_id}")
    def get_evidence(case_id: str, evidence_id: str) -> dict[str, Any]:
        repo = worker.repo
        case = repo.get_case(case_id)
        if case is None:
            raise HTTPException(status_code=404, detail="case not found")
        evidence = repo.get_evidence(evidence_id)
        if evidence is None:
            raise HTTPException(status_code=404, detail="evidence not found")
        if evidence.source_locator.vms_instance_id != case.effective_scope.vms_instance_id:
            raise HTTPException(status_code=403, detail="evidence outside case VMS")
        if evidence.camera_id not in case.effective_scope.camera_ids:
            raise HTTPException(status_code=403, detail="evidence camera not authorized")
        return {
            "evidence_id": evidence.evidence_id,
            "evidence_kind": evidence.evidence_kind.value,
            "camera_id": evidence.camera_id,
            "content_sha256": evidence.content_sha256,
            "availability": evidence.availability.value,
            "retrieval_spec": {
                k: evidence.retrieval_spec.get(k)
                for k in (
                    "retained_media_id",
                    "byte_count",
                    "content_type",
                    "media_kind",
                )
                if k in evidence.retrieval_spec
            },
            "source_locator": {
                "vms_instance_id": evidence.source_locator.vms_instance_id,
                "native_record_id": evidence.source_locator.native_record_id,
                "record_type": evidence.source_locator.record_type.value,
            },
        }

    @app.get("/api/investigations/{case_id}/media/{retained_media_id}")
    def get_media(case_id: str, retained_media_id: str) -> Response:
        repo = worker.repo
        case = repo.get_case(case_id)
        if case is None:
            raise HTTPException(status_code=404, detail="case not found")
        # Bind retained media to an EvidenceRef on this case.
        report = repo.get_report(case_id)
        evidence_ids = list(report.evidence_index) if report is not None else []
        if report is not None:
            for claim_id in report.claims:
                claim = repo.get_claim(claim_id)
                if claim is not None:
                    evidence_ids.extend(claim.evidence_ids)
        matched = None
        for evidence_id in dict.fromkeys(evidence_ids):
            evidence = repo.get_evidence(evidence_id)
            if evidence is None:
                continue
            if evidence.evidence_kind is not EvidenceKind.SNAPSHOT:
                continue
            if evidence.retrieval_spec.get("retained_media_id") != retained_media_id:
                continue
            if evidence.camera_id not in case.effective_scope.camera_ids:
                raise HTTPException(status_code=403, detail="unauthorized camera")
            matched = evidence
            break
        if matched is None:
            raise HTTPException(status_code=404, detail="media not bound to case evidence")

        store = LocalMediaStore(cfg.media_store_root / case_id)
        try:
            retained = store.get(case_id=case_id, retained_media_id=retained_media_id)
        except MediaStoreError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if matched.content_sha256 and retained.content_sha256 != matched.content_sha256:
            raise HTTPException(status_code=409, detail="media integrity mismatch")
        return Response(
            content=retained.bytes,
            media_type=retained.content_type or "application/octet-stream",
            headers={"Cache-Control": "no-store"},
        )

    @app.get("/")
    def index() -> HTMLResponse:
        index_path = STATIC_DIR / "index.html"
        if not index_path.is_file():
            return HTMLResponse(
                "<h1>Nanexus</h1><p>UI assets missing.</p>", status_code=500
            )
        # Serve HTML with escaped brand; static JS/CSS via mount.
        text = index_path.read_text(encoding="utf-8")
        return HTMLResponse(text)

    if STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    return app


def main() -> None:
    import uvicorn

    cfg = WebConfig.from_env()
    uvicorn.run(
        "nanexus_investigation.web.app:create_app",
        factory=True,
        host=cfg.host,
        port=cfg.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
