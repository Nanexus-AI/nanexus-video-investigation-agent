"""Single bounded investigation worker for the local Web console."""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
from nanexus_investigation.contracts.identity import Interval
from nanexus_investigation.graph.runtime import ScenarioConfig, make_case
from nanexus_investigation.graph.workflow import InvestigationRuntime, run_investigation
from nanexus_investigation.model.provider import OllamaChatProvider
from nanexus_investigation.model.visual_provider import (
    OllamaVisualProvider,
    detect_local_visual_provider,
)
from nanexus_investigation.persistence.db import InvestigationDatabase
from nanexus_investigation.persistence.repository import InvestigationRepository
from nanexus_investigation.vms.frigate.client import FrigateReadOnlyClient
from nanexus_investigation.vms.frigate.record import (
    config_from_env,
    live_client_ready,
    load_frigate_env,
)
from nanexus_investigation.web.config import WebConfig
from nanexus_investigation.web.operator_view import (
    build_operator_view,
    build_operator_view_from_runtime,
)


class WorkerBusyError(RuntimeError):
    """Another investigation is already running."""


class InvestigationRequestError(ValueError):
    """Invalid investigation request."""


@dataclass
class JobRecord:
    case_id: str
    status: str
    created_at: str
    question: str
    error: str | None = None
    view: dict[str, Any] | None = None
    interrupted: bool = False


@dataclass
class InvestigationWorker:
    """At most one concurrent investigation. No unbounded queue."""

    config: WebConfig
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _busy: bool = False
    _jobs: dict[str, JobRecord] = field(default_factory=dict)
    _thread: threading.Thread | None = None
    _db: InvestigationDatabase | None = None

    def open(self) -> None:
        self.config.domain_db_path.parent.mkdir(parents=True, exist_ok=True)
        self.config.media_store_root.mkdir(parents=True, exist_ok=True)
        self._db = InvestigationDatabase.open(self.config.domain_db_path)
        self._reconcile_interrupted()

    def close(self) -> None:
        if self._db is not None:
            self._db.close()
            self._db = None

    @property
    def repo(self) -> InvestigationRepository:
        if self._db is None:
            raise RuntimeError("worker database is not open")
        return InvestigationRepository(self._db)

    def _reconcile_interrupted(self) -> None:
        """Mark durable running cases without reports as interrupted after restart."""
        repo = self.repo
        for row in repo.list_cases(limit=200):
            if row["status"] != "running":
                continue
            case_id = row["case_id"]
            if repo.get_report(case_id) is not None:
                continue
            self._jobs[case_id] = JobRecord(
                case_id=case_id,
                status="failed",
                created_at=row["created_at"],
                question=row["question"],
                error="interrupted_before_finalization",
                interrupted=True,
                view=build_operator_view(
                    repo=repo, case_id=case_id, interrupted=True
                ),
            )

    def is_busy(self) -> bool:
        with self._lock:
            return self._busy

    def submit(
        self,
        *,
        question: str,
        camera_ids: list[str],
        start_utc: datetime,
        end_utc: datetime,
    ) -> JobRecord:
        question = (question or "").strip()
        if not question:
            raise InvestigationRequestError("question is required")
        if not camera_ids:
            raise InvestigationRequestError("at least one camera is required")
        if start_utc.tzinfo is None or end_utc.tzinfo is None:
            raise InvestigationRequestError("start/end must be timezone-aware UTC")
        if end_utc <= start_utc:
            raise InvestigationRequestError("end_utc must be after start_utc")

        authorized = self.authorized_cameras()
        unauthorized = [c for c in camera_ids if c not in authorized]
        if unauthorized:
            raise InvestigationRequestError(
                f"unauthorized cameras: {unauthorized}; authorized={list(authorized)}"
            )

        with self._lock:
            if self._busy:
                raise WorkerBusyError(
                    "an investigation is already running; retry after it finishes"
                )
            self._busy = True
            case_id = f"case-web-{uuid.uuid4().hex[:12]}"
            job = JobRecord(
                case_id=case_id,
                status="running",
                created_at=datetime.now(tz=start_utc.tzinfo).isoformat(),
                question=question,
            )
            self._jobs[case_id] = job
            cameras = tuple(camera_ids)
            self._thread = threading.Thread(
                target=self._run_job,
                kwargs={
                    "case_id": case_id,
                    "question": question,
                    "cameras": cameras,
                    "start_utc": start_utc,
                    "end_utc": end_utc,
                },
                daemon=True,
                name=f"nanexus-investigate-{case_id}",
            )
            self._thread.start()
            return job

    def get_job(self, case_id: str) -> JobRecord | None:
        with self._lock:
            job = self._jobs.get(case_id)
            if job is not None:
                return job
        # Durable reopen path for finalized cases not in memory.
        repo = self.repo
        case = repo.get_case(case_id)
        if case is None:
            return None
        if case.status.value == "running":
            view = build_operator_view(repo=repo, case_id=case_id, interrupted=True)
            return JobRecord(
                case_id=case_id,
                status="failed",
                created_at=case.created_at.isoformat(),
                question=case.question,
                error="interrupted_before_finalization",
                interrupted=True,
                view=view,
            )
        view = build_operator_view(repo=repo, case_id=case_id)
        return JobRecord(
            case_id=case_id,
            status=view.get("status", case.status.value),
            created_at=case.created_at.isoformat(),
            question=case.question,
            view=view,
        )

    def list_jobs(self, *, limit: int = 10) -> list[dict[str, Any]]:
        rows = self.repo.list_cases(limit=limit)
        out: list[dict[str, Any]] = []
        for row in rows:
            case_id = row["case_id"]
            with self._lock:
                job = self._jobs.get(case_id)
            status = job.status if job is not None else row["status"]
            if status == "running" and (job is None or job.interrupted):
                # Durable running without active worker = interrupted.
                if job is None or not self._busy:
                    status = "failed"
            out.append(
                {
                    "case_id": case_id,
                    "status": status,
                    "question": row["question"],
                    "created_at": row["created_at"],
                    "vms_instance_id": row["vms_instance_id"],
                }
            )
        return out

    def authorized_cameras(self) -> tuple[str, ...]:
        """Return the configured camera allowlist (fail-closed when unset).

        Never invent cameras. Incomplete Frigate config must not raise — health and
        the camera list stay available while investigations remain blocked until
        ``live_client_ready`` succeeds.
        """
        if self.config.tool_backend == "stub":
            return self.config.stub_cameras
        env = load_frigate_env(self.config.frigate_env_path)
        return self._camera_allowlist_from_env(env)

    @staticmethod
    def _camera_allowlist_from_env(env: dict[str, str]) -> tuple[str, ...]:
        return tuple(
            camera.strip()
            for camera in (env.get("NANEXUS_FRIGATE_CAMERAS") or "").split(",")
            if camera.strip()
        )

    @staticmethod
    def _public_dependency_detail(detail: str) -> str:
        """Keep readiness details free of credentials and absolute paths."""
        text = (detail or "").strip()
        if "NANEXUS_FRIGATE_CA_BUNDLE file not found" in text:
            return "NANEXUS_FRIGATE_CA_BUNDLE file not found"
        return text

    def health(self) -> dict[str, Any]:
        """Application liveness plus dependency readiness (not a Frigate probe)."""
        if self.config.tool_backend == "stub":
            return {
                "ok": True,
                "bind": f"{self.config.host}:{self.config.port}",
                "busy": self.is_busy(),
                "timezone": self.config.timezone_label,
                "backend": "stub",
                "frigate": {"ready": False, "detail": "stub backend"},
                "ollama_chat": {"ready": True, "model": "static", "detail": "stub"},
                "ollama_visual": {"ready": False, "model": self.config.visual_model},
                "cameras": list(self.authorized_cameras()),
            }
        env = load_frigate_env(self.config.frigate_env_path)
        frigate_ready, frigate_reason = live_client_ready(env)
        chat_ok, chat_detail = self._probe_ollama_model(self.config.chat_model)
        visual = OllamaVisualProvider(
            model=self.config.visual_model, base_url=self.config.ollama_base_url
        )
        visual_ok = visual.is_available()
        cameras = self._camera_allowlist_from_env(env)
        return {
            # ok = Web process is up; dependency readiness is under frigate/ollama_*.
            "ok": True,
            "bind": f"{self.config.host}:{self.config.port}",
            "busy": self.is_busy(),
            "timezone": self.config.timezone_label,
            "backend": "live",
            "frigate": {
                "ready": frigate_ready,
                "detail": self._public_dependency_detail(frigate_reason),
            },
            "ollama_chat": {
                "ready": chat_ok,
                "model": self.config.chat_model,
                "detail": chat_detail,
            },
            "ollama_visual": {
                "ready": visual_ok,
                "model": self.config.visual_model,
            },
            "cameras": list(cameras),
        }

    def _probe_ollama_model(self, model: str) -> tuple[bool, str]:
        import json
        import urllib.error
        import urllib.request

        try:
            with urllib.request.urlopen(
                f"{self.config.ollama_base_url}/api/tags", timeout=3
            ) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return False, str(exc)
        names = {
            m.get("name") for m in payload.get("models", []) if isinstance(m, dict)
        }
        if model in names:
            return True, "installed"
        return False, f"model {model!r} not installed; have={sorted(names)}"

    def _run_job(
        self,
        *,
        case_id: str,
        question: str,
        cameras: tuple[str, ...],
        start_utc: datetime,
        end_utc: datetime,
    ) -> None:
        try:
            view = self._execute(
                case_id=case_id,
                question=question,
                cameras=cameras,
                start_utc=start_utc,
                end_utc=end_utc,
            )
            with self._lock:
                job = self._jobs[case_id]
                job.status = view.get("status", "failed")
                job.view = view
                job.error = None
        except Exception as exc:  # noqa: BLE001 — surface to operator
            with self._lock:
                job = self._jobs[case_id]
                job.status = "failed"
                job.error = str(exc)
                job.view = {
                    "case_id": case_id,
                    "question": question,
                    "status": "failed",
                    "summary": f"Investigation failed: {exc}",
                    "scope": {
                        "cameras": list(cameras),
                        "start_utc": start_utc.isoformat(),
                        "end_utc": end_utc.isoformat(),
                    },
                    "findings": [],
                    "timeline": [],
                    "relations": [],
                    "media": [],
                    "limitations": [str(exc)],
                    "unresolved_questions": [],
                    "technical": {},
                }
        finally:
            with self._lock:
                self._busy = False

    def _execute(
        self,
        *,
        case_id: str,
        question: str,
        cameras: tuple[str, ...],
        start_utc: datetime,
        end_utc: datetime,
    ) -> dict[str, Any]:
        if self.config.tool_backend == "stub":
            return self._execute_stub(
                case_id=case_id,
                question=question,
                cameras=cameras,
                start_utc=start_utc,
                end_utc=end_utc,
            )
        return self._execute_live(
            case_id=case_id,
            question=question,
            cameras=cameras,
            start_utc=start_utc,
            end_utc=end_utc,
        )

    def _execute_stub(
        self,
        *,
        case_id: str,
        question: str,
        cameras: tuple[str, ...],
        start_utc: datetime,
        end_utc: datetime,
    ) -> dict[str, Any]:
        from datetime import timedelta

        from nanexus_investigation.graph.runtime import ok_event_result
        from nanexus_investigation.model.provider import (
            InvestigationProposal,
            StaticModelProvider,
        )

        t0 = start_utc + timedelta(minutes=5)
        if t0 >= end_utc:
            t0 = start_utc
        provider = StaticModelProvider(
            initial_proposal=InvestigationProposal(
                action="invoke_tool",
                tool_name="search_events",
                label="person",
                rationale="stub search",
                camera_ids=cameras[:1],
            ),
            proposals=[
                InvestigationProposal(
                    action="proceed_to_report",
                    tool_name=None,
                    label=None,
                    rationale="stub complete",
                )
            ],
        )
        bounds = Interval(start_utc=start_utc, end_utc=end_utc)
        case = make_case(
            case_id=case_id,
            question=question,
            cameras=cameras,
            start=start_utc,
            end=end_utc,
            vms_instance_id="vms-local-1",
        )
        media_root = self.config.media_store_root / case_id
        media_root.mkdir(parents=True, exist_ok=True)
        scenario = ScenarioConfig(
            name="web_stub",
            tool_backend="stub",
            force_time_bounds=bounds,
            use_model_initial_plan=True,
            use_model_follow_up=True,
            max_model_calls=4,
            max_tool_attempts=4,
            max_plan_steps=4,
            enable_temporal_analysis=False,
            enable_media_resolve=False,
            enable_observe_media=False,
            media_store_root=str(media_root),
            tool_scripts={
                "search_events": [
                    ok_event_result(
                        execution_id="exec-web-stub",
                        native_id="evt-web-stub",
                        label="person",
                        start=t0,
                        end=t0 + timedelta(seconds=20),
                        camera_id=cameras[0],
                    )
                ]
            },
        )
        assert self._db is not None
        runtime = InvestigationRuntime.from_durable_fixture(
            case=case,
            scenario=scenario,
            domain_db=self._db,
            model_provider=provider,
        )
        result = run_investigation(runtime, thread_id=f"web-{case_id}")
        if result.get("domain_finalized"):
            return build_operator_view(repo=self.repo, case_id=case_id)
        report = result.get("domain_report") or result.get("report")
        return build_operator_view_from_runtime(
            case_id=case_id,
            question=question,
            status=runtime.case.status.value,
            report=report,
            claims=dict(runtime.store.claims),
            observations=dict(runtime.store.observations),
            relations=dict(runtime.store.relations),
            evidence=dict(runtime.store.evidence),
            scope={
                "vms": "vms-local-1",
                "cameras": list(cameras),
                "start_utc": start_utc.isoformat(),
                "end_utc": end_utc.isoformat(),
            },
            error=result.get("finalization_error"),
        )

    def _execute_live(
        self,
        *,
        case_id: str,
        question: str,
        cameras: tuple[str, ...],
        start_utc: datetime,
        end_utc: datetime,
    ) -> dict[str, Any]:
        env = load_frigate_env(self.config.frigate_env_path)
        ready, reason = live_client_ready(env)
        if not ready:
            raise InvestigationRequestError(f"live Frigate unavailable: {reason}")

        cfg = config_from_env(env)
        client = FrigateReadOnlyClient(cfg)
        chat_ok, chat_detail = self._probe_ollama_model(self.config.chat_model)
        if not chat_ok:
            raise InvestigationRequestError(
                f"local chat model unavailable: {chat_detail}"
            )

        visual_provider = detect_local_visual_provider(
            preferred_models=(self.config.visual_model, "moondream:1.8b", "moondream"),
            base_url=self.config.ollama_base_url,
        )
        enable_observe = bool(
            self.config.enable_observe_media and visual_provider is not None
        )

        bounds = Interval(start_utc=start_utc, end_utc=end_utc)
        case = make_case(
            case_id=case_id,
            question=question,
            cameras=cameras,
            start=start_utc,
            end=end_utc,
            vms_instance_id=cfg.vms_instance_id,
        )
        media_root = self.config.media_store_root / case_id
        media_root.mkdir(parents=True, exist_ok=True)
        scenario = ScenarioConfig(
            name="web_live",
            tool_backend="frigate_live",
            force_time_bounds=bounds,
            use_model_initial_plan=True,
            use_model_follow_up=True,
            max_model_calls=self.config.max_model_calls,
            max_tool_attempts=self.config.max_tool_attempts,
            max_plan_steps=self.config.max_plan_steps,
            enable_temporal_analysis=True,
            enable_media_resolve=True,
            enable_observe_media=enable_observe,
            media_store_root=str(media_root),
            visual_model=self.config.visual_model if enable_observe else None,
            visual_base_url=self.config.ollama_base_url,
            search_label=None,
        )
        model_provider = OllamaChatProvider(
            model=self.config.chat_model,
            base_url=self.config.ollama_base_url,
            timeout_s=180.0,
        )
        assert self._db is not None
        runtime = InvestigationRuntime.from_durable_fixture(
            case=case,
            scenario=scenario,
            domain_db=self._db,
            model_provider=model_provider,
            frigate_client=client,
            visual_provider=visual_provider,
        )
        result = run_investigation(runtime, thread_id=f"web-{case_id}")
        report = result.get("domain_report") or result.get("report")
        status = runtime.case.status.value
        scope = {
            "vms": case.effective_scope.vms_instance_id,
            "cameras": list(cameras),
            "start_utc": start_utc.isoformat(),
            "end_utc": end_utc.isoformat(),
        }
        if result.get("domain_finalized") and report is not None:
            return build_operator_view(repo=self.repo, case_id=case_id)
        return build_operator_view_from_runtime(
            case_id=case_id,
            question=question,
            status=status,
            report=report,
            claims=dict(runtime.store.claims),
            observations=dict(runtime.store.observations),
            relations=dict(runtime.store.relations),
            evidence=dict(runtime.store.evidence),
            scope=scope,
            error=result.get("finalization_error"),
        )
