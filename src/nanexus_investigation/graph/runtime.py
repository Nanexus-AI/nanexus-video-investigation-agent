"""Closed-over investigation runtime for one E0 graph execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from nanexus_investigation.contracts.capability import (
    CapabilitySnapshot,
    OperationCapability,
)
from nanexus_investigation.contracts.case import InvestigationCase
from nanexus_investigation.contracts.enums import (
    CapabilityAvailability,
    CaseStatus,
    ClaimType,
    ClaimValidationStatus,
    EvidenceAvailability,
    EvidenceKind,
    ObservationType,
    RecordType,
    SearchCoverage,
    TemporalEvaluationStatus,
    TemporalRelationType,
    ToolResultStatus,
)
from nanexus_investigation.contracts.claim import Claim, InvestigationReport
from nanexus_investigation.contracts.evidence import EvidenceRef, Observation
from nanexus_investigation.contracts.identity import Interval, SourceLocator, TimeContext
from nanexus_investigation.contracts.scope import InvestigationScope
from nanexus_investigation.contracts.search import SearchExecutionReceipt
from nanexus_investigation.contracts.tool import ToolRequest, ToolResult
from nanexus_investigation.evidence.claims import assess_claim, observation_interval
from nanexus_investigation.evidence.relations import evaluate_predicate
from nanexus_investigation.evidence.search_coverage import interpret_search
from nanexus_investigation.graph.planner import PlanStep, build_deterministic_plan
from nanexus_investigation.graph.store import InMemoryCaseRepository
from nanexus_investigation.harness.budget import InMemoryBudgetLedger
from nanexus_investigation.harness.durable_budget import DurableBudgetAdapter
from nanexus_investigation.harness.output_gate import OutputGateDecision, run_output_gate
from nanexus_investigation.harness.tool_guard import (
    BudgetLedgerView,
    GuardDecision,
    ToolGuard,
)
from nanexus_investigation.harness.trace import ExecutionTrace
from nanexus_investigation.persistence.budget import DurableBudgetLedger
from nanexus_investigation.persistence.db import InvestigationDatabase
from nanexus_investigation.persistence.errors import DomainFinalizationError
from nanexus_investigation.persistence.operation_id import (
    build_attempt_identity,
    build_operation_identity,
)
from nanexus_investigation.persistence.repository import (
    AcceptedReportBinding,
    FinalizationResult,
    InvestigationRepository,
)
from nanexus_investigation.tools.frigate_backend import FrigateToolRegistry
from nanexus_investigation.tools.protocol import ToolRegistryView
from nanexus_investigation.tools.registry import StubToolRegistry

UTC = timezone.utc
ToolBackend = Literal["stub", "frigate_recorded", "frigate_live"]


@dataclass
class ScenarioConfig:
    """Controlled fixture knobs for deterministic E0 investigations."""

    name: str = "happy_path"
    include_follow_up: bool = False
    max_tool_attempts: int = 3
    max_plan_steps: int = 4
    force_tool_name: str | None = None
    force_camera_ids: tuple[str, ...] | None = None
    force_time_bounds: Interval | None = None
    tool_scripts: dict[str, list[ToolResult | dict[str, Any]]] = field(
        default_factory=dict
    )
    inject_fabricated_evidence_claim: bool = False
    inject_false_temporal_claim: bool = False
    inject_unknown_temporal_claim: bool = False
    inject_real_world_absence_claim: bool = False
    trust_producer_supported_label: bool = False
    # When true, intake interrupts for a narrowed operational time interval
    # even if effective_scope.time_bounds are already complete.
    require_time_clarification: bool = False
    # When true, assess_sufficiency consults a replaceable Model Provider for
    # evidence-dependent follow-up after authorized tool results. Default
    # remains the deterministic E0 path.
    use_model_follow_up: bool = False
    # When true, build_plan asks the Model Provider for the initial PlanStep
    # instead of the deterministic fixture planner. Default False preserves E0.
    use_model_initial_plan: bool = False
    # Shared budget for local model consultations (initial + follow-ups).
    # Default 1 preserves E1 one-shot follow-up when only use_model_follow_up.
    max_model_calls: int = 1
    # Tool body selection. Default ``stub`` preserves E0 deterministic scripts.
    # Frigate modes dispatch through FrigateToolRegistry (never mixed with stubs).
    tool_backend: ToolBackend = "stub"
    # Path to recorded envelope (``.json``) or sanitized events fixture.
    frigate_recording_path: str | None = None
    frigate_max_limit: int = 50
    # Primary search_events label. Default ``person`` matches E0 planner behavior.
    # Set to ``None`` for an unfiltered Frigate events query.
    # Ignored when use_model_initial_plan selects the label from the model.
    search_label: str | None = "person"
    # When false, omit analyze_temporal_relations from CapabilitySnapshot
    # (legacy E1 related-events stub scenarios). Default True for T3+.
    enable_temporal_analysis: bool = True
    # When false, omit resolve_media from CapabilitySnapshot. Default True for T4+.
    enable_media_resolve: bool = True
    # When true AND a visual provider is bound/available, advertise observe_media.
    # Default False so T2–T4 scenarios are unchanged unless a VLM backend is configured.
    enable_observe_media: bool = False
    # Private local root for retained media bytes (gitignored / tmp in tests).
    media_store_root: str | None = None
    # Optional recorded media directory for frigate_recorded resolve_media.
    frigate_recorded_media_dir: str | None = None
    max_snapshot_bytes: int = 5 * 1024 * 1024
    max_clip_bytes: int = 25 * 1024 * 1024
    # Optional local VLM model name for OllamaVisualProvider auto-bind.
    visual_model: str | None = None
    visual_base_url: str = "http://127.0.0.1:11434"


@dataclass
class InvestigationRuntime:
    """Mutable side store for one investigation. Not a second workflow controller."""

    case: InvestigationCase
    store: InMemoryCaseRepository
    registry: ToolRegistryView
    budget: BudgetLedgerView
    guard: ToolGuard
    trace: ExecutionTrace
    scenario: ScenarioConfig
    capabilities: CapabilitySnapshot
    plan_steps: list[PlanStep] = field(default_factory=list)
    last_guard: GuardDecision | None = None
    last_gate: OutputGateDecision | None = None
    retry_policy_attached: bool = False
    domain_repo: InvestigationRepository | None = None
    durable_budget: DurableBudgetLedger | None = None
    model_provider: Any | None = None
    # Count of completed local-model consultations (initial + follow-up).
    model_calls: int = 0
    last_model_proposal: Any | None = None
    model_proposals: list[Any] = field(default_factory=list)
    # In-investigation anti-repeat set (not durable / not cross-case).
    executed_request_fingerprints: set[str] = field(default_factory=set)
    # Planning/replanning stops that must surface as report limitations (not
    # treated as proof that every investigation objective was satisfied).
    planning_limitations: list[str] = field(default_factory=list)
    media_store: Any | None = None
    visual_provider: Any | None = None
    _last_result: ToolResult | None = None

    def note_planning_limitation(self, message: str) -> None:
        if message and message not in self.planning_limitations:
            self.planning_limitations.append(message)

    @property
    def uses_durable_persistence(self) -> bool:
        return self.domain_repo is not None

    @property
    def model_follow_up_applied(self) -> bool:
        """True after at least one model consultation (compat with E1 tests)."""
        return self.model_calls > 0

    @classmethod
    def from_fixture(
        cls,
        *,
        case: InvestigationCase,
        scenario: ScenarioConfig,
        capabilities: CapabilitySnapshot | None = None,
        model_provider: Any | None = None,
        frigate_client: Any | None = None,
        visual_provider: Any | None = None,
    ) -> "InvestigationRuntime":
        base_caps = capabilities or capabilities_for_scenario(case, scenario)
        registry = build_tool_registry(
            case=case,
            scenario=scenario,
            frigate_client=frigate_client,
        )
        budget: BudgetLedgerView = InMemoryBudgetLedger(
            max_attempts=scenario.max_tool_attempts
        )
        store = InMemoryCaseRepository(case=case)
        bind = getattr(registry, "bind_case_store", None)
        if callable(bind):
            bind(store, case.effective_scope)
        media_store = _media_store_for_scenario(case=case, scenario=scenario)
        _bind_media_registry(
            registry,
            media_store=media_store,
            case_id=case.case_id,
            scenario=scenario,
        )
        vision = _visual_provider_for_scenario(
            scenario=scenario, visual_provider=visual_provider
        )
        _bind_visual_registry(registry, visual_provider=vision)
        caps = _capabilities_with_observe(
            base_caps,
            case=case,
            scenario=scenario,
            visual_provider=vision,
        )
        guard = ToolGuard(
            registry=registry,
            budget=budget,
            effective_scope=case.effective_scope,
            capabilities=caps,
        )
        return cls(
            case=case,
            store=store,
            registry=registry,
            budget=budget,
            guard=guard,
            trace=ExecutionTrace(),
            scenario=scenario,
            capabilities=caps,
            model_provider=model_provider,
            media_store=media_store,
            visual_provider=vision,
        )

    @classmethod
    def from_durable_fixture(
        cls,
        *,
        case: InvestigationCase,
        scenario: ScenarioConfig,
        domain_db: InvestigationDatabase,
        capabilities: CapabilitySnapshot | None = None,
        frigate_client: Any | None = None,
        model_provider: Any | None = None,
        visual_provider: Any | None = None,
    ) -> "InvestigationRuntime":
        """E0 test fixture helper for the SQLite-backed investigation path.

        Not a production authentication, authorization, or admission service.
        Effective scope is trusted caller-supplied fixture data — not live VMS
        authorization. Admission commits ``created`` then ``running`` before this
        returns. Callers must not invoke the Graph until admission has completed.
        Terminal finalization occurs after Graph completion via
        ``finalize_domain_investigation``.
        """
        caps = capabilities or capabilities_for_scenario(case, scenario)
        repo = InvestigationRepository(domain_db)
        if case.status is CaseStatus.CREATED:
            case = repo.admit_and_start_case(case)
        elif case.status is CaseStatus.RUNNING:
            existing = repo.get_case(case.case_id)
            if existing is None:
                case = repo.admit_and_start_case(
                    case.model_copy(update={"status": CaseStatus.CREATED})
                )
            elif existing.status is CaseStatus.RUNNING:
                case = existing
            elif existing.status is CaseStatus.CREATED:
                case = repo.transition_case_to_running(case.case_id)
            else:
                raise ValueError(
                    f"durable fixture cannot start terminal case {case.case_id} "
                    f"({existing.status.value})"
                )
        else:
            raise ValueError(
                "durable fixture requires created/running Case for admission "
                f"(got {case.status.value})"
            )
        durable = DurableBudgetLedger(domain_db)
        durable.configure_policy(
            case_id=case.case_id,
            max_tool_attempts=scenario.max_tool_attempts,
        )
        budget = DurableBudgetAdapter(
            ledger=durable,
            case_id=case.case_id,
            max_attempts=scenario.max_tool_attempts,
        )
        registry = build_tool_registry(
            case=case,
            scenario=scenario,
            frigate_client=frigate_client,
        )
        store = InMemoryCaseRepository(case=case)
        bind = getattr(registry, "bind_case_store", None)
        if callable(bind):
            bind(store, case.effective_scope)
        media_store = _media_store_for_scenario(case=case, scenario=scenario)
        _bind_media_registry(
            registry,
            media_store=media_store,
            case_id=case.case_id,
            scenario=scenario,
        )
        vision = _visual_provider_for_scenario(
            scenario=scenario, visual_provider=visual_provider
        )
        _bind_visual_registry(registry, visual_provider=vision)
        caps = _capabilities_with_observe(
            caps,
            case=case,
            scenario=scenario,
            visual_provider=vision,
        )
        guard = ToolGuard(
            registry=registry,
            budget=budget,
            effective_scope=case.effective_scope,
            capabilities=caps,
        )
        runtime = cls(
            case=case,
            store=store,
            registry=registry,
            budget=budget,
            guard=guard,
            trace=ExecutionTrace(),
            scenario=scenario,
            capabilities=caps,
            domain_repo=repo,
            durable_budget=durable,
            model_provider=model_provider,
            media_store=media_store,
            visual_provider=vision,
        )
        runtime.trace.record(
            "domain_case_admitted",
            case_id=case.case_id,
            status=case.status.value,
        )
        return runtime

    def build_plan(self) -> list[PlanStep]:
        self.plan_steps = build_deterministic_plan(
            scenario=self.scenario.name,
            authorized_cameras=self.case.effective_scope.camera_ids,
            time_bounds=self.case.effective_scope.time_bounds,
            force_tool_name=self.scenario.force_tool_name,
            force_camera_ids=self.scenario.force_camera_ids,
            force_time_bounds=self.scenario.force_time_bounds,
            include_follow_up=self.scenario.include_follow_up,
            search_label=self.scenario.search_label,
        )
        return self.plan_steps


def _temporal_capability(cameras: frozenset[str]) -> OperationCapability:
    """Local deterministic analysis — always available when the engine is bound."""
    return OperationCapability(
        operation="analyze_temporal_relations",
        availability=CapabilityAvailability.SUPPORTED,
        authorized_camera_ids=cameras,
    )


def _media_capability(cameras: frozenset[str]) -> OperationCapability:
    """Authorized selective media retrieval for Case-bound evidence."""
    return OperationCapability(
        operation="resolve_media",
        availability=CapabilityAvailability.SUPPORTED,
        authorized_camera_ids=cameras,
    )


def _observe_capability(cameras: frozenset[str]) -> OperationCapability:
    """Local VLM observation over retained Case media."""
    return OperationCapability(
        operation="observe_media",
        availability=CapabilityAvailability.SUPPORTED,
        authorized_camera_ids=cameras,
    )


def default_capabilities(
    vms_instance_id: str,
    camera_ids: tuple[str, ...],
) -> CapabilitySnapshot:
    cameras = frozenset(camera_ids)
    return CapabilitySnapshot(
        vms_instance_id=vms_instance_id,
        operations=(
            OperationCapability(
                operation="search_events",
                availability=CapabilityAvailability.SUPPORTED,
                authorized_camera_ids=cameras,
            ),
            OperationCapability(
                operation="search_related_events",
                availability=CapabilityAvailability.SUPPORTED,
                authorized_camera_ids=cameras,
            ),
            _temporal_capability(cameras),
            _media_capability(cameras),
        ),
    )


def frigate_capabilities(
    vms_instance_id: str,
    camera_ids: tuple[str, ...],
) -> CapabilitySnapshot:
    """Frigate read-only adapter: search_events + temporal + media resolve."""
    cameras = frozenset(camera_ids)
    return CapabilitySnapshot(
        vms_instance_id=vms_instance_id,
        operations=(
            OperationCapability(
                operation="search_events",
                availability=CapabilityAvailability.SUPPORTED,
                authorized_camera_ids=cameras,
            ),
            OperationCapability(
                operation="search_related_events",
                availability=CapabilityAvailability.UNSUPPORTED,
                authorized_camera_ids=cameras,
            ),
            _temporal_capability(cameras),
            _media_capability(cameras),
        ),
    )


def capabilities_for_scenario(
    case: InvestigationCase,
    scenario: ScenarioConfig,
) -> CapabilitySnapshot:
    if scenario.tool_backend == "stub":
        caps = default_capabilities(
            case.effective_scope.vms_instance_id,
            case.effective_scope.camera_ids,
        )
    else:
        caps = frigate_capabilities(
            case.effective_scope.vms_instance_id,
            case.effective_scope.camera_ids,
        )
    operations = caps.operations
    if not scenario.enable_temporal_analysis:
        operations = tuple(
            op for op in operations if op.operation != "analyze_temporal_relations"
        )
    if not scenario.enable_media_resolve:
        operations = tuple(op for op in operations if op.operation != "resolve_media")
    # observe_media is never advertised from the feature flag alone.
    # Runtime construction adds it only when an executable VLM backend is available.
    operations = tuple(op for op in operations if op.operation != "observe_media")
    if operations == caps.operations:
        return caps
    return caps.model_copy(update={"operations": operations})


def _media_store_for_scenario(
    *,
    case: InvestigationCase,
    scenario: ScenarioConfig,
) -> Any | None:
    if not scenario.enable_media_resolve:
        return None
    from nanexus_investigation.evidence.media_store import LocalMediaStore

    if scenario.media_store_root:
        root = Path(scenario.media_store_root)
    else:
        root = Path(".local") / "media_store" / case.case_id
    return LocalMediaStore(root)


def _visual_provider_for_scenario(
    *,
    scenario: ScenarioConfig,
    visual_provider: Any | None,
) -> Any | None:
    if visual_provider is not None:
        return visual_provider
    if not scenario.enable_observe_media:
        return None
    if scenario.visual_model:
        from nanexus_investigation.model.visual_provider import OllamaVisualProvider

        provider = OllamaVisualProvider(
            model=scenario.visual_model,
            base_url=scenario.visual_base_url,
        )
        return provider if provider.is_available() else provider
    from nanexus_investigation.model.visual_provider import detect_local_visual_provider

    return detect_local_visual_provider(base_url=scenario.visual_base_url)


def _bind_visual_registry(registry: Any, *, visual_provider: Any | None) -> None:
    bind = getattr(registry, "bind_visual_provider", None)
    if callable(bind):
        bind(visual_provider)


def _capabilities_with_observe(
    caps: CapabilitySnapshot,
    *,
    case: InvestigationCase,
    scenario: ScenarioConfig,
    visual_provider: Any | None,
) -> CapabilitySnapshot:
    """Advertise observe_media only when flag is on AND backend is executable."""
    operations = tuple(
        op for op in caps.operations if op.operation != "observe_media"
    )
    available = False
    if scenario.enable_observe_media and visual_provider is not None:
        checker = getattr(visual_provider, "is_available", None)
        available = bool(checker()) if callable(checker) else False
    if available:
        cameras = frozenset(case.effective_scope.camera_ids)
        operations = operations + (_observe_capability(cameras),)
    if operations == caps.operations:
        return caps
    return caps.model_copy(update={"operations": operations})


def _bind_media_registry(
    registry: Any,
    *,
    media_store: Any | None,
    case_id: str,
    scenario: ScenarioConfig,
) -> None:
    bind = getattr(registry, "bind_media_store", None)
    if callable(bind) and media_store is not None:
        bind(media_store, case_id=case_id)
    if isinstance(registry, FrigateToolRegistry):
        registry.max_snapshot_bytes = scenario.max_snapshot_bytes
        registry.max_clip_bytes = scenario.max_clip_bytes
        if scenario.frigate_recorded_media_dir:
            registry.recorded_media_dir = Path(scenario.frigate_recorded_media_dir)


def build_tool_registry(
    *,
    case: InvestigationCase,
    scenario: ScenarioConfig,
    frigate_client: Any | None = None,
) -> ToolRegistryView:
    """Select stub vs Frigate recorded/live registry. Never mixes stub scripts with Frigate."""
    if scenario.tool_backend == "stub":
        return StubToolRegistry(scripts=dict(scenario.tool_scripts))

    vms_id = case.effective_scope.vms_instance_id
    if scenario.tool_backend == "frigate_recorded":
        if not scenario.frigate_recording_path:
            raise ValueError(
                "frigate_recorded backend requires scenario.frigate_recording_path"
            )
        recorded_media = (
            Path(scenario.frigate_recorded_media_dir)
            if scenario.frigate_recorded_media_dir
            else None
        )
        return FrigateToolRegistry(
            mode="recorded",
            vms_instance_id=vms_id,
            max_limit=scenario.frigate_max_limit,
            recorded_source=Path(scenario.frigate_recording_path),
            recorded_media_dir=recorded_media,
            max_snapshot_bytes=scenario.max_snapshot_bytes,
            max_clip_bytes=scenario.max_clip_bytes,
        )

    if scenario.tool_backend == "frigate_live":
        client = frigate_client
        if client is None:
            from nanexus_investigation.vms.frigate.record import (
                config_from_env,
                live_client_ready,
                load_frigate_env,
            )
            from nanexus_investigation.vms.frigate.client import FrigateReadOnlyClient

            env = load_frigate_env()
            ready, reason = live_client_ready(env)
            if not ready:
                raise RuntimeError(
                    f"frigate_live backend unavailable: {reason}"
                )
            cfg = _client_config_for_case(
                config_from_env(env), case, scenario
            )
            client = FrigateReadOnlyClient(cfg)
        return FrigateToolRegistry(
            mode="live",
            vms_instance_id=vms_id,
            max_limit=scenario.frigate_max_limit,
            client=client,
            max_snapshot_bytes=scenario.max_snapshot_bytes,
            max_clip_bytes=scenario.max_clip_bytes,
        )

    raise ValueError(f"unknown tool_backend: {scenario.tool_backend}")


def _client_config_for_case(cfg: Any, case: InvestigationCase, scenario: ScenarioConfig):
    from nanexus_investigation.vms.frigate.client import FrigateClientConfig

    return FrigateClientConfig(
        base_url=cfg.base_url,
        vms_instance_id=case.effective_scope.vms_instance_id,
        username=cfg.username,
        password=cfg.password,
        bearer_token=cfg.bearer_token,
        session_cookie=cfg.session_cookie,
        ca_bundle=cfg.ca_bundle,
        disable_hostname_check=cfg.disable_hostname_check,
        timeout_s=cfg.timeout_s,
        authorized_cameras=case.effective_scope.camera_ids,
        max_limit=scenario.frigate_max_limit,
    )


def make_case(
    *,
    case_id: str = "case-e0-1",
    question: str = "Find person activity near the driveway.",
    cameras: tuple[str, ...] = ("front_yard",),
    start: datetime | None = None,
    end: datetime | None = None,
    vms_instance_id: str = "vms-local-1",
) -> InvestigationCase:
    start_utc = start or datetime(2026, 11, 19, 20, 0, tzinfo=UTC)
    end_utc = end or datetime(2026, 11, 19, 23, 0, tzinfo=UTC)
    scope = InvestigationScope(
        vms_instance_id=vms_instance_id,
        camera_ids=cameras,
        time_bounds=Interval(start_utc=start_utc, end_utc=end_utc),
    )
    return InvestigationCase(
        case_id=case_id,
        question=question,
        requested_scope=scope,
        effective_scope=scope,
        time_context=TimeContext(
            timezone="America/New_York",
            start_utc=start_utc,
            end_utc=end_utc,
            original_expression="yesterday 8 PM to 11 PM",
        ),
        status=CaseStatus.CREATED,
        policy_ref="policy:e0-deterministic",
        created_at=datetime(2026, 11, 20, 12, 0, tzinfo=UTC),
        budget_profile_ref="budget:e0-inmemory",
    )


def ok_event_result(
    *,
    execution_id: str,
    native_id: str,
    label: str,
    start: datetime,
    end: datetime,
    camera_id: str = "front_yard",
    source_id: str = "vms-local-1",
) -> ToolResult:
    return ToolResult(
        status=ToolResultStatus.OK,
        execution_id=execution_id,
        data=[
            {
                "native_record_id": native_id,
                "record_type": RecordType.EVENT,
                "camera_id": camera_id,
                "label": label,
                "start_utc": start,
                "end_utc": end,
                "source_revision": "rev-1",
            }
        ],
        source_id=source_id,
        retrieved_at=datetime(2026, 11, 20, 12, 5, tzinfo=UTC),
        coverage=SearchCoverage.EXHAUSTED,
        provenance=f"stub:search/{native_id}",
        usage={"tool_attempts": 1},
    )


def empty_result(
    *,
    execution_id: str = "exec-empty-1",
    coverage: SearchCoverage = SearchCoverage.EXHAUSTED,
) -> ToolResult:
    return ToolResult(
        status=ToolResultStatus.EMPTY,
        execution_id=execution_id,
        data=[],
        source_id="vms-local-1",
        retrieved_at=datetime(2026, 11, 20, 12, 5, tzinfo=UTC),
        coverage=coverage,
        provenance="stub:empty",
        usage={"tool_attempts": 1},
    )


class DomainRegistrationError(RuntimeError):
    """Durable domain write failed; graph must fail closed without accepting IDs."""


def finalize_domain_investigation(
    runtime: InvestigationRuntime,
) -> FinalizationResult:
    """Commit authoritative terminal Case status (and report when applicable).

    Completed/partial paths re-run the Output Gate against the candidate report
    and bind that exact accepted content before domain persistence. A Graph
    checkpoint alone is not durable finalization. Failure without a valid
    partial report records Case status ``failed`` with no InvestigationReport.
    """
    if runtime.domain_repo is None:
        raise DomainFinalizationError("finalize_domain_investigation requires domain_repo")

    repo = runtime.domain_repo
    domain_case = repo.get_case(runtime.case.case_id)
    if domain_case is None:
        raise DomainFinalizationError(
            f"case {runtime.case.case_id} is not admitted for finalization"
        )

    accepted_binding: AcceptedReportBinding | None = None
    terminal_status = CaseStatus.FAILED
    gate = runtime.last_gate
    candidate = None if gate is None else gate.report
    if (
        gate is not None
        and gate.accepted
        and candidate is not None
        and gate.case_status in {CaseStatus.COMPLETED, CaseStatus.PARTIAL}
        and candidate.case_id == domain_case.case_id
    ):
        # Independently re-validate the candidate; do not trust a stale boolean.
        rebound = run_output_gate(
            case=domain_case,
            report=candidate,
            claims_by_id=runtime.store.claims,
            assessments=runtime.store.assessments,
            evidence_ids=frozenset(runtime.store.evidence),
            observation_ids=frozenset(runtime.store.observations),
            relation_ids=frozenset(runtime.store.relations),
            search_receipt_ids=frozenset(runtime.store.receipts),
            evidence_by_id=runtime.store.evidence,
            observations_by_id=runtime.store.observations,
            relations_by_id=runtime.store.relations,
            receipts_by_id=runtime.store.receipts,
            tool_results_by_receipt_id=runtime.store.tool_results_by_receipt,
            observation_intervals=runtime.store.observation_intervals,
        )
        if (
            rebound.accepted
            and rebound.report is not None
            and rebound.case_status in {CaseStatus.COMPLETED, CaseStatus.PARTIAL}
            and rebound.report.case_id == domain_case.case_id
            and rebound.report.report_id == candidate.report_id
        ):
            accepted_binding = AcceptedReportBinding.from_gate_report(
                case_id=domain_case.case_id,
                terminal_status=rebound.case_status,
                report=rebound.report,
            )
            terminal_status = rebound.case_status
        elif gate.accepted:
            raise DomainFinalizationError(
                "Output Gate rebound rejected or altered the candidate report; "
                "refusing completed/partial finalization"
            )

    if accepted_binding is not None:
        result = repo.finalize_case(
            case_id=domain_case.case_id,
            terminal_status=terminal_status,
            accepted=accepted_binding,
            claims_by_id=dict(runtime.store.claims),
            relations_by_id=dict(runtime.store.relations),
        )
    else:
        # No Output-Gate-accepted report: durable failure without fabricating one.
        result = repo.finalize_case(
            case_id=domain_case.case_id,
            terminal_status=CaseStatus.FAILED,
            accepted=None,
        )

    runtime.case = result.case
    runtime.store.case = result.case
    if result.report is not None:
        runtime.store.report = result.report
    runtime.trace.record(
        "domain_case_finalized",
        case_id=result.case.case_id,
        terminal_status=result.terminal_status.value,
        report_id=None if result.report is None else result.report.report_id,
        reused=result.reused,
        domain_finalized=True,
    )
    return result


def register_tool_artifacts(
    runtime: InvestigationRuntime,
    *,
    request: ToolRequest,
    result: ToolResult,
    plan_step_id: str | None = None,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Create Observation/EvidenceRef/receipt only for validated non-empty results.

    Durable path ordering (fail closed):
    1. Validate ToolResult status / build candidate artifacts.
    2. Commit the logical registration group atomically in Investigation Persistence
       (Observation/EvidenceRef/Receipt as applicable, plus group marker).
    3. Only then register in-memory and return IDs to graph state.

    A domain write failure rolls back the registration group, discards in-memory
    candidates for this attempt, and raises ``DomainRegistrationError``. Graph
    state never receives IDs for uncommitted domain artifacts. Domain and
    Checkpointer transactions remain independent authorities.
    """
    store = runtime.store
    observation_ids: list[str] = []
    evidence_ids: list[str] = []
    receipt_ids: list[str] = []
    coverage_notes: list[str] = []

    is_temporal = request.tool_name == "analyze_temporal_relations"
    is_media = request.tool_name == "resolve_media"
    is_observe = request.tool_name == "observe_media"
    match_count = 0
    if isinstance(result.data, list):
        match_count = len(result.data)

    time_bounds = request.arguments.get("time_bounds") or {}
    if isinstance(time_bounds, dict) and "start_utc" in time_bounds:
        start = time_bounds["start_utc"]
        end = time_bounds["end_utc"]
        if isinstance(start, str):
            start = datetime.fromisoformat(start)
        if isinstance(end, str):
            end = datetime.fromisoformat(end)
        bounds = Interval(start_utc=start, end_utc=end)
    else:
        bounds = runtime.case.effective_scope.time_bounds

    usage = dict(result.usage or {})
    page_limit_raw = usage.get("page_limit", request.arguments.get("limit", 100))
    try:
        page_limit = int(page_limit_raw)
    except (TypeError, ValueError):
        page_limit = 100
    consistency = None
    if (result.provenance or "").startswith("adapter:frigate-read"):
        consistency = (
            "adapter:frigate-read; has_clip/has_snapshot are source indicators "
            "only and do not prove media were fetched"
        )
    if is_temporal:
        consistency = (
            "temporal-engine: derived relations are deterministic computations "
            "over registered observations; proximity is not interaction or identity"
        )
    if is_media:
        consistency = (
            "adapter:frigate-media; acquired media bytes and content hash only — "
            "not visual interpretation, scene authenticity, or proof of event occurrence"
        )
    if is_observe:
        consistency = (
            "adapter:local-vlm; local visual model interpretation over retained "
            "media — not an independently verified physical-world fact, identity, "
            "ownership, causation, or interaction"
        )

    filters: dict[str, Any] = {"label": request.arguments.get("label")}
    if is_temporal:
        usage_meta = result.usage or {}
        filters = {
            "predicate": request.arguments.get("predicate", "near"),
            "window_seconds": request.arguments.get("window_seconds"),
            "anchor_label": request.arguments.get("anchor_label"),
            "candidate_label": request.arguments.get("candidate_label"),
            "max_pairs": request.arguments.get("max_pairs", usage_meta.get("max_pairs")),
            "pairs_evaluated": usage_meta.get("pairs_evaluated"),
            "truncated": bool(usage_meta.get("truncated")),
        }
        if result.coverage is SearchCoverage.TRUNCATED:
            consistency = (
                f"{consistency}; pair evaluation truncated at max_pairs — "
                "established relations are not exhaustive and absence of "
                "additional NEAR matches is not proven"
            )
    if is_media:
        filters = {
            "media_kind": request.arguments.get("media_kind"),
            "evidence_id": request.arguments.get("evidence_id"),
            "observation_id": request.arguments.get("observation_id"),
        }
    if is_observe:
        observe_record: dict[str, Any] = {}
        if (
            result.status is ToolResultStatus.OK
            and isinstance(result.data, list)
            and result.data
            and isinstance(result.data[0], dict)
        ):
            observe_record = result.data[0]
        filters = {
            "evidence_id": request.arguments.get("evidence_id"),
            "objective": request.arguments.get("objective"),
            "media_evidence_id": observe_record.get("media_evidence_id"),
            "retained_media_id": observe_record.get("retained_media_id"),
            "content_sha256": observe_record.get("content_sha256"),
            "source_content_sha256": observe_record.get("source_content_sha256"),
            "inference_input_sha256": observe_record.get("inference_input_sha256"),
            "inference_input_byte_count": observe_record.get(
                "inference_input_byte_count"
            ),
            "inference_input_width": observe_record.get("inference_input_width"),
            "inference_input_height": observe_record.get("inference_input_height"),
            "preprocessing": observe_record.get("preprocessing"),
            "preprocessing_max_edge": observe_record.get("preprocessing_max_edge"),
            "model_name": observe_record.get("model_name"),
            "model_digest": observe_record.get("model_digest"),
            "prompt_template_id": observe_record.get("prompt_template_id"),
            "prompt_text": observe_record.get("prompt_text"),
            "used_fallback_prompt": observe_record.get("used_fallback_prompt"),
            "temperature": observe_record.get("temperature"),
            "num_predict": observe_record.get("num_predict"),
            "outcome": observe_record.get("outcome"),
            "observation_text": observe_record.get("observation_text"),
            "uncertainty": observe_record.get("uncertainty"),
            "physical_world_fact": False,
        }

    receipt = SearchExecutionReceipt(
        search_execution_id=f"receipt-{result.execution_id}",
        tool_name=request.tool_name,
        source_id=request.source_id or runtime.case.effective_scope.vms_instance_id,
        camera_ids=request.camera_ids or runtime.case.effective_scope.camera_ids,
        time_bounds=bounds,
        filters=filters,
        retrieved_at=result.retrieved_at
        or datetime.now(tz=UTC),
        pages_fetched=1,
        page_limit=page_limit,
        match_count=match_count,
        coverage=result.coverage or SearchCoverage.UNKNOWN,
        source_capability=request.tool_name,
        consistency_limitations=consistency,
        error=(
            result.error
            if result.status
            in {ToolResultStatus.ERROR, ToolResultStatus.UNSUPPORTED}
            else None
        ),
    )
    # ERROR/UNSUPPORTED receipts must not claim exhausted empty semantics with matches.
    if result.status in {ToolResultStatus.ERROR, ToolResultStatus.UNSUPPORTED}:
        receipt = receipt.model_copy(
            update={
                "match_count": 0,
                "coverage": result.coverage or SearchCoverage.UNKNOWN,
            }
        )

    interpretation = interpret_search(result, receipt)
    coverage_notes.append(interpretation.summary)
    runtime.trace.record(
        "search_interpretation",
        execution_id=result.execution_id,
        status=result.status.value,
        allows_real_world_absence=interpretation.allows_real_world_absence_statement,
        summary=interpretation.summary,
        temporal=is_temporal,
        media=is_media,
    )

    observations: list[Observation] = []
    evidences: list[EvidenceRef] = []
    media_evidences: list[EvidenceRef] = []
    # Temporal/media/observe tools never fabricate VMS detection observations.
    if (
        not is_temporal
        and not is_media
        and not is_observe
        and result.status
        not in {
            ToolResultStatus.EMPTY,
            ToolResultStatus.ERROR,
            ToolResultStatus.UNSUPPORTED,
        }
    ):
        records = result.data if isinstance(result.data, list) else []
        seen_native_ids: set[str] = set()
        for index, record in enumerate(records):
            native_id = str(record["native_record_id"])
            # Duplicate native IDs in one ToolResult must not create duplicate evidence.
            if native_id in seen_native_ids:
                continue
            seen_native_ids.add(native_id)
            locator = SourceLocator(
                vms_instance_id=request.source_id
                or runtime.case.effective_scope.vms_instance_id,
                record_type=str(record.get("record_type", RecordType.EVENT)),
                native_record_id=native_id,
            )
            start = record["start_utc"]
            end = record.get("end_utc")
            if isinstance(start, str):
                start = datetime.fromisoformat(start)
            if isinstance(end, str):
                end = datetime.fromisoformat(end)
            # Preserve missing end_utc; do not invent a completed interval.
            obs_id = f"obs-{result.execution_id}-{index}"
            observation = Observation(
                observation_id=obs_id,
                source_locator=locator,
                source_revision=str(record.get("source_revision", "rev-1")),
                observation_type=ObservationType.DETECTION_EVENT,
                camera_id=str(record["camera_id"]),
                start_utc=start,
                end_utc=end,
                labels=(str(record["label"]),) if record.get("label") else (),
                retrieved_at=result.retrieved_at or datetime.now(tz=UTC),
                provenance=result.provenance or f"stub:{request.tool_name}",
            )
            retrieval_spec: dict[str, Any] = {
                "operation": request.tool_name,
                "native_id": native_id,
            }
            # Media indicators are metadata only — not proof that media were fetched.
            if "has_clip" in record:
                retrieval_spec["has_clip_indicator"] = bool(record.get("has_clip"))
            if "has_snapshot" in record:
                retrieval_spec["has_snapshot_indicator"] = bool(
                    record.get("has_snapshot")
                )
            evidence = EvidenceRef(
                evidence_id=f"ev-{result.execution_id}-{index}",
                source_locator=locator,
                evidence_kind=EvidenceKind.EVENT_METADATA,
                camera_id=observation.camera_id,
                interval=Interval(start_utc=start, end_utc=end),
                retrieval_spec=retrieval_spec,
                availability=EvidenceAvailability.AVAILABLE,
                provenance=result.provenance or f"stub:{request.tool_name}",
            )
            observations.append(observation)
            evidences.append(evidence)

    relations: list[Any] = []
    if (
        is_temporal
        and result.status is ToolResultStatus.OK
        and isinstance(result.data, list)
    ):
        from nanexus_investigation.tools.temporal_analysis import (
            materialize_relations_from_tool_data,
        )

        relations = materialize_relations_from_tool_data(result.data)

    if (
        is_media
        and result.status is ToolResultStatus.OK
        and isinstance(result.data, list)
        and result.data
    ):
        from nanexus_investigation.tools.media_resolve import (
            materialize_media_evidence_from_tool_data,
        )

        source_evidence_id = result.data[0].get("source_evidence_id")
        source_evidence = (
            store.evidence.get(str(source_evidence_id))
            if source_evidence_id is not None
            else None
        )
        if source_evidence is None:
            raise DomainRegistrationError(
                "resolve_media OK result missing Case-bound source EvidenceRef"
            )
        media_evidences = materialize_media_evidence_from_tool_data(
            result.data,
            execution_id=result.execution_id,
            source_evidence=source_evidence,
        )

    if is_temporal and result.coverage is SearchCoverage.TRUNCATED:
        coverage_notes.append(
            "Temporal pair evaluation stopped at max_pairs before all eligible "
            "pairs were examined; established relations are not exhaustive, and "
            "absence of additional NEAR matches is not proven."
        )

    # Durable path: atomic domain group commit before in-memory / graph IDs.
    committed_receipt = receipt
    committed_observations = observations
    committed_evidences = evidences
    committed_relations = relations
    committed_media = media_evidences
    if runtime.domain_repo is not None:
        operation = _operation_for_request(
            runtime, request=request, plan_step_id=plan_step_id
        )
        try:
            if (
                is_observe
                and result.status is ToolResultStatus.OK
                and isinstance(result.data, list)
                and result.data
            ):
                media_id = str(result.data[0].get("media_evidence_id") or "")
                if not media_id or runtime.domain_repo.get_evidence(media_id) is None:
                    raise DomainRegistrationError(
                        "observe_media requires durable media EvidenceRef before "
                        "registration"
                    )
            if media_evidences:
                group = runtime.domain_repo.register_artifact_group(
                    operation=operation,
                    receipt=receipt,
                    media_evidences=tuple(media_evidences),
                    relations=tuple(relations),
                )
            else:
                group = runtime.domain_repo.register_artifact_group(
                    operation=operation,
                    receipt=receipt,
                    observations=tuple(observations),
                    evidences=tuple(evidences),
                    relations=tuple(relations),
                )
            committed_receipt = group.receipt
            committed_observations = list(group.observations)
            if media_evidences:
                committed_media = list(group.evidences)
                committed_evidences = []
            else:
                committed_evidences = list(group.evidences)
            committed_relations = list(group.relations)
            runtime.trace.record(
                "domain_receipt_persisted",
                receipt_id=group.receipt_id,
                operation_id=group.operation_id,
                reused=group.reused,
            )
            if group.observations:
                for observation_item, evidence_item in zip(
                    group.observations, group.evidences, strict=True
                ):
                    runtime.trace.record(
                        "domain_evidence_persisted",
                        observation_id=observation_item.observation_id,
                        evidence_id=evidence_item.evidence_id,
                        operation_id=group.operation_id,
                        source_revision=observation_item.source_revision,
                        reused=group.reused,
                    )
            for relation_item in group.relations:
                runtime.trace.record(
                    "domain_temporal_relation_persisted",
                    relation_id=relation_item.relation_id,
                    operation_id=group.operation_id,
                    left=relation_item.left_observation_id,
                    right=relation_item.right_observation_id,
                    relation_type=relation_item.relation_type.value,
                    window_seconds=relation_item.window_seconds,
                    reused=group.reused,
                )
            for media_item in committed_media:
                runtime.trace.record(
                    "domain_media_evidence_persisted",
                    evidence_id=media_item.evidence_id,
                    operation_id=group.operation_id,
                    evidence_kind=media_item.evidence_kind.value,
                    content_sha256=media_item.content_sha256,
                    retained_media_id=media_item.retrieval_spec.get(
                        "retained_media_id"
                    ),
                    reused=group.reused,
                )
            runtime.trace.record(
                "domain_registration_group_committed",
                operation_id=group.operation_id,
                group_fingerprint=group.group_fingerprint,
                observation_count=len(group.observations),
                evidence_count=len(group.evidences),
                relation_count=len(group.relations),
                reused=group.reused,
            )
        except Exception as exc:
            # Fail closed: never leave this attempt's candidates in the working store.
            for observation in observations:
                store.observations.pop(observation.observation_id, None)
                store.observation_intervals.pop(observation.observation_id, None)
            for evidence in evidences:
                store.evidence.pop(evidence.evidence_id, None)
            for evidence in media_evidences:
                store.evidence.pop(evidence.evidence_id, None)
            for relation in relations:
                store.relations.pop(relation.relation_id, None)
            store.receipts.pop(receipt.search_execution_id, None)
            store.tool_results_by_receipt.pop(receipt.search_execution_id, None)
            store.tool_results.pop(result.execution_id, None)
            runtime.trace.record(
                "domain_registration_failed",
                execution_id=result.execution_id,
                error=str(exc),
            )
            raise DomainRegistrationError(
                f"domain evidence registration failed closed: {exc}"
            ) from exc

    # In-memory registration only after durable success (or in-memory-only path).
    store.register_receipt(committed_receipt)
    store.register_tool_result(result, receipt_id=committed_receipt.search_execution_id)
    receipt_ids.append(committed_receipt.search_execution_id)

    if result.status in {
        ToolResultStatus.EMPTY,
        ToolResultStatus.ERROR,
        ToolResultStatus.UNSUPPORTED,
    }:
        if is_temporal and result.status is ToolResultStatus.EMPTY:
            if result.coverage is SearchCoverage.TRUNCATED:
                coverage_notes.append(
                    "Temporal pair evaluation was truncated at max_pairs with no "
                    "established NEAR relations among evaluated pairs; absence of "
                    "NEAR matches among unevaluated pairs is not proven."
                )
            else:
                coverage_notes.append(
                    "Temporal analysis found no established NEAR relations for the "
                    "selected authorized observation pairs; this is not real-world absence."
                )
        if is_media and result.status is ToolResultStatus.EMPTY:
            coverage_notes.append(
                "Authorized media resolve found the requested snapshot/clip "
                "unavailable; this is not proof the source event did not occur "
                "and is not visual interpretation."
            )
        if is_observe and result.status in {
            ToolResultStatus.EMPTY,
            ToolResultStatus.ERROR,
            ToolResultStatus.UNSUPPORTED,
        }:
            coverage_notes.append(
                "Local visual observation did not produce a usable interpretation; "
                "no visual finding was registered as a physical-world fact."
            )
        return observation_ids, evidence_ids, receipt_ids, coverage_notes

    for observation, evidence in zip(
        committed_observations, committed_evidences, strict=True
    ):
        store.register_observation(observation)
        store.register_evidence(evidence)
        store.observation_intervals[observation.observation_id] = observation_interval(
            observation
        )
        observation_ids.append(observation.observation_id)
        evidence_ids.append(evidence.evidence_id)
        runtime.trace.record(
            "evidence_registered",
            observation_id=observation.observation_id,
            evidence_id=evidence.evidence_id,
            native_record_id=evidence.source_locator.native_record_id,
        )

    for media_evidence in committed_media:
        store.register_evidence(media_evidence)
        evidence_ids.append(media_evidence.evidence_id)
        runtime.trace.record(
            "media_evidence_registered",
            evidence_id=media_evidence.evidence_id,
            evidence_kind=media_evidence.evidence_kind.value,
            content_sha256=media_evidence.content_sha256,
            retained_media_id=media_evidence.retrieval_spec.get("retained_media_id"),
            byte_count=media_evidence.retrieval_spec.get("byte_count"),
            visual_interpretation=False,
        )

    for relation in committed_relations:
        if relation.relation_id in store.relations:
            continue
        store.register_relation(relation)
        runtime.trace.record(
            "temporal_relation_registered",
            relation_id=relation.relation_id,
            left=relation.left_observation_id,
            right=relation.right_observation_id,
            relation_type=relation.relation_type.value,
            gap_seconds=relation.gap_seconds,
            window_seconds=relation.window_seconds,
        )

    return observation_ids, evidence_ids, receipt_ids, coverage_notes


def _operation_for_request(
    runtime: InvestigationRuntime,
    *,
    request: ToolRequest,
    plan_step_id: str | None,
):
    from nanexus_investigation.persistence.operation_id import OperationIdentity

    step_id = plan_step_id or f"anon-{request.tool_name}"
    identity: OperationIdentity = build_operation_identity(
        case_id=runtime.case.case_id,
        plan_step_id=step_id,
        tool_name=request.tool_name,
        vms_instance_id=request.source_id
        or runtime.case.effective_scope.vms_instance_id,
        request_parameters={
            "arguments": request.arguments,
            "camera_ids": list(request.camera_ids),
        },
    )
    return identity


def evaluate_temporal(runtime: InvestigationRuntime) -> list[str]:
    """Evaluate BEFORE from validated Observation intervals via V1-C predicates.

    Observation count alone never establishes a relation. Intervals come from
    registered Observation timestamps (``observation_intervals``). Materialization
    uses ``TemporalPredicateResult.to_relation``:

    - ``holds=True`` → established relation registered
    - ``holds=False`` → no relation (decidable false; distinct from unknown)
    - ``holds=None`` → non-established indeterminate/unsupported relation may be
      recorded; never promoted to established

    When the model-selected ``analyze_temporal_relations`` tool already produced
    a durable receipt for this Case, skip the automatic first-pair BEFORE so the
    tool's NEAR results remain the authoritative temporal analysis.

    The graph runs this node after every tool registration, so an earlier
    search_events step may already have materialized an in-memory BEFORE
    relation. Those automatic BEFORE relations are not durable registration
    members; purge them once the temporal tool owns the Case so the accepted
    report cannot cite undurable relationship IDs alongside durable NEAR.
    """
    if any(
        receipt.tool_name == "analyze_temporal_relations"
        for receipt in runtime.store.receipts.values()
    ):
        purged: list[str] = []
        for relation_id, relation in list(runtime.store.relations.items()):
            if (
                relation.relation_type is TemporalRelationType.BEFORE
                and relation_id.endswith("-before")
            ):
                runtime.store.relations.pop(relation_id, None)
                purged.append(relation_id)
        existing = sorted(runtime.store.relations.keys())
        runtime.trace.record(
            "temporal_eval_skipped",
            reason="analyze_temporal_relations_receipt_present",
            purged_before_relation_ids=purged,
            relation_ids=existing,
        )
        return existing

    obs_ids = list(runtime.store.observations.keys())
    if len(obs_ids) < 2:
        return []
    left_id, right_id = obs_ids[0], obs_ids[1]
    left = runtime.store.observation_intervals[left_id]
    right = runtime.store.observation_intervals[right_id]
    predicate = evaluate_predicate(left, right, TemporalRelationType.BEFORE)
    relation = predicate.to_relation(
        relation_id=f"rel-{left_id}-{right_id}-before",
        left_observation_id=left_id,
        right_observation_id=right_id,
        camera_relation="same_or_authorized_scope",
    )
    if relation is None:
        # Decidable-false: distinguishable from indeterminate (holds=None).
        runtime.trace.record(
            "temporal_eval",
            holds=False,
            evaluation_status=predicate.evaluation_status.value,
            reason=predicate.reason,
            materialized=False,
            relation_type="before",
            left=left_id,
            right=right_id,
        )
        return []
    if (
        predicate.holds is True
        and relation.evaluation_status is not TemporalEvaluationStatus.ESTABLISHED
    ):
        raise RuntimeError("true BEFORE predicate must materialize as established")
    if (
        predicate.holds is not True
        and relation.evaluation_status is TemporalEvaluationStatus.ESTABLISHED
    ):
        raise RuntimeError("non-true BEFORE predicate must not materialize as established")
    runtime.store.register_relation(relation)
    runtime.trace.record(
        "temporal_eval",
        holds=predicate.holds,
        evaluation_status=relation.evaluation_status.value,
        reason=predicate.reason,
        materialized=True,
        relation_id=relation.relation_id,
        left=left_id,
        right=right_id,
    )
    return [relation.relation_id]


def draft_candidate_claims(runtime: InvestigationRuntime) -> list[str]:
    """Deterministic claim stub. Producer labels are not evidence authority."""
    claim_ids: list[str] = []
    store = runtime.store
    now = datetime(2026, 11, 20, 12, 10, tzinfo=UTC)

    for evidence_id, evidence in store.evidence.items():
        claim_id = f"claim-obs-{evidence_id}"
        status = (
            ClaimValidationStatus.SUPPORTED
            if runtime.scenario.trust_producer_supported_label
            else ClaimValidationStatus.QUALIFIED
        )
        if evidence.evidence_kind in {EvidenceKind.SNAPSHOT, EvidenceKind.CLIP}:
            retained = evidence.retrieval_spec.get("retained_media_id")
            has_observe = False
            for rid, receipt in store.receipts.items():
                if receipt.tool_name != "observe_media":
                    continue
                tool = store.tool_results_by_receipt.get(rid)
                if tool is not None and tool.status is ToolResultStatus.OK:
                    has_observe = True
                    break
            statement = (
                f"Authorized media resolve acquired {evidence.evidence_kind.value} "
                f"bytes for {evidence_id}"
                + (f" (retained_media_id={retained})" if retained else "")
                + f" with content_sha256={evidence.content_sha256}. "
                "Byte acquisition and content hashing are not visual interpretation, "
                "scene authenticity, or proof of physical-world events."
            )
            if has_observe:
                limitations = (
                    "content_sha256 verifies byte identity only.",
                    "Separate local VLM observe_media results are model interpretations.",
                )
            else:
                limitations = (
                    "Retrieved media bytes were not visually analyzed in this stage.",
                    "content_sha256 verifies byte identity only.",
                )
            producer = "adapter:frigate-media"
        else:
            statement = (
                f"Authorized source returned event metadata for {evidence_id}."
            )
            limitations = ()
            producer = "stub:deterministic-drafter"
        claim = Claim(
            claim_id=claim_id,
            statement=statement,
            claim_type=ClaimType.SOURCE_OBSERVATION,
            evidence_ids=(evidence_id,),
            producer=producer,
            validation_status=status,
            limitations=limitations,
            validated_at=None,
        )
        store.register_claim(claim)
        claim_ids.append(claim_id)

    # Map observation → evidence for computed temporal claims.
    obs_to_evidence: dict[str, str] = {}
    for evidence in store.evidence.values():
        native = evidence.source_locator.native_record_id
        for obs in store.observations.values():
            if obs.source_locator.native_record_id == native:
                obs_to_evidence[obs.observation_id] = evidence.evidence_id

    temporal_receipt_ids = tuple(
        receipt_id
        for receipt_id, receipt in store.receipts.items()
        if receipt.tool_name == "analyze_temporal_relations"
    )
    for relation_id, relation in store.relations.items():
        claim_id = f"claim-rel-{relation_id}"
        left_ev = obs_to_evidence.get(relation.left_observation_id)
        right_ev = obs_to_evidence.get(relation.right_observation_id)
        evidence_ids = tuple(
            item for item in (left_ev, right_ev) if item is not None
        )
        gap = relation.gap_seconds
        gap_text = "unknown gap" if gap is None else f"{gap:.1f}s gap"
        producer = (
            "temporal-engine:analyze_temporal_relations"
            if temporal_receipt_ids
            else "temporal-engine:evaluate_temporal"
        )
        claim = Claim(
            claim_id=claim_id,
            statement=(
                f"Deterministic {relation.relation_type.value} relation established "
                f"between observations {relation.left_observation_id} and "
                f"{relation.right_observation_id} ({gap_text}, "
                f"window={relation.window_seconds}s). "
                "Temporal proximity alone does not prove physical interaction, "
                "common identity, ownership, or causation."
            ),
            claim_type=ClaimType.COMPUTED_RELATION,
            evidence_ids=evidence_ids,
            relation_ids=(relation_id,),
            search_receipt_ids=temporal_receipt_ids,
            producer=producer,
            validation_status=ClaimValidationStatus.QUALIFIED,
            limitations=(
                "near/before/after relations are interval computations only",
            ),
            validated_at=None,
        )
        store.register_claim(claim)
        claim_ids.append(claim_id)

    for receipt_id, receipt in store.receipts.items():
        tool_result = store.tool_results_by_receipt.get(receipt_id)
        if tool_result is None:
            continue
        if (
            receipt.tool_name == "observe_media"
            and tool_result.status is ToolResultStatus.OK
            and isinstance(tool_result.data, list)
            and tool_result.data
            and isinstance(tool_result.data[0], dict)
        ):
            record = tool_result.data[0]
            if record.get("record_kind") != "visual_observation":
                continue
            if record.get("physical_world_fact") is True:
                continue
            media_evidence_id = str(record.get("media_evidence_id") or "")
            if not media_evidence_id or media_evidence_id not in store.evidence:
                continue
            observation_text = str(record.get("observation_text") or "").strip()
            if not observation_text:
                continue
            model_name = str(record.get("model_name") or "local-vlm")
            claim_id = f"claim-observe-{receipt_id}"
            uncertainty = record.get("uncertainty")
            limitations = [
                "Local VLM interpretation is not an independently verified physical-world fact.",
                "Does not establish identity, ownership, causation, or person-vehicle interaction.",
                "Model statements about absence are not verified real-world absence.",
            ]
            if isinstance(uncertainty, str) and uncertainty.strip():
                limitations.append(uncertainty.strip())
            if record.get("used_fallback_prompt") is True:
                limitations.append(
                    "Inference used a bounded describe-the-image fallback prompt."
                )
            digest = record.get("model_digest")
            producer = f"local-vlm:{model_name}"
            if isinstance(digest, str) and digest.strip():
                producer = f"{producer}@{digest.strip()[:24]}"
            # Faithful quotation of registered model output — no paraphrased facts.
            claim = Claim(
                claim_id=claim_id,
                statement=(
                    f"The local visual model ({model_name}) interpreted retained "
                    f"snapshot {media_evidence_id} as: {observation_text}"
                ),
                claim_type=ClaimType.MODEL_INTERPRETATION,
                evidence_ids=(media_evidence_id,),
                search_receipt_ids=(receipt_id,),
                producer=producer,
                validation_status=ClaimValidationStatus.QUALIFIED,
                limitations=tuple(limitations),
                validated_at=None,
            )
            store.register_claim(claim)
            claim_ids.append(claim_id)
            continue
        if tool_result.status is ToolResultStatus.EMPTY:
            claim_id = f"claim-search-{receipt_id}"
            if receipt.tool_name == "resolve_media":
                statement = (
                    "Authorized media resolve completed without acquiring the "
                    "requested snapshot or clip bytes."
                )
                limitations = (
                    "Unavailable media is not proof the source event did not occur.",
                    "No visual interpretation was performed.",
                )
            elif receipt.tool_name == "analyze_temporal_relations":
                if receipt.coverage is SearchCoverage.TRUNCATED:
                    statement = (
                        "Authorized temporal analysis was truncated at max_pairs "
                        "with no established NEAR relations among evaluated pairs; "
                        "unevaluated pairs were not examined."
                    )
                    limitations = (
                        "Truncated temporal EMPTY is not an exhaustive no-match result.",
                        "Absence of NEAR matches among unevaluated pairs is not proven.",
                        "Proximity analysis never invents interactions.",
                    )
                else:
                    statement = (
                        "Authorized temporal analysis completed with no established "
                        "NEAR relations for the selected observation pairs."
                    )
                    limitations = (
                        "Empty temporal result is not real-world absence.",
                        "Proximity analysis never invents interactions.",
                    )
            else:
                statement = (
                    "The completed authorized query returned no matching records."
                    if not runtime.scenario.inject_real_world_absence_claim
                    else "Nothing happened in the world during the search window."
                )
                limitations = ("EMPTY search never proves real-world absence.",)
            claim = Claim(
                claim_id=claim_id,
                statement=statement,
                claim_type=ClaimType.QUALIFIED_SEARCH_RESULT,
                search_receipt_ids=(receipt_id,),
                producer="stub:deterministic-drafter",
                validation_status=ClaimValidationStatus.QUALIFIED,
                limitations=limitations,
            )
            store.register_claim(claim)
            claim_ids.append(claim_id)
        elif tool_result.status in {
            ToolResultStatus.ERROR,
            ToolResultStatus.UNSUPPORTED,
        }:
            claim_id = f"claim-limit-{receipt_id}"
            claim = Claim(
                claim_id=claim_id,
                statement=(
                    f"Search {receipt.tool_name} ended with {tool_result.status.value}; "
                    "this is distinct from EMPTY and yields no fabricated evidence."
                ),
                claim_type=ClaimType.QUALIFIED_SEARCH_RESULT,
                search_receipt_ids=(receipt_id,),
                producer="stub:deterministic-drafter",
                validation_status=ClaimValidationStatus.QUALIFIED,
                limitations=(f"tool_status={tool_result.status.value}",),
            )
            store.register_claim(claim)
            claim_ids.append(claim_id)

    if runtime.scenario.inject_fabricated_evidence_claim:
        claim_id = "claim-fabricated-ev"
        claim = Claim(
            claim_id=claim_id,
            statement="Supported conclusion based on missing evidence.",
            claim_type=ClaimType.SOURCE_OBSERVATION,
            evidence_ids=("ev-does-not-exist",),
            producer="stub:deterministic-drafter",
            validation_status=ClaimValidationStatus.SUPPORTED,
        )
        store.register_claim(claim)
        claim_ids.append(claim_id)

    if runtime.scenario.inject_false_temporal_claim and len(store.observations) >= 2:
        obs_ids = list(store.observations.keys())
        left_id, right_id = obs_ids[0], obs_ids[1]
        # Force a false BEFORE by swapping evaluation: claim cites established
        # AFTER when BEFORE actually holds (or vice versa).
        left = store.observation_intervals[left_id]
        right = store.observation_intervals[right_id]
        actual_before = evaluate_predicate(left, right, TemporalRelationType.BEFORE)
        # Materialize a fake established relation that contradicts reality.
        from nanexus_investigation.contracts.temporal import TemporalRelation
        from nanexus_investigation.contracts.enums import TemporalEvaluationStatus

        false_rel = TemporalRelation(
            relation_id="rel-false-before",
            left_observation_id=right_id if actual_before.is_true else left_id,
            right_observation_id=left_id if actual_before.is_true else right_id,
            relation_type=TemporalRelationType.BEFORE,
            evaluation_status=TemporalEvaluationStatus.ESTABLISHED,
            camera_relation="same_or_authorized_scope",
            time_basis="completed_half_open_utc_micros",
            calculation_provenance="stub:injected-false",
        )
        # If swapping still holds true, invert by using AFTER established wrongly.
        swapped = evaluate_predicate(
            store.observation_intervals[false_rel.left_observation_id],
            store.observation_intervals[false_rel.right_observation_id],
            TemporalRelationType.BEFORE,
        )
        if not swapped.is_false:
            false_rel = TemporalRelation(
                relation_id="rel-false-before",
                left_observation_id=right_id,
                right_observation_id=left_id,
                relation_type=TemporalRelationType.BEFORE,
                evaluation_status=TemporalEvaluationStatus.ESTABLISHED,
                camera_relation="same_or_authorized_scope",
                time_basis="completed_half_open_utc_micros",
                calculation_provenance="stub:injected-false",
            )
        store.register_relation(false_rel)
        claim = Claim(
            claim_id="claim-false-temporal",
            statement="Supported conclusion: temporal order is established.",
            claim_type=ClaimType.COMPUTED_RELATION,
            relation_ids=(false_rel.relation_id,),
            evidence_ids=tuple(store.evidence.keys())[:1],
            producer="stub:deterministic-drafter",
            validation_status=ClaimValidationStatus.SUPPORTED,
            validated_at=now,
        )
        store.register_claim(claim)
        claim_ids.append(claim.claim_id)

    if runtime.scenario.inject_unknown_temporal_claim and store.observations:
        from nanexus_investigation.contracts.temporal import TemporalRelation
        from nanexus_investigation.contracts.enums import TemporalEvaluationStatus

        obs_ids = list(store.observations.keys())
        left_id = obs_ids[0]
        # Create a second open-ended observation so BEFORE is unknown.
        open_obs = Observation(
            observation_id="obs-open-ended",
            source_locator=SourceLocator(
                vms_instance_id=runtime.case.effective_scope.vms_instance_id,
                record_type=RecordType.EVENT,
                native_record_id="evt-open",
            ),
            source_revision="rev-1",
            observation_type=ObservationType.DETECTION_EVENT,
            camera_id=runtime.case.effective_scope.camera_ids[0],
            start_utc=runtime.case.effective_scope.time_bounds.start_utc,
            end_utc=None,
            retrieved_at=now,
            provenance="stub:open-ended",
        )
        store.register_observation(open_obs)
        store.observation_intervals[open_obs.observation_id] = observation_interval(
            open_obs
        )
        unknown_rel = TemporalRelation(
            relation_id="rel-unknown-before",
            left_observation_id=left_id,
            right_observation_id=open_obs.observation_id,
            relation_type=TemporalRelationType.BEFORE,
            evaluation_status=TemporalEvaluationStatus.ESTABLISHED,
            camera_relation="same_or_authorized_scope",
            time_basis="completed_half_open_utc_micros",
            calculation_provenance="stub:injected-unknown",
        )
        store.register_relation(unknown_rel)
        claim = Claim(
            claim_id="claim-unknown-temporal",
            statement="Temporal order is established.",
            claim_type=ClaimType.COMPUTED_RELATION,
            relation_ids=(unknown_rel.relation_id,),
            evidence_ids=tuple(store.evidence.keys())[:1] or (),
            producer="stub:deterministic-drafter",
            validation_status=ClaimValidationStatus.SUPPORTED,
        )
        # SOURCE_OBSERVATION needs evidence; COMPUTED_RELATION with supported needs citations.
        # If no evidence, attach search receipt if any.
        if not claim.evidence_ids and store.receipts:
            claim = claim.model_copy(
                update={
                    "claim_type": ClaimType.COMPUTED_RELATION,
                    "search_receipt_ids": (next(iter(store.receipts)),),
                    "validation_status": ClaimValidationStatus.QUALIFIED,
                }
            )
        store.register_claim(claim)
        claim_ids.append(claim.claim_id)

    runtime.trace.record("claims_drafted", claim_ids=list(claim_ids))
    return claim_ids


def validate_claims(runtime: InvestigationRuntime, claim_ids: list[str]) -> list[str]:
    validated: list[str] = []
    store = runtime.store
    for claim_id in claim_ids:
        claim = store.claims[claim_id]
        assessment = assess_claim(
            claim,
            effective_scope=runtime.case.effective_scope,
            evidence_by_id=store.evidence,
            observations_by_id=store.observations,
            relations_by_id=store.relations,
            observation_intervals=store.observation_intervals,
            receipts_by_id=store.receipts,
            tool_results_by_receipt_id=store.tool_results_by_receipt,
            require_grant_membership=False,
        )
        store.set_assessment(assessment)
        runtime.trace.record(
            "claim_assessment",
            claim_id=claim_id,
            status=assessment.status.value,
            reasons=list(assessment.reasons),
            producer_status=claim.validation_status.value,
        )
        if assessment.status in {
            ClaimValidationStatus.SUPPORTED,
            ClaimValidationStatus.QUALIFIED,
        }:
            validated.append(claim_id)
    return validated


def _format_scope_summary(runtime: InvestigationRuntime) -> str:
    """Operator-readable scope line: question, VMS, cameras, authorized window."""
    scope = runtime.case.effective_scope
    bounds = scope.time_bounds
    start = bounds.start_utc.isoformat()
    end = None if bounds.end_utc is None else bounds.end_utc.isoformat()
    question = (runtime.case.question or "").strip()
    if len(question) > 160:
        question = question[:157] + "..."
    return (
        f"question={question!r}; "
        f"vms={scope.vms_instance_id}; "
        f"cameras={list(scope.camera_ids)}; "
        f"time=[{start} .. {end}]"
    )


def _search_receipt_label_tokens(raw: object) -> tuple[set[str], bool]:
    """Parse search receipt label filters into tokens.

    Returns ``(labels, unfiltered)``. Frigate multi-label forms such as
    ``person,car`` are split. Empty / missing label means unfiltered search.
    """
    if raw is None:
        return set(), True
    if isinstance(raw, (list, tuple, set)):
        labels = {str(item).strip().lower() for item in raw if str(item).strip()}
        return labels, not labels
    if isinstance(raw, str):
        text = raw.strip().lower()
        if not text:
            return set(), True
        labels = {part.strip() for part in text.split(",") if part.strip()}
        return labels, not labels
    return set(), True


def _unexecuted_detection_objectives(
    runtime: InvestigationRuntime,
) -> tuple[str, ...]:
    """Detect person+vehicle questions where only one label family was searched.

    Heuristic coverage check only — not a general objective planner. An
    unfiltered ``search_events`` (no label filter) counts as covering both.
    A multi-label filter that includes both families (for example ``person,car``)
    also covers both objectives.
    """
    question = (runtime.case.question or "").lower()
    asks_person = any(
        token in question for token in ("person", "people", "pedestrian")
    )
    asks_vehicle = any(
        token in question
        for token in ("vehicle", "car", "cars", "truck", "automobile")
    )
    if not (asks_person and asks_vehicle):
        return ()

    explicit_labels: set[str] = set()
    has_unfiltered_search = False
    for receipt in runtime.store.receipts.values():
        if receipt.tool_name != "search_events":
            continue
        labels, unfiltered = _search_receipt_label_tokens(receipt.filters.get("label"))
        if unfiltered:
            has_unfiltered_search = True
        else:
            explicit_labels.update(labels)
    if has_unfiltered_search:
        return ()

    person_done = "person" in explicit_labels
    vehicle_done = bool(
        explicit_labels & {"car", "vehicle", "truck", "bus", "motorcycle"}
    )
    missing: list[str] = []
    if asks_person and not person_done:
        missing.append("person")
    if asks_vehicle and not vehicle_done:
        missing.append("vehicle")
    return tuple(missing)


def assemble_report(
    runtime: InvestigationRuntime,
    *,
    validated_claim_ids: list[str],
) -> InvestigationReport:
    store = runtime.store
    limitations: list[str] = []
    limitations.extend(runtime.planning_limitations)
    for receipt_id, receipt in store.receipts.items():
        tool_result = store.tool_results_by_receipt.get(receipt_id)
        if tool_result is None:
            continue
        interpretation = interpret_search(tool_result, receipt)
        if tool_result.status is ToolResultStatus.OK:
            if not interpretation.is_exhaustive_search:
                limitations.append(interpretation.summary)
            else:
                limitations.append(
                    "Findings are limited to the authorized scope and completed query."
                )
        elif tool_result.status is ToolResultStatus.EMPTY:
            limitations.append(interpretation.summary)
            limitations.append(
                "EMPTY exhausted search is not a real-world absence statement."
            )
        elif tool_result.status in {
            ToolResultStatus.ERROR,
            ToolResultStatus.UNSUPPORTED,
        }:
            limitations.append(interpretation.summary)
            limitations.append(
                f"{tool_result.status.value} remains distinct from EMPTY; "
                "no observations were fabricated."
            )
        else:
            limitations.append(interpretation.summary)

    for assessment in store.assessments.values():
        if assessment.status in {
            ClaimValidationStatus.UNSUPPORTED,
            ClaimValidationStatus.UNVERIFIABLE,
        }:
            limitations.append(
                f"{assessment.claim_id}:{assessment.status.value}:"
                + ";".join(assessment.reasons[:2])
            )

    # Only include gate-eligible claims: validated set plus explicit limitation claims
    # that are still structurally cited. Fabricated/contradicted claims stay out of
    # the successful report path unless a scenario forces them for negative tests.
    report_claims = list(validated_claim_ids)
    if runtime.scenario.inject_fabricated_evidence_claim:
        report_claims.append("claim-fabricated-ev")
    if runtime.scenario.inject_false_temporal_claim:
        report_claims.append("claim-false-temporal")
    if runtime.scenario.inject_unknown_temporal_claim:
        report_claims.append("claim-unknown-temporal")
    if runtime.scenario.inject_real_world_absence_claim:
        for claim_id in store.claims:
            if claim_id.startswith("claim-search-"):
                report_claims.append(claim_id)
    if runtime.scenario.trust_producer_supported_label:
        report_claims = list(store.claims.keys())

    # Deduplicate while preserving order.
    seen: set[str] = set()
    ordered_claims: list[str] = []
    for claim_id in report_claims:
        if claim_id not in seen and claim_id in store.claims:
            seen.add(claim_id)
            ordered_claims.append(claim_id)

    timeline_entries = _build_evidence_timeline(store)
    if store.relations:
        limitations.append(
            "Timeline temporal relations are deterministic interval computations; "
            "proximity alone does not establish interaction, identity, ownership, "
            "or causation."
        )
    if any(
        evidence.evidence_kind in {EvidenceKind.SNAPSHOT, EvidenceKind.CLIP}
        for evidence in store.evidence.values()
    ):
        limitations.append(
            "Acquired media bytes and content hashes verify byte identity only; "
            "media retrieval is not visual interpretation, person/vehicle identity, "
            "scene authenticity, or proof that an event occurred."
        )
    if any(
        receipt.tool_name == "observe_media" for receipt in store.receipts.values()
    ):
        limitations.append(
            "Local VLM interpretations are model inferences over retained media, "
            "not independently verified physical-world facts; they do not establish "
            "identity, ownership, causation, or person-vehicle interaction."
        )
    for receipt in store.receipts.values():
        if (
            receipt.tool_name == "analyze_temporal_relations"
            and receipt.coverage is SearchCoverage.TRUNCATED
        ):
            limitations.append(
                "Temporal pair evaluation was truncated at max_pairs; established "
                "NEAR relations are not exhaustive and absence of additional matches "
                "is not proven."
            )
            break

    unresolved: list[str] = []
    missing_objectives = _unexecuted_detection_objectives(runtime)
    if missing_objectives:
        missing_txt = ", ".join(missing_objectives)
        limitations.append(
            "partial investigation coverage: the question asked for both person "
            "and vehicle detections, but the completed search did not cover: "
            f"{missing_txt}. Results shown are only for the searches that ran; "
            "missing objectives are unanswered and are not negative findings."
        )
        unresolved.append(
            f"Requested detection review not completed for: {missing_txt}."
        )

    report = InvestigationReport(
        report_id=f"report-{uuid4().hex[:10]}",
        case_id=runtime.case.case_id,
        generated_at=datetime(2026, 11, 20, 12, 15, tzinfo=UTC),
        scope_summary=_format_scope_summary(runtime),
        execution_status=CaseStatus.FAILED,  # placeholder until Output Gate decides
        timeline=timeline_entries,
        claims=tuple(ordered_claims),
        evidence_index=tuple(store.evidence.keys()),
        limitations=tuple(dict.fromkeys(limitations)),
        unresolved_questions=tuple(unresolved),
        validation_summary="pending_output_gate",
        report_version="e0-v1",
    )
    return report


def _build_evidence_timeline(store: InMemoryCaseRepository) -> tuple[str, ...]:
    """Ordered observation IDs by start time, then established relation IDs."""
    observations = sorted(
        store.observations.values(),
        key=lambda item: (item.start_utc, item.observation_id),
    )
    entries = [item.observation_id for item in observations]
    for relation_id in sorted(store.relations.keys()):
        entries.append(relation_id)
    return tuple(entries)
