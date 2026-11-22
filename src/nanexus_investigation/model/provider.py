"""Minimal replaceable local Model Provider for investigation proposals.

Supports natural-language initial planning and evidence-dependent follow-up.
The provider generates untrusted structured proposals only. Tool Guard retains
authorization; the Evidence Engine and Output Gate retain correctness authority.
No filesystem, shell, or unrestricted network tools are exposed to the model.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

ProposalAction = Literal["invoke_tool", "proceed_to_report"]

ALLOWED_PROPOSAL_ACTIONS: frozenset[str] = frozenset(
    {"invoke_tool", "proceed_to_report"}
)


class ProposalParseError(ValueError):
    """Model output could not be parsed into a typed investigation proposal."""


@dataclass(frozen=True, slots=True)
class InvestigationProposal:
    """Typed action proposal. Untrusted until Tool Guard validates."""

    action: ProposalAction
    tool_name: str | None
    label: str | None
    rationale: str
    camera_ids: tuple[str, ...] | None = None
    # Optional half-open UTC window; must still pass Tool Guard scope checks.
    time_bounds: dict[str, str | None] | None = None
    # Temporal analysis fields (analyze_temporal_relations).
    predicate: str | None = None
    window_seconds: float | None = None
    anchor_label: str | None = None
    candidate_label: str | None = None
    max_pairs: int | None = None
    # Media resolve fields (resolve_media).
    media_kind: str | None = None
    evidence_id: str | None = None
    observation_id: str | None = None
    # Visual observation fields (observe_media).
    objective: str | None = None
    raw_text: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "tool_name": self.tool_name,
            "label": self.label,
            "rationale": self.rationale,
            "camera_ids": list(self.camera_ids) if self.camera_ids is not None else None,
            "time_bounds": self.time_bounds,
            "predicate": self.predicate,
            "window_seconds": self.window_seconds,
            "anchor_label": self.anchor_label,
            "candidate_label": self.candidate_label,
            "max_pairs": self.max_pairs,
            "media_kind": self.media_kind,
            "evidence_id": self.evidence_id,
            "observation_id": self.observation_id,
            "objective": self.objective,
        }

    def analysis_arguments(self) -> dict[str, Any] | None:
        """Structured arguments for analyze_temporal_relations."""
        if self.tool_name != "analyze_temporal_relations":
            return None
        payload: dict[str, Any] = {
            "predicate": self.predicate or "near",
        }
        if self.window_seconds is not None:
            payload["window_seconds"] = self.window_seconds
        if self.anchor_label is not None:
            payload["anchor_label"] = self.anchor_label
        if self.candidate_label is not None:
            payload["candidate_label"] = self.candidate_label
        if self.max_pairs is not None:
            payload["max_pairs"] = self.max_pairs
        return payload

    def media_arguments(self) -> dict[str, Any] | None:
        """Structured arguments for resolve_media."""
        if self.tool_name != "resolve_media":
            return None
        payload: dict[str, Any] = {
            "media_kind": self.media_kind or "snapshot",
        }
        if self.evidence_id is not None:
            payload["evidence_id"] = self.evidence_id
        if self.observation_id is not None:
            payload["observation_id"] = self.observation_id
        return payload

    def observe_arguments(self) -> dict[str, Any] | None:
        """Structured arguments for observe_media."""
        if self.tool_name != "observe_media":
            return None
        payload: dict[str, Any] = {}
        if self.evidence_id is not None:
            payload["evidence_id"] = self.evidence_id
        if self.objective is not None:
            payload["objective"] = self.objective
        return payload


@dataclass(frozen=True, slots=True)
class ProposalContext:
    """Bounded, non-media context supplied to the local model."""

    question: str
    case_id: str
    authorized_tools: tuple[str, ...]
    authorized_cameras: tuple[str, ...]
    vms_instance_id: str
    time_bounds: dict[str, str | None]
    prior_tool_name: str | None
    prior_tool_status: str | None
    prior_match_count: int
    prior_result_summary: str
    evidence_notes: tuple[str, ...] = ()
    coverage_notes: tuple[str, ...] = ()
    # Optional human-readable tool descriptions for initial planning.
    tool_descriptions: tuple[str, ...] = ()


class ModelProvider(Protocol):
    """Replaceable local inference adapter. Must not execute tools itself."""

    def propose_follow_up(self, context: ProposalContext) -> InvestigationProposal: ...


def parse_investigation_proposal(raw_text: str) -> InvestigationProposal:
    """Parse and validate model JSON into a typed proposal. Fails closed."""
    text = (raw_text or "").strip()
    if not text:
        raise ProposalParseError("empty model response")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ProposalParseError(f"model response is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ProposalParseError("model response JSON must be an object")

    action = payload.get("action")
    tool_name = payload.get("tool_name")
    if tool_name is not None and not isinstance(tool_name, str):
        raise ProposalParseError("tool_name must be a string or null")
    if isinstance(tool_name, str):
        tool_name = tool_name.strip() or None

    # Local models occasionally put the tool name in ``action`` instead of
    # ``invoke_tool``. Recover that shape before fail-closed rejection.
    _tool_shaped_actions = frozenset(
        {
            "search_events",
            "search_related_events",
            "analyze_temporal_relations",
            "resolve_media",
            "observe_media",
        }
    )
    if action not in ALLOWED_PROPOSAL_ACTIONS:
        if isinstance(action, str) and action.strip() in _tool_shaped_actions:
            if tool_name is None:
                tool_name = action.strip()
            action = "invoke_tool"
        else:
            raise ProposalParseError(
                f"unsupported or missing action: {action!r}; "
                f"allowed={sorted(ALLOWED_PROPOSAL_ACTIONS)}"
            )

    label = payload.get("label")
    if label is not None and not isinstance(label, str):
        raise ProposalParseError("label must be a string or null")
    if isinstance(label, str):
        label = label.strip() or None

    rationale = payload.get("rationale")
    if "rationale" not in payload:
        # Local models sometimes omit rationale when JSON output is truncated.
        # Recover only when the key is absent — explicit empty string still fails.
        tool_hint = payload.get("tool_name")
        action_hint = payload.get("action")
        if action_hint == "invoke_tool" and isinstance(tool_hint, str) and tool_hint.strip():
            rationale = (
                f"Model omitted rationale; proposed invoke_tool "
                f"tool_name={tool_hint.strip()!r}."
            )
        elif action_hint == "proceed_to_report":
            rationale = "Model omitted rationale; proposed proceed_to_report."
        else:
            raise ProposalParseError("rationale must be a non-empty string")
    elif not isinstance(rationale, str) or not rationale.strip():
        raise ProposalParseError("rationale must be a non-empty string")

    camera_ids: tuple[str, ...] | None = None
    if "camera_ids" in payload and payload["camera_ids"] is not None:
        cameras = payload["camera_ids"]
        if not isinstance(cameras, list) or not all(isinstance(c, str) for c in cameras):
            raise ProposalParseError("camera_ids must be a list of strings when provided")
        camera_ids = tuple(c.strip() for c in cameras if c.strip())

    time_bounds: dict[str, str | None] | None = None
    if "time_bounds" in payload and payload["time_bounds"] is not None:
        raw_bounds = payload["time_bounds"]
        if not isinstance(raw_bounds, dict):
            raise ProposalParseError("time_bounds must be an object when provided")
        start = raw_bounds.get("start_utc")
        end = raw_bounds.get("end_utc")
        if start is not None and not isinstance(start, str):
            raise ProposalParseError("time_bounds.start_utc must be a string or null")
        if end is not None and not isinstance(end, str):
            raise ProposalParseError("time_bounds.end_utc must be a string or null")
        if not isinstance(start, str) or not start.strip():
            raise ProposalParseError("time_bounds.start_utc is required when time_bounds is set")
        time_bounds = {
            "start_utc": start.strip(),
            "end_utc": None if end is None else str(end).strip() or None,
        }

    predicate = payload.get("predicate")
    if predicate is not None and not isinstance(predicate, str):
        raise ProposalParseError("predicate must be a string or null")
    if isinstance(predicate, str):
        predicate = predicate.strip().lower() or None

    window_seconds: float | None = None
    if "window_seconds" in payload and payload["window_seconds"] is not None:
        try:
            window_seconds = float(payload["window_seconds"])
        except (TypeError, ValueError) as exc:
            raise ProposalParseError("window_seconds must be a number") from exc
        if window_seconds < 0:
            raise ProposalParseError("window_seconds must be nonnegative")

    anchor_label = payload.get("anchor_label")
    if anchor_label is not None and not isinstance(anchor_label, str):
        raise ProposalParseError("anchor_label must be a string or null")
    if isinstance(anchor_label, str):
        anchor_label = anchor_label.strip() or None

    candidate_label = payload.get("candidate_label")
    if candidate_label is not None and not isinstance(candidate_label, str):
        raise ProposalParseError("candidate_label must be a string or null")
    if isinstance(candidate_label, str):
        candidate_label = candidate_label.strip() or None

    max_pairs: int | None = None
    if "max_pairs" in payload and payload["max_pairs"] is not None:
        try:
            max_pairs = int(payload["max_pairs"])
        except (TypeError, ValueError) as exc:
            raise ProposalParseError("max_pairs must be an integer") from exc
        if max_pairs < 1:
            raise ProposalParseError("max_pairs must be >= 1")

    media_kind = payload.get("media_kind")
    if media_kind is not None and not isinstance(media_kind, str):
        raise ProposalParseError("media_kind must be a string or null")
    if isinstance(media_kind, str):
        media_kind = media_kind.strip().lower() or None
    evidence_id = payload.get("evidence_id")
    if evidence_id is not None and not isinstance(evidence_id, str):
        raise ProposalParseError("evidence_id must be a string or null")
    if isinstance(evidence_id, str):
        evidence_id = evidence_id.strip() or None
    observation_id = payload.get("observation_id")
    if observation_id is not None and not isinstance(observation_id, str):
        raise ProposalParseError("observation_id must be a string or null")
    if isinstance(observation_id, str):
        observation_id = observation_id.strip() or None
    objective = payload.get("objective")
    if objective is None:
        objective = payload.get("question")
    if objective is not None and not isinstance(objective, str):
        raise ProposalParseError("objective must be a string or null")
    if isinstance(objective, str):
        objective = objective.strip() or None

    if action == "invoke_tool" and not tool_name:
        raise ProposalParseError("invoke_tool requires tool_name")
    if action == "proceed_to_report":
        tool_name = None
        time_bounds = None
        predicate = None
        window_seconds = None
        anchor_label = None
        candidate_label = None
        max_pairs = None
        media_kind = None
        evidence_id = None
        observation_id = None
        objective = None

    if tool_name == "analyze_temporal_relations":
        if window_seconds is None:
            raise ProposalParseError(
                "analyze_temporal_relations requires window_seconds"
            )
        predicate = predicate or "near"
    if tool_name == "resolve_media":
        # Default matches media_arguments(); models often omit media_kind.
        if media_kind is None:
            media_kind = "snapshot"
        if media_kind not in {"snapshot", "clip"}:
            raise ProposalParseError(
                "resolve_media requires media_kind=snapshot|clip"
            )
        if not evidence_id and not observation_id:
            raise ProposalParseError(
                "resolve_media requires evidence_id or observation_id"
            )
    if tool_name == "observe_media":
        if not evidence_id:
            raise ProposalParseError("observe_media requires evidence_id")
        if not objective:
            raise ProposalParseError("observe_media requires objective")
        media_kind = None
        observation_id = None

    return InvestigationProposal(
        action=action,  # type: ignore[arg-type]
        tool_name=tool_name,
        label=label,
        rationale=rationale.strip(),
        camera_ids=camera_ids,
        time_bounds=time_bounds,
        predicate=predicate,
        window_seconds=window_seconds,
        anchor_label=anchor_label,
        candidate_label=candidate_label,
        max_pairs=max_pairs,
        media_kind=media_kind,
        evidence_id=evidence_id,
        observation_id=observation_id,
        objective=objective,
        raw_text=text,
    )


def _decision_rule_for_tools(authorized_tools: tuple[str, ...]) -> str:
    """Capability-aware follow-up guidance. Never advertise unsupported tools."""
    tools = set(authorized_tools)
    has_search = "search_events" in tools
    has_temporal = "analyze_temporal_relations" in tools
    has_related = "search_related_events" in tools
    has_media = "resolve_media" in tools
    has_observe = "observe_media" in tools

    parts: list[str] = ["Decision rule:"]
    if has_search:
        parts.append(
            "If the question needs another detection label (e.g. car after person) "
            "and matches exist, choose action=invoke_tool with tool_name=search_events "
            "and label=person|car (never label=vehicle)."
        )
    if has_temporal:
        parts.append(
            "If person and vehicle/car observations are already registered and the "
            "question asks about nearby activity, timeline, or proximity within a "
            "bounded window, choose action=invoke_tool with "
            "tool_name=analyze_temporal_relations, predicate=near, "
            "window_seconds as a nonnegative number of seconds "
            "(e.g. 180 for three minutes), anchor_label=person, candidate_label=car. "
            "Do not claim the mathematical relation is true — the deterministic "
            "engine decides. Temporal proximity alone never proves interaction, "
            "identity, ownership, or causation."
        )
    if has_media:
        parts.append(
            "If the question explicitly asks for a snapshot, clip, image, video, or "
            "visual evidence of an already registered detection, choose "
            "action=invoke_tool with tool_name=resolve_media, media_kind=snapshot|clip, "
            "and evidence_id (preferred, values starting with 'ev-') or observation_id "
            "(values starting with 'obs-') copied exactly from registered Case evidence. "
            "Never invent IDs, timestamps-as-IDs, URLs, paths, or hosts. "
            "Acquiring media bytes is not visual verification — do not claim you "
            "inspected image content."
        )
    if has_observe:
        parts.append(
            "If retained SNAPSHOT media evidence already exists (evidence_id starting "
            "with 'ev-media-') and the question asks to examine, inspect, look at, or "
            "visually verify the image contents, choose action=invoke_tool with "
            "tool_name=observe_media, evidence_id copied exactly from that SNAPSHOT "
            "EvidenceRef, and a short objective string. Do not invent media IDs. "
            "Do not call observe_media before resolve_media has retained a snapshot. "
            "A local VLM interpretation is not a verified physical-world fact."
        )
    if has_related:
        parts.append(
            "If search_related_events is authorized and warranted, it may be used."
        )
    if has_search and not has_temporal:
        parts.append(
            "Do not invent analyze_temporal_relations or unsupported tools."
        )
    parts.append(
        "If prior results are empty/exhausted, required analysis is already done, "
        "or no further authorized action is useful, choose action=proceed_to_report. "
        "Do not invent search_related_events when it is not authorized. "
        "Do not call resolve_media for ordinary retrieval or temporal questions. "
        "Do not call observe_media for ordinary retrieval without a visual objective."
    )
    return " ".join(parts)


def _json_proposal_schema_hint() -> str:
    return (
        "Reply with ONLY a JSON object. Always include a non-empty rationale. "
        "Keys: "
        "action (invoke_tool|proceed_to_report), "
        "rationale (REQUIRED non-empty string), "
        "tool_name (string|null), label (string|null), "
        "camera_ids (array of strings|null), "
        "time_bounds (object with start_utc/end_utc strings|null), "
        "predicate (string|null; use near for temporal analysis), "
        "window_seconds (number|null; required for analyze_temporal_relations), "
        "anchor_label (string|null), candidate_label (string|null), "
        "max_pairs (integer|null), "
        "media_kind (snapshot|clip|null), evidence_id (string|null), "
        "observation_id (string|null), "
        "objective (string|null; required for observe_media). "
        "Do not invent evidence. Do not request tools outside the authorized list. "
        "Do not claim authorization. Do not invent causal person-vehicle relationships. "
        "Do not invent URLs or filesystem paths. No markdown."
    )


def _label_mapping_hint() -> str:
    return (
        "For this Frigate-backed source, detection labels are typically "
        "person and car. Map natural-language 'vehicle'/'car' requests to "
        "label=car (never label=vehicle). Map 'person'/'people' to label=person."
    )


def build_initial_plan_prompt(context: ProposalContext) -> tuple[str, str]:
    """Return (system, user) messages for the first investigation action."""
    system = (
        "You are the Nanexus investigation initial planner. "
        "Propose the FIRST investigation action for the user's question. "
        f"{_json_proposal_schema_hint()}"
    )
    tools = ", ".join(context.authorized_tools) or "(none)"
    cameras = ", ".join(context.authorized_cameras) or "(none)"
    descriptions = (
        "\n".join(f"- {item}" for item in context.tool_descriptions)
        if context.tool_descriptions
        else "(see authorized tools)"
    )
    user = (
        f"Case: {context.case_id}\n"
        f"Question: {context.question}\n"
        f"Authorized VMS: {context.vms_instance_id}\n"
        f"Authorized cameras: {cameras}\n"
        f"Authorized time bounds: {json.dumps(context.time_bounds)}\n"
        f"Authorized tools: {tools}\n"
        f"Tool descriptions:\n{descriptions}\n"
        f"{_label_mapping_hint()}\n"
        "Decision rule: for the FIRST step choose action=invoke_tool with "
        "tool_name=search_events and the label that best matches the question "
        "(person or car). Stay inside the authorized cameras and time bounds. "
        "Do not invent unsupported tools. Do not expand authorization scope. "
        "Do not start with analyze_temporal_relations, resolve_media, or "
        "observe_media — retrieve detections first."
    )
    return system, user


def build_follow_up_prompt(context: ProposalContext) -> tuple[str, str]:
    """Return (system, user) messages for structured follow-up selection."""
    system = (
        "You are the Nanexus investigation follow-up planner. "
        "Propose the next investigation action only. "
        f"{_json_proposal_schema_hint()}"
    )
    tools = ", ".join(context.authorized_tools) or "(none)"
    cameras = ", ".join(context.authorized_cameras) or "(none)"
    notes = "; ".join(context.evidence_notes) or "(none)"
    coverage = "; ".join(context.coverage_notes) or "(none)"
    user = (
        f"Case: {context.case_id}\n"
        f"Question: {context.question}\n"
        f"Authorized VMS: {context.vms_instance_id}\n"
        f"Authorized cameras: {cameras}\n"
        f"Authorized time bounds: {json.dumps(context.time_bounds)}\n"
        f"Authorized tools: {tools}\n"
        f"Prior tool: {context.prior_tool_name or '(none)'}\n"
        f"Prior status: {context.prior_tool_status or '(none)'}\n"
        f"Prior match count: {context.prior_match_count}\n"
        f"Prior result summary: {context.prior_result_summary}\n"
        f"Evidence notes: {notes}\n"
        f"Coverage notes: {coverage}\n"
        f"{_label_mapping_hint()}\n"
        f"{_decision_rule_for_tools(context.authorized_tools)}"
    )
    return system, user


def default_tool_descriptions(authorized_tools: tuple[str, ...]) -> tuple[str, ...]:
    """Short capability descriptions for tools the model may propose."""
    descriptions: list[str] = []
    for name in authorized_tools:
        if name == "search_events":
            descriptions.append(
                "search_events: bounded read-only detection search by optional "
                "label (person|car), cameras, and time_bounds inside the "
                "authorized envelope."
            )
        elif name == "search_related_events":
            descriptions.append(
                "search_related_events: related-event search when supported by "
                "the VMS backend."
            )
        elif name == "analyze_temporal_relations":
            descriptions.append(
                "analyze_temporal_relations: deterministic local NEAR analysis "
                "over already registered Case observations. Requires "
                "window_seconds; optional anchor_label/candidate_label "
                "(e.g. person/car). Does not fetch VMS data and does not prove "
                "physical interaction."
            )
        elif name == "resolve_media":
            descriptions.append(
                "resolve_media: acquire authorized snapshot or event clip bytes "
                "for an already registered Case evidence_id or observation_id. "
                "Requires media_kind=snapshot|clip. Does not accept URLs/paths. "
                "Byte acquisition is not visual interpretation."
            )
        elif name == "observe_media":
            descriptions.append(
                "observe_media: local VLM visual observation over an already "
                "retained SNAPSHOT EvidenceRef (evidence_id starting with "
                "ev-media-). Requires objective. Does not accept URLs/paths. "
                "Interpretation is model opinion, not a verified physical fact."
            )
        else:
            descriptions.append(f"{name}: authorized investigation tool.")
    return tuple(descriptions)


@dataclass
class OllamaChatProvider:
    """Local Ollama chat provider. HTTP only to the configured local endpoint."""

    model: str = "qwen3:14b"
    base_url: str = "http://127.0.0.1:11434"
    timeout_s: float = 180.0
    temperature: float = 0.0
    num_predict: int = 1024
    # qwen3 may spend the token budget on internal "thinking" and return empty
    # content; disable thinking for structured proposal reliability.
    think: bool = False

    def propose_initial_plan(self, context: ProposalContext) -> InvestigationProposal:
        system, user = build_initial_plan_prompt(context)
        return self._complete(system, user)

    def propose_follow_up(self, context: ProposalContext) -> InvestigationProposal:
        system, user = build_follow_up_prompt(context)
        return self._complete(system, user)

    def _complete(self, system: str, user: str) -> InvestigationProposal:
        endpoint = self.base_url.rstrip("/") + "/api/chat"
        body = {
            "model": self.model,
            "stream": False,
            "format": "json",
            "think": self.think,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.num_predict,
            },
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise ProposalParseError(f"local model endpoint unreachable: {exc}") from exc
        except TimeoutError as exc:
            raise ProposalParseError(f"local model timed out: {exc}") from exc

        content = ""
        message = payload.get("message") if isinstance(payload, dict) else None
        if isinstance(message, dict):
            content = str(message.get("content") or "").strip()
            # Narrow recovery if a thinking model still returns empty content.
            if not content:
                thinking = str(message.get("thinking") or "")
                start = thinking.find("{")
                end = thinking.rfind("}")
                if start >= 0 and end > start:
                    content = thinking[start : end + 1]
        return parse_investigation_proposal(content)


@dataclass
class StaticModelProvider:
    """Test double that returns fixed/sequential proposals. Not production inference."""

    proposal: InvestigationProposal | None = None
    raw_text: str | None = None
    initial_proposal: InvestigationProposal | None = None
    # Sequential follow-up proposals; when exhausted, proceed_to_report.
    proposals: list[InvestigationProposal] | None = None
    calls: list[ProposalContext] = field(default_factory=list)
    initial_calls: list[ProposalContext] = field(default_factory=list)
    _follow_up_index: int = 0

    def propose_initial_plan(self, context: ProposalContext) -> InvestigationProposal:
        self.initial_calls.append(context)
        if self.initial_proposal is not None:
            return self.initial_proposal
        if self.proposal is not None:
            return self.proposal
        if self.raw_text is not None:
            return parse_investigation_proposal(self.raw_text)
        raise ProposalParseError("StaticModelProvider has no initial proposal configured")

    def propose_follow_up(self, context: ProposalContext) -> InvestigationProposal:
        self.calls.append(context)
        if self.proposals is not None:
            if self._follow_up_index >= len(self.proposals):
                return InvestigationProposal(
                    action="proceed_to_report",
                    tool_name=None,
                    label=None,
                    rationale="No further scripted follow-up proposals; stop.",
                )
            item = self.proposals[self._follow_up_index]
            self._follow_up_index += 1
            return item
        if self.proposal is not None:
            return self.proposal
        if self.raw_text is not None:
            return parse_investigation_proposal(self.raw_text)
        raise ProposalParseError("StaticModelProvider has no proposal configured")
