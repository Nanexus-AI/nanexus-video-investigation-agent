"""Deterministic investigation planner stub. Test behavior only — not an LLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nanexus_investigation.contracts.identity import Interval
from nanexus_investigation.tools.registry import ALLOWED_TOOLS


@dataclass(frozen=True, slots=True)
class PlanStep:
    """One bounded plan step. May only name allowlisted tools."""

    step_id: str
    tool_name: str
    label: str | None = None
    camera_ids: tuple[str, ...] | None = None
    time_bounds: Interval | None = None
    requires_prior_matches: bool = False
    # Tool-specific structured arguments (e.g. temporal predicate/window).
    arguments: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "step_id": self.step_id,
            "tool_name": self.tool_name,
            "label": self.label,
            "requires_prior_matches": self.requires_prior_matches,
        }
        if self.camera_ids is not None:
            payload["camera_ids"] = list(self.camera_ids)
        if self.time_bounds is not None:
            payload["time_bounds"] = {
                "start_utc": self.time_bounds.start_utc,
                "end_utc": self.time_bounds.end_utc,
            }
        if self.arguments is not None:
            payload["arguments"] = dict(self.arguments)
        return payload


def build_deterministic_plan(
    *,
    scenario: str,
    authorized_cameras: tuple[str, ...],
    time_bounds: Interval,
    force_tool_name: str | None = None,
    force_camera_ids: tuple[str, ...] | None = None,
    force_time_bounds: Interval | None = None,
    include_follow_up: bool = False,
    search_label: str | None = "person",
) -> list[PlanStep]:
    """Generate valid plan steps from controlled fixtures.

    All planning behavior here is deterministic test behavior. It does not
    call a model, create_react_agent, create_deep_agent, or a second planning
    loop.
    """
    primary_tool = force_tool_name or "search_events"
    if primary_tool not in ALLOWED_TOOLS and force_tool_name is None:
        raise ValueError(f"planner may only propose allowlisted tools: {primary_tool}")

    label = None if scenario == "error_path" else search_label
    primary = PlanStep(
        step_id="step-1",
        tool_name=primary_tool,
        label=label,
        camera_ids=force_camera_ids or authorized_cameras[:1],
        time_bounds=force_time_bounds or time_bounds,
        requires_prior_matches=False,
    )
    steps = [primary]
    if include_follow_up:
        steps.append(
            PlanStep(
                step_id="step-2",
                tool_name="search_related_events",
                label="vehicle",
                camera_ids=authorized_cameras[:1],
                time_bounds=time_bounds,
                requires_prior_matches=True,
            )
        )
    return steps
