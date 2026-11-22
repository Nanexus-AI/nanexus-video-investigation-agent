"""Replaceable local Model Provider surface. Proposals are untrusted."""

from nanexus_investigation.model.provider import (
    InvestigationProposal,
    ModelProvider,
    OllamaChatProvider,
    ProposalContext,
    ProposalParseError,
    StaticModelProvider,
    build_follow_up_prompt,
    build_initial_plan_prompt,
    default_tool_descriptions,
    parse_investigation_proposal,
)

__all__ = [
    "InvestigationProposal",
    "ModelProvider",
    "OllamaChatProvider",
    "ProposalContext",
    "ProposalParseError",
    "StaticModelProvider",
    "build_follow_up_prompt",
    "build_initial_plan_prompt",
    "default_tool_descriptions",
    "parse_investigation_proposal",
]
