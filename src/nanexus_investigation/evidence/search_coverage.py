"""Search coverage interpretation. G3-A §6.1 and §6.3."""

from __future__ import annotations

from dataclasses import dataclass

from nanexus_investigation.contracts.enums import SearchCoverage, ToolResultStatus
from nanexus_investigation.contracts.search import SearchExecutionReceipt
from nanexus_investigation.contracts.tool import ToolResult


@dataclass(frozen=True, slots=True)
class SearchInterpretation:
    """Deterministic meaning of a search ToolResult plus optional receipt."""

    tool_status: ToolResultStatus
    coverage: SearchCoverage | None
    allows_completed_query_absence_statement: bool
    allows_real_world_absence_statement: bool
    is_exhaustive_search: bool
    summary: str


def interpret_search(
    result: ToolResult,
    receipt: SearchExecutionReceipt | None = None,
) -> SearchInterpretation:
    """Interpret search outcomes without inventing real-world absence.

    An authorized exhausted EMPTY search may support only the narrow statement
    that the executed query returned no matching records. It never supports
    "nothing happened in the world."
    """
    coverage = result.coverage if result.coverage is not None else (
        receipt.coverage if receipt is not None else None
    )

    if receipt is not None and result.execution_id and receipt.search_execution_id:
        if receipt.match_count > 0 and result.status is ToolResultStatus.EMPTY:
            raise ValueError("EMPTY ToolResult cannot pair with a receipt that has matches")
        if receipt.error is not None and result.status is ToolResultStatus.EMPTY:
            raise ValueError("EMPTY ToolResult cannot pair with an error receipt")

    if result.status is ToolResultStatus.UNSUPPORTED:
        return SearchInterpretation(
            tool_status=result.status,
            coverage=coverage,
            allows_completed_query_absence_statement=False,
            allows_real_world_absence_statement=False,
            is_exhaustive_search=False,
            summary="unsupported search must not be described as empty or exhaustive",
        )

    if result.status is ToolResultStatus.ERROR:
        return SearchInterpretation(
            tool_status=result.status,
            coverage=coverage,
            allows_completed_query_absence_statement=False,
            allows_real_world_absence_statement=False,
            is_exhaustive_search=False,
            summary="failed search must not be converted into an absence claim",
        )

    if result.status is ToolResultStatus.PARTIAL:
        return SearchInterpretation(
            tool_status=result.status,
            coverage=coverage or SearchCoverage.PARTIAL,
            allows_completed_query_absence_statement=False,
            allows_real_world_absence_statement=False,
            is_exhaustive_search=False,
            summary="partial results are incomplete and not exhaustive",
        )

    if result.status is ToolResultStatus.EMPTY:
        exhausted = coverage is SearchCoverage.EXHAUSTED
        return SearchInterpretation(
            tool_status=result.status,
            coverage=coverage,
            allows_completed_query_absence_statement=exhausted,
            allows_real_world_absence_statement=False,
            is_exhaustive_search=exhausted,
            summary=(
                "exhausted empty search supports only that the completed query "
                "returned no matching records"
                if exhausted
                else "non-exhausted empty search is not a complete no-match result"
            ),
        )

    if result.status is ToolResultStatus.OK:
        exhaustive = coverage is SearchCoverage.EXHAUSTED
        return SearchInterpretation(
            tool_status=result.status,
            coverage=coverage,
            allows_completed_query_absence_statement=False,
            allows_real_world_absence_statement=False,
            is_exhaustive_search=exhaustive,
            summary=(
                "search returned records"
                if exhaustive
                else "search returned records with non-exhaustive coverage"
            ),
        )

    raise ValueError(f"unknown ToolResult status: {result.status}")


def rejects_real_world_absence(interpretation: SearchInterpretation) -> bool:
    """Invariant: empty structured search never proves real-world absence."""
    return interpretation.allows_real_world_absence_statement is False
