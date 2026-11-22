"""Verify the pinned LangGraph release imports in the locked environment.

Hosted tracing is disabled before LangGraph is imported. This test does not
construct or invoke a graph.
"""

import importlib.metadata
import os

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ.pop("LANGSMITH_API_KEY", None)
os.environ.pop("LANGCHAIN_API_KEY", None)

REQUIRED_SYMBOLS = (
    "StateGraph",
    "START",
    "END",
    "InMemorySaver",
    "Command",
    "interrupt",
    "RetryPolicy",
)


def test_pinned_langgraph_symbols_import():
    from langgraph.checkpoint.memory import InMemorySaver
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command, RetryPolicy, interrupt

    imported = {
        "StateGraph": StateGraph,
        "START": START,
        "END": END,
        "InMemorySaver": InMemorySaver,
        "Command": Command,
        "interrupt": interrupt,
        "RetryPolicy": RetryPolicy,
    }
    assert tuple(imported) == REQUIRED_SYMBOLS
    assert all(symbol is not None for symbol in imported.values())
    assert importlib.metadata.version("langgraph") == "1.2.24"


def test_hosted_tracing_is_disabled():
    assert os.environ["LANGSMITH_TRACING"] == "false"
    assert os.environ["LANGCHAIN_TRACING_V2"] == "false"
    assert "LANGSMITH_API_KEY" not in os.environ
    assert "LANGCHAIN_API_KEY" not in os.environ

    from langsmith import utils as langsmith_utils

    assert langsmith_utils.tracing_is_enabled() is False


def test_unapproved_agent_provider_distributions_are_absent():
    """E0 still excludes full Agent/provider stacks not authorized for this stage.

    ``langgraph-checkpoint-sqlite`` and ``sqlite-vec`` are required by G4-V1-EA
    and are positively exercised in ``test_sqlite_roundtrip.py``. Future
    authorized stages may add further packages without implying a permanent
    absence policy for every optional distribution.
    """
    absent = (
        "langchain",
        "langchain-ollama",
    )
    for distribution in absent:
        try:
            importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
        raise AssertionError(f"{distribution} is installed")
