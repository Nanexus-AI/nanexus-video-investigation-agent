"""Minimal tool registry protocol shared by stub and VMS adapter backends."""

from __future__ import annotations

from typing import Protocol

from nanexus_investigation.contracts.tool import ToolRequest, ToolResult


class ToolRegistryView(Protocol):
    """Allowlisted tool dispatch surface used by Tool Guard and execute_tool."""

    def is_registered(self, tool_name: str) -> bool: ...

    def invoke(self, request: ToolRequest) -> ToolResult: ...
