"""MCP (Model Context Protocol) client support for CrewAI agents.

This module provides native MCP client functionality, allowing CrewAI agents
to connect to any MCP-compliant server using various transport types.

Heavy imports (MCPClient, MCPToolResolver, BaseTransport, TransportType) are
lazy-loaded on first access to avoid pulling in the ``mcp`` SDK (~400ms)
when only lightweight config/filter types are needed.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from crewai.mcp.config import (
    MCPServerConfig,
    MCPServerHTTP,
    MCPServerSSE,
    MCPServerStdio,
)
from crewai.mcp.filters import (
    StaticToolFilter,
    ToolFilter,
    ToolFilterContext,
    create_dynamic_tool_filter,
    create_static_tool_filter,
)

if TYPE_CHECKING:
    from crewai.mcp.client import MCPClient
    from crewai.mcp.tool_resolver import MCPToolResolver
    from crewai.mcp.transports.base import BaseTransport, TransportType

_LAZY: dict[str, tuple[str, str]] = {
    "MCPClient": ("crewai.mcp.client", "MCPClient"),
    "MCPToolResolver": ("crewai.mcp.tool_resolver", "MCPToolResolver"),
    "BaseTransport": ("crewai.mcp.transports.base", "BaseTransport"),
    "TransportType": ("crewai.mcp.transports.base", "TransportType"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        mod_path, attr = _LAZY[name]
        mod = importlib.import_module(mod_path)
        val = getattr(mod, attr)
        globals()[name] = val  # cache for subsequent access
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "BaseTransport",
    "MCPClient",
    "MCPServerConfig",
    "MCPServerHTTP",
    "MCPServerSSE",
    "MCPServerStdio",
    "MCPToolResolver",
    "StaticToolFilter",
    "ToolFilter",
    "ToolFilterContext",
    "TransportType",
    "create_dynamic_tool_filter",
    "create_static_tool_filter",
]
