"""MCP (Model Context Protocol) client support for CrewAI agents.

This module provides native MCP client functionality, allowing CrewAI agents
to connect to any MCP-compliant server using various transport types.
"""

from crewai.mcp.client import MCPClient
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
from crewai.mcp.transports.base import BaseTransport, TransportType


__all__ = [
    "BaseTransport",
    "MCPClient",
    "MCPServerConfig",
    "MCPServerHTTP",
    "MCPServerSSE",
    "MCPServerStdio",
    "StaticToolFilter",
    "ToolFilter",
    "ToolFilterContext",
    "TransportType",
    "create_dynamic_tool_filter",
    "create_static_tool_filter",
]
