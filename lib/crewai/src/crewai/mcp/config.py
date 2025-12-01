"""MCP server configuration models for CrewAI agents.

This module provides Pydantic models for configuring MCP servers with
various transport types, similar to OpenAI's Agents SDK.
"""

import inspect
from typing import Literal

from pydantic import BaseModel, Field, PrivateAttr, model_validator

from crewai.mcp.filters import ToolFilter


def _count_required_params(sig: inspect.Signature) -> int:
    """Count only required positional parameters (no defaults, no *args/**kwargs).

    This is used to distinguish between:
    - Static filters: 1 required param (tool: dict)
    - Dynamic filters: 2 required params (context: ToolFilterContext, tool: dict)
    - Invalid filters: 0 or 3+ params (treated as no filter)
    """
    count = 0
    for param in sig.parameters.values():
        # Skip *args and **kwargs
        if param.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue
        # Count only parameters without defaults
        if param.default is inspect.Parameter.empty:
            count += 1
    return count


def _determine_filter_type(
    tool_filter: "ToolFilter | None",
) -> Literal["dynamic", "static", "none"]:
    """Determine the filter type based on function signature.

    Returns:
        "dynamic" for 2-param filters (context, tool)
        "static" for 1-param filters (tool)
        "none" for no filter or invalid signatures (0 or 3+ params)
    """
    if tool_filter is None:
        return "none"

    if not callable(tool_filter):
        return "none"

    sig = inspect.signature(tool_filter)
    num_required = _count_required_params(sig)

    if num_required == 2:
        return "dynamic"
    elif num_required == 1:
        return "static"
    else:
        # 0 or 3+ params is invalid - treat as no filter to avoid errors
        # This handles edge cases like lambda: True or malformed filters
        return "none"


class MCPServerStdio(BaseModel):
    """Stdio MCP server configuration.

    This configuration is used for connecting to local MCP servers
    that run as processes and communicate via standard input/output.

    Example:
        ```python
        mcp_server = MCPServerStdio(
            command="python",
            args=["path/to/server.py"],
            env={"API_KEY": "..."},
            tool_filter=create_static_tool_filter(
                allowed_tool_names=["read_file", "write_file"]
            ),
        )
        ```
    """

    command: str = Field(
        ...,
        description="Command to execute (e.g., 'python', 'node', 'npx', 'uvx').",
    )
    args: list[str] = Field(
        default_factory=list,
        description="Command arguments (e.g., ['server.py'] or ['-y', '@mcp/server']).",
    )
    env: dict[str, str] | None = Field(
        default=None,
        description="Environment variables to pass to the process.",
    )
    tool_filter: ToolFilter | None = Field(
        default=None,
        description="Optional tool filter for filtering available tools.",
    )
    cache_tools_list: bool = Field(
        default=False,
        description="Whether to cache the tool list for faster subsequent access.",
    )
    _filter_type: Literal["dynamic", "static", "none"] = PrivateAttr(default="none")

    @model_validator(mode="after")
    def _cache_filter_type(self):
        """Cache the filter type for performance optimization."""
        self._filter_type = _determine_filter_type(self.tool_filter)
        return self


class MCPServerHTTP(BaseModel):
    """HTTP/Streamable HTTP MCP server configuration.

    This configuration is used for connecting to remote MCP servers
    over HTTP/HTTPS using streamable HTTP transport.

    Example:
        ```python
        mcp_server = MCPServerHTTP(
            url="https://api.example.com/mcp",
            headers={"Authorization": "Bearer ..."},
            cache_tools_list=True,
        )
        ```
    """

    url: str = Field(
        ..., description="Server URL (e.g., 'https://api.example.com/mcp')."
    )
    headers: dict[str, str] | None = Field(
        default=None,
        description="Optional HTTP headers for authentication or other purposes.",
    )
    streamable: bool = Field(
        default=True,
        description="Whether to use streamable HTTP transport (default: True).",
    )
    tool_filter: ToolFilter | None = Field(
        default=None,
        description="Optional tool filter for filtering available tools.",
    )
    cache_tools_list: bool = Field(
        default=False,
        description="Whether to cache the tool list for faster subsequent access.",
    )
    _filter_type: Literal["dynamic", "static", "none"] = PrivateAttr(default="none")

    @model_validator(mode="after")
    def _cache_filter_type(self):
        """Cache the filter type for performance optimization."""
        self._filter_type = _determine_filter_type(self.tool_filter)
        return self


class MCPServerSSE(BaseModel):
    """Server-Sent Events (SSE) MCP server configuration.

    This configuration is used for connecting to remote MCP servers
    using Server-Sent Events for real-time streaming communication.

    Example:
        ```python
        mcp_server = MCPServerSSE(
            url="https://api.example.com/mcp/sse",
            headers={"Authorization": "Bearer ..."},
        )
        ```
    """

    url: str = Field(
        ...,
        description="Server URL (e.g., 'https://api.example.com/mcp/sse').",
    )
    headers: dict[str, str] | None = Field(
        default=None,
        description="Optional HTTP headers for authentication or other purposes.",
    )
    tool_filter: ToolFilter | None = Field(
        default=None,
        description="Optional tool filter for filtering available tools.",
    )
    cache_tools_list: bool = Field(
        default=False,
        description="Whether to cache the tool list for faster subsequent access.",
    )
    _filter_type: Literal["dynamic", "static", "none"] = PrivateAttr(default="none")

    @model_validator(mode="after")
    def _cache_filter_type(self):
        """Cache the filter type for performance optimization."""
        self._filter_type = _determine_filter_type(self.tool_filter)
        return self


# Type alias for all MCP server configurations
MCPServerConfig = MCPServerStdio | MCPServerHTTP | MCPServerSSE
