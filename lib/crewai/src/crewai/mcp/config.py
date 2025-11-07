"""MCP server configuration models for CrewAI agents.

This module provides Pydantic models for configuring MCP servers with
various transport types, similar to OpenAI's Agents SDK.
"""

from pydantic import BaseModel, Field

from crewai.mcp.filters import ToolFilter


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


# Type alias for all MCP server configurations
MCPServerConfig = MCPServerStdio | MCPServerHTTP | MCPServerSSE
