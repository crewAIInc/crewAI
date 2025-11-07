from datetime import datetime
from typing import Any

from crewai.events.base_events import BaseEvent


class MCPEvent(BaseEvent):
    """Base event for MCP operations."""

    server_name: str
    server_url: str | None = None
    transport_type: str | None = None  # "stdio", "http", "sse"
    agent_id: str | None = None
    agent_role: str | None = None
    from_agent: Any | None = None
    from_task: Any | None = None

    def __init__(self, **data):
        super().__init__(**data)
        self._set_agent_params(data)
        self._set_task_params(data)


class MCPConnectionStartedEvent(MCPEvent):
    """Event emitted when starting to connect to an MCP server."""

    type: str = "mcp_connection_started"
    connect_timeout: int | None = None
    is_reconnect: bool = (
        False  # True if this is a reconnection, False for first connection
    )


class MCPConnectionCompletedEvent(MCPEvent):
    """Event emitted when successfully connected to an MCP server."""

    type: str = "mcp_connection_completed"
    started_at: datetime | None = None
    completed_at: datetime | None = None
    connection_duration_ms: float | None = None
    is_reconnect: bool = (
        False  # True if this was a reconnection, False for first connection
    )


class MCPConnectionFailedEvent(MCPEvent):
    """Event emitted when connection to an MCP server fails."""

    type: str = "mcp_connection_failed"
    error: str
    error_type: str | None = None  # "timeout", "authentication", "network", etc.
    started_at: datetime | None = None
    failed_at: datetime | None = None


class MCPToolExecutionStartedEvent(MCPEvent):
    """Event emitted when starting to execute an MCP tool."""

    type: str = "mcp_tool_execution_started"
    tool_name: str
    tool_args: dict[str, Any] | None = None


class MCPToolExecutionCompletedEvent(MCPEvent):
    """Event emitted when MCP tool execution completes."""

    type: str = "mcp_tool_execution_completed"
    tool_name: str
    tool_args: dict[str, Any] | None = None
    result: Any | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    execution_duration_ms: float | None = None


class MCPToolExecutionFailedEvent(MCPEvent):
    """Event emitted when MCP tool execution fails."""

    type: str = "mcp_tool_execution_failed"
    tool_name: str
    tool_args: dict[str, Any] | None = None
    error: str
    error_type: str | None = None  # "timeout", "validation", "server_error", etc.
    started_at: datetime | None = None
    failed_at: datetime | None = None
