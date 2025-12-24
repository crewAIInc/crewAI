"""MCP server configuration model for serializable configs."""

from datetime import datetime
from typing import Any, Literal
import uuid

from pydantic import BaseModel, Field


class MCPServerConfig(BaseModel):
    """Serializable configuration for an MCP (Model Context Protocol) Server.

    This model captures MCP server configurations in a serializable format,
    allowing MCP servers to be stored, loaded, and managed programmatically.

    Supports three transport types:
    - stdio: Local process communication
    - http: HTTP-based communication
    - sse: Server-Sent Events

    Attributes:
        id: Unique identifier for the MCP server config
        name: Server name (used as tool prefix)
        transport: Transport type (stdio, http, sse)
        command: Command to execute (for stdio)
        args: Command arguments (for stdio)
        env: Environment variables (for stdio)
        url: Server URL (for http/sse)
        headers: HTTP headers (for http/sse)
        streamable: Use streamable HTTP (for http only)
        allowed_tools: Whitelist of tool names
        blocked_tools: Blacklist of tool names
        cache_tools_list: Whether to cache tool list
        connection_timeout: Connection timeout in seconds
        execution_timeout: Execution timeout in seconds
        created_at: When the config was created
        updated_at: When the config was last updated
        tags: Optional tags for categorization
        metadata: Additional custom metadata
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Server name (used as tool prefix)")

    # Transport type
    transport: Literal["stdio", "http", "sse"] = Field(
        ..., description="Transport type"
    )

    # Stdio transport configuration
    command: str | None = Field(
        default=None, description="Command to execute (for stdio)"
    )
    args: list[str] = Field(
        default_factory=list, description="Command arguments (for stdio)"
    )
    env: dict[str, str] = Field(
        default_factory=dict, description="Environment variables (for stdio)"
    )

    # HTTP/SSE transport configuration
    url: str | None = Field(
        default=None, description="Server URL (for http/sse)"
    )
    headers: dict[str, str] = Field(
        default_factory=dict, description="HTTP headers (for http/sse)"
    )
    streamable: bool = Field(
        default=True, description="Use streamable HTTP (for http only)"
    )

    # Tool filtering
    allowed_tools: list[str] = Field(
        default_factory=list, description="Whitelist of tool names"
    )
    blocked_tools: list[str] = Field(
        default_factory=list, description="Blacklist of tool names"
    )

    # Caching and timeouts
    cache_tools_list: bool = Field(
        default=True, description="Cache tool list"
    )
    connection_timeout: int = Field(
        default=30, description="Connection timeout in seconds"
    )
    execution_timeout: int = Field(
        default=30, description="Execution timeout in seconds"
    )

    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    tags: list[str] = Field(default_factory=list, description="Tags for categorization")
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional custom metadata"
    )

    model_config = {"extra": "allow"}

    def update_timestamp(self) -> None:
        """Update the updated_at timestamp."""
        self.updated_at = datetime.now()
