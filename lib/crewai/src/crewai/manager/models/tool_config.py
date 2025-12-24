"""Tool configuration model for serializable tool configs."""

from datetime import datetime
from typing import Any, Literal
import uuid

from pydantic import BaseModel, Field


class ToolConfig(BaseModel):
    """Serializable configuration for a CrewAI Tool.

    This model captures tool configurations in a serializable format,
    allowing tools to be registered, stored, and managed programmatically.

    Attributes:
        id: Unique identifier for the tool config
        name: Display name for the tool
        description: Description of what the tool does
        tool_type: Type of tool (builtin, crewai_tools, custom, mcp)
        class_name: Class name for builtin/crewai_tools imports
        module_path: Module path for imports
        function_ref: Function reference name for custom tools
        args_schema: JSON schema for tool arguments
        env_vars: Required environment variables
        init_kwargs: Initialization keyword arguments
        result_as_answer: Whether tool result should be the final answer
        max_usage_count: Maximum times this tool can be used
        cache_function: Cache function name
        created_at: When the config was created
        updated_at: When the config was last updated
        tags: Optional tags for categorization
        metadata: Additional custom metadata
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Display name for the tool")
    description: str = Field(..., description="Description of the tool")

    # Tool type and source
    tool_type: Literal["builtin", "crewai_tools", "custom", "mcp"] = Field(
        default="crewai_tools", description="Type of tool"
    )

    # For builtin/crewai_tools imports
    class_name: str | None = Field(
        default=None, description="Class name to import"
    )
    module_path: str | None = Field(
        default=None, description="Module path for import"
    )

    # For custom tools
    function_ref: str | None = Field(
        default=None, description="Function reference name"
    )

    # Schema and configuration
    args_schema: dict[str, Any] | None = Field(
        default=None, description="JSON schema for tool arguments"
    )
    env_vars: list[str] = Field(
        default_factory=list, description="Required environment variables"
    )
    init_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Initialization arguments"
    )

    # Behavior
    result_as_answer: bool = Field(
        default=False, description="Tool result is final answer"
    )
    max_usage_count: int | None = Field(
        default=None, description="Maximum usage count"
    )
    cache_function: str | None = Field(
        default=None, description="Cache function name"
    )

    # MCP-specific configuration
    mcp_server_url: str | None = Field(
        default=None, description="MCP server URL"
    )
    mcp_server_command: str | None = Field(
        default=None, description="MCP server command (for stdio)"
    )
    mcp_server_args: list[str] = Field(
        default_factory=list, description="MCP server command arguments"
    )
    mcp_server_env: dict[str, str] = Field(
        default_factory=dict, description="MCP server environment variables"
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
