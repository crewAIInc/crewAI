"""Tool filtering support for MCP servers.

This module provides utilities for filtering tools from MCP servers,
including static allow/block lists and dynamic context-aware filtering.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field


if TYPE_CHECKING:
    pass


class ToolFilterContext(BaseModel):
    """Context for dynamic tool filtering.

    This context is passed to dynamic tool filters to provide
    information about the agent, run context, and server.
    """

    agent: Any = Field(..., description="The agent requesting tools.")
    server_name: str = Field(..., description="Name of the MCP server.")
    run_context: dict[str, Any] | None = Field(
        default=None,
        description="Optional run context for additional filtering logic.",
    )


# Type alias for tool filter functions
ToolFilter = (
    Callable[[ToolFilterContext, dict[str, Any]], bool]
    | Callable[[dict[str, Any]], bool]
)


class StaticToolFilter:
    """Static tool filter with allow/block lists.

    This filter provides simple allow/block list filtering based on
    tool names. Useful for restricting which tools are available
    from an MCP server.

    Example:
        ```python
        filter = StaticToolFilter(
            allowed_tool_names=["read_file", "write_file"],
            blocked_tool_names=["delete_file"],
        )
        ```
    """

    def __init__(
        self,
        allowed_tool_names: list[str] | None = None,
        blocked_tool_names: list[str] | None = None,
    ) -> None:
        """Initialize static tool filter.

        Args:
            allowed_tool_names: List of tool names to allow. If None,
                all tools are allowed (unless blocked).
            blocked_tool_names: List of tool names to block. Blocked tools
                take precedence over allowed tools.
        """
        self.allowed_tool_names = set(allowed_tool_names or [])
        self.blocked_tool_names = set(blocked_tool_names or [])

    def __call__(self, tool: dict[str, Any]) -> bool:
        """Filter tool based on allow/block lists.

        Args:
            tool: Tool definition dictionary with at least 'name' key.

        Returns:
            True if tool should be included, False otherwise.
        """
        tool_name = tool.get("name", "")

        # Blocked tools take precedence
        if self.blocked_tool_names and tool_name in self.blocked_tool_names:
            return False

        # If allow list exists, tool must be in it
        if self.allowed_tool_names:
            return tool_name in self.allowed_tool_names

        # No restrictions - allow all
        return True


def create_static_tool_filter(
    allowed_tool_names: list[str] | None = None,
    blocked_tool_names: list[str] | None = None,
) -> Callable[[dict[str, Any]], bool]:
    """Create a static tool filter function.

    This is a convenience function for creating static tool filters
    with allow/block lists.

    Args:
        allowed_tool_names: List of tool names to allow. If None,
            all tools are allowed (unless blocked).
        blocked_tool_names: List of tool names to block. Blocked tools
            take precedence over allowed tools.

    Returns:
        Tool filter function that returns True for allowed tools.

    Example:
        ```python
        filter_fn = create_static_tool_filter(
            allowed_tool_names=["read_file", "write_file"],
            blocked_tool_names=["delete_file"],
        )

        # Use in MCPServerStdio
        mcp_server = MCPServerStdio(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem"],
            tool_filter=filter_fn,
        )
        ```
    """
    return StaticToolFilter(
        allowed_tool_names=allowed_tool_names,
        blocked_tool_names=blocked_tool_names,
    )


def create_dynamic_tool_filter(
    filter_func: Callable[[ToolFilterContext, dict[str, Any]], bool],
) -> Callable[[ToolFilterContext, dict[str, Any]], bool]:
    """Create a dynamic tool filter function.

    This function wraps a dynamic filter function that has access
    to the tool filter context (agent, server, run context).

    Args:
        filter_func: Function that takes (context, tool) and returns bool.

    Returns:
        Tool filter function that can be used with MCP server configs.

    Example:
        ```python
        async def context_aware_filter(
            context: ToolFilterContext, tool: dict[str, Any]
        ) -> bool:
            # Block dangerous tools for code reviewers
            if context.agent.role == "Code Reviewer":
                if tool["name"].startswith("danger_"):
                    return False
            return True


        filter_fn = create_dynamic_tool_filter(context_aware_filter)

        mcp_server = MCPServerStdio(
            command="python", args=["server.py"], tool_filter=filter_fn
        )
        ```
    """
    return filter_func
