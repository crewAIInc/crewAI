from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from crewai.agent import Agent
    from crewai.agents.agent_builder.base_agent import BaseAgent
    from crewai.crew import Crew
    from crewai.task import Task
    from crewai.tools.structured_tool import CrewStructuredTool


class ToolCallHookContext:
    """Context object passed to tool call hooks.

    Provides hooks with access to the tool being called, its input,
    the agent/task/crew context, and the result (for after hooks).

    Attributes:
        tool_name: Name of the tool being called
        tool_input: Tool input parameters (mutable dict).
            Can be modified in-place by before_tool_call hooks.
            IMPORTANT: Modify in-place (e.g., context.tool_input['key'] = value).
            Do NOT replace the dict (e.g., context.tool_input = {}), as this
            will not affect the actual tool execution.
        tool: Reference to the CrewStructuredTool instance
        agent: Agent executing the tool (may be None)
        task: Current task being executed (may be None)
        crew: Crew instance (may be None)
        tool_result: Tool execution result (only set for after_tool_call hooks).
            Can be modified by returning a new string from after_tool_call hook.
    """

    def __init__(
        self,
        tool_name: str,
        tool_input: dict[str, Any],
        tool: CrewStructuredTool,
        agent: Agent | BaseAgent | None = None,
        task: Task | None = None,
        crew: Crew | None = None,
        tool_result: str | None = None,
    ) -> None:
        """Initialize tool call hook context.

        Args:
            tool_name: Name of the tool being called
            tool_input: Tool input parameters (mutable)
            tool: Tool instance reference
            agent: Optional agent executing the tool
            task: Optional current task
            crew: Optional crew instance
            tool_result: Optional tool result (for after hooks)
        """
        self.tool_name = tool_name
        self.tool_input = tool_input
        self.tool = tool
        self.agent = agent
        self.task = task
        self.crew = crew
        self.tool_result = tool_result


# Global hook registries
_before_tool_call_hooks: list[Callable[[ToolCallHookContext], bool | None]] = []
_after_tool_call_hooks: list[Callable[[ToolCallHookContext], str | None]] = []


def register_before_tool_call_hook(
    hook: Callable[[ToolCallHookContext], bool | None],
) -> None:
    """Register a global before_tool_call hook.

    Global hooks are added to all tool executions automatically.
    This is a convenience function for registering hooks that should
    apply to all tool calls across all agents and crews.

    Args:
        hook: Function that receives ToolCallHookContext and can:
            - Modify tool_input in-place
            - Return False to block tool execution
            - Return True or None to allow execution
            IMPORTANT: Modify tool_input in-place (e.g., context.tool_input['key'] = value).
            Do NOT replace the dict (context.tool_input = {}), as this will not affect
            the actual tool execution.

    Example:
        >>> def log_tool_usage(context: ToolCallHookContext) -> None:
        ...     print(f"Executing tool: {context.tool_name}")
        ...     print(f"Input: {context.tool_input}")
        ...     return None  # Allow execution
        >>>
        >>> register_before_tool_call_hook(log_tool_usage)

        >>> def block_dangerous_tools(context: ToolCallHookContext) -> bool | None:
        ...     if context.tool_name == "delete_database":
        ...         print("Blocked dangerous tool execution!")
        ...         return False  # Block execution
        ...     return None  # Allow execution
        >>>
        >>> register_before_tool_call_hook(block_dangerous_tools)
    """
    _before_tool_call_hooks.append(hook)


def register_after_tool_call_hook(
    hook: Callable[[ToolCallHookContext], str | None],
) -> None:
    """Register a global after_tool_call hook.

    Global hooks are added to all tool executions automatically.
    This is a convenience function for registering hooks that should
    apply to all tool calls across all agents and crews.

    Args:
        hook: Function that receives ToolCallHookContext and can modify
            the tool result. Return modified result string or None to keep
            the original result. The tool_result is available in context.tool_result.

    Example:
        >>> def sanitize_output(context: ToolCallHookContext) -> str | None:
        ...     if context.tool_result and "SECRET_KEY" in context.tool_result:
        ...         return context.tool_result.replace("SECRET_KEY=...", "[REDACTED]")
        ...     return None  # Keep original result
        >>>
        >>> register_after_tool_call_hook(sanitize_output)

        >>> def log_tool_results(context: ToolCallHookContext) -> None:
        ...     print(f"Tool {context.tool_name} returned: {context.tool_result[:100]}")
        ...     return None  # Keep original result
        >>>
        >>> register_after_tool_call_hook(log_tool_results)
    """
    _after_tool_call_hooks.append(hook)


def get_before_tool_call_hooks() -> list[Callable[[ToolCallHookContext], bool | None]]:
    """Get all registered global before_tool_call hooks.

    Returns:
        List of registered before hooks
    """
    return _before_tool_call_hooks.copy()


def get_after_tool_call_hooks() -> list[Callable[[ToolCallHookContext], str | None]]:
    """Get all registered global after_tool_call hooks.

    Returns:
        List of registered after hooks
    """
    return _after_tool_call_hooks.copy()
