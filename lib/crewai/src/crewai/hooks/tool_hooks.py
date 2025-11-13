from __future__ import annotations

from typing import TYPE_CHECKING, Any

from crewai.events.event_listener import event_listener
from crewai.hooks.types import AfterToolCallHookType, BeforeToolCallHookType
from crewai.utilities.printer import Printer


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

    def request_human_input(
        self,
        prompt: str,
        default_message: str = "Press Enter to continue, or provide feedback:",
    ) -> str:
        """Request human input during tool hook execution.

        This method pauses live console updates, displays a prompt to the user,
        waits for their input, and then resumes live updates. This is useful for
        approval gates, reviewing tool results, or getting human feedback during execution.

        Args:
            prompt: Custom message to display to the user
            default_message: Message shown after the prompt

        Returns:
            User's input as a string (empty string if just Enter pressed)

        Example:
            >>> def approval_hook(context: ToolCallHookContext) -> bool | None:
            ...     if context.tool_name == "delete_file":
            ...         response = context.request_human_input(
            ...             prompt="Allow file deletion?",
            ...             default_message="Type 'approve' to continue:",
            ...         )
            ...         if response.lower() != "approve":
            ...             return False  # Block execution
            ...     return None  # Allow execution
        """

        printer = Printer()
        event_listener.formatter.pause_live_updates()

        try:
            printer.print(content=f"\n{prompt}", color="bold_yellow")
            printer.print(content=default_message, color="cyan")
            response = input().strip()

            if response:
                printer.print(content="\nProcessing your input...", color="cyan")

            return response
        finally:
            event_listener.formatter.resume_live_updates()


# Global hook registries
_before_tool_call_hooks: list[BeforeToolCallHookType] = []
_after_tool_call_hooks: list[AfterToolCallHookType] = []


def register_before_tool_call_hook(
    hook: BeforeToolCallHookType,
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
    hook: AfterToolCallHookType,
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


def get_before_tool_call_hooks() -> list[BeforeToolCallHookType]:
    """Get all registered global before_tool_call hooks.

    Returns:
        List of registered before hooks
    """
    return _before_tool_call_hooks.copy()


def get_after_tool_call_hooks() -> list[AfterToolCallHookType]:
    """Get all registered global after_tool_call hooks.

    Returns:
        List of registered after hooks
    """
    return _after_tool_call_hooks.copy()


def unregister_before_tool_call_hook(
    hook: BeforeToolCallHookType,
) -> bool:
    """Unregister a specific global before_tool_call hook.

    Args:
        hook: The hook function to remove

    Returns:
        True if the hook was found and removed, False otherwise

    Example:
        >>> def my_hook(context: ToolCallHookContext) -> None:
        ...     print("Before tool call")
        >>>
        >>> register_before_tool_call_hook(my_hook)
        >>> unregister_before_tool_call_hook(my_hook)
        True
    """
    try:
        _before_tool_call_hooks.remove(hook)
        return True
    except ValueError:
        return False


def unregister_after_tool_call_hook(
    hook: AfterToolCallHookType,
) -> bool:
    """Unregister a specific global after_tool_call hook.

    Args:
        hook: The hook function to remove

    Returns:
        True if the hook was found and removed, False otherwise

    Example:
        >>> def my_hook(context: ToolCallHookContext) -> str | None:
        ...     return None
        >>>
        >>> register_after_tool_call_hook(my_hook)
        >>> unregister_after_tool_call_hook(my_hook)
        True
    """
    try:
        _after_tool_call_hooks.remove(hook)
        return True
    except ValueError:
        return False


def clear_before_tool_call_hooks() -> int:
    """Clear all registered global before_tool_call hooks.

    Returns:
        Number of hooks that were cleared

    Example:
        >>> register_before_tool_call_hook(hook1)
        >>> register_before_tool_call_hook(hook2)
        >>> clear_before_tool_call_hooks()
        2
    """
    count = len(_before_tool_call_hooks)
    _before_tool_call_hooks.clear()
    return count


def clear_after_tool_call_hooks() -> int:
    """Clear all registered global after_tool_call hooks.

    Returns:
        Number of hooks that were cleared

    Example:
        >>> register_after_tool_call_hook(hook1)
        >>> register_after_tool_call_hook(hook2)
        >>> clear_after_tool_call_hooks()
        2
    """
    count = len(_after_tool_call_hooks)
    _after_tool_call_hooks.clear()
    return count


def clear_all_tool_call_hooks() -> tuple[int, int]:
    """Clear all registered global tool call hooks (both before and after).

    Returns:
        Tuple of (before_hooks_cleared, after_hooks_cleared)

    Example:
        >>> register_before_tool_call_hook(before_hook)
        >>> register_after_tool_call_hook(after_hook)
        >>> clear_all_tool_call_hooks()
        (1, 1)
    """
    before_count = clear_before_tool_call_hooks()
    after_count = clear_after_tool_call_hooks()
    return (before_count, after_count)
