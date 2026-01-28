from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

from crewai.events.event_listener import event_listener
from crewai.hooks.types import AfterLLMCallHookType, BeforeLLMCallHookType
from crewai.utilities.printer import Printer


if TYPE_CHECKING:
    from crewai.agents.crew_agent_executor import CrewAgentExecutor
    from crewai.experimental.agent_executor import AgentExecutor
    from crewai.lite_agent import LiteAgent
    from crewai.llms.base_llm import BaseLLM
    from crewai.utilities.types import LLMMessage


class LLMCallHookContext:
    """Context object passed to LLM call hooks.

    Provides hooks with complete access to the execution state, allowing
    modification of messages, responses, and executor attributes.

    Supports both executor-based calls (agents in crews/flows) and direct LLM calls.

    Attributes:
        executor: Reference to the executor (CrewAgentExecutor/LiteAgent) or None for direct calls
        messages: Direct reference to messages (mutable list).
            Can be modified in both before_llm_call and after_llm_call hooks.
            Modifications in after_llm_call hooks persist to the next iteration,
            allowing hooks to modify conversation history for subsequent LLM calls.
            IMPORTANT: Modify messages in-place (e.g., append, extend, remove items).
            Do NOT replace the list (e.g., context.messages = []), as this will break
            the executor. Use context.messages.append() or context.messages.extend()
            instead of assignment.
        agent: Reference to the agent executing the task (None for direct LLM calls)
        task: Reference to the task being executed (None for direct LLM calls or LiteAgent)
        crew: Reference to the crew instance (None for direct LLM calls or LiteAgent)
        llm: Reference to the LLM instance
        iterations: Current iteration count (0 for direct LLM calls)
        response: LLM response string (only set for after_llm_call hooks).
            Can be modified by returning a new string from after_llm_call hook.
    """

    executor: CrewAgentExecutor | AgentExecutor | LiteAgent | None
    messages: list[LLMMessage]
    agent: Any
    task: Any
    crew: Any
    llm: BaseLLM | None | str | Any
    iterations: int
    response: str | None

    def __init__(
        self,
        executor: CrewAgentExecutor | AgentExecutor | LiteAgent | None = None,
        response: str | None = None,
        messages: list[LLMMessage] | None = None,
        llm: BaseLLM | str | Any | None = None,  # TODO: look into
        agent: Any | None = None,
        task: Any | None = None,
        crew: Any | None = None,
    ) -> None:
        """Initialize hook context with executor reference or direct parameters.

        Args:
            executor: The CrewAgentExecutor or LiteAgent instance (None for direct LLM calls)
            response: Optional response string (for after_llm_call hooks)
            messages: Optional messages list (for direct LLM calls when executor is None)
            llm: Optional LLM instance (for direct LLM calls when executor is None)
            agent: Optional agent reference (for direct LLM calls when executor is None)
            task: Optional task reference (for direct LLM calls when executor is None)
            crew: Optional crew reference (for direct LLM calls when executor is None)
        """
        if executor is not None:
            # Existing path: extract from executor
            self.executor = executor
            self.messages = executor.messages
            self.llm = executor.llm
            self.iterations = executor.iterations
            # Handle CrewAgentExecutor vs LiteAgent differences
            if hasattr(executor, "agent"):
                self.agent = executor.agent
                self.task = cast("CrewAgentExecutor", executor).task
                self.crew = cast("CrewAgentExecutor", executor).crew
            else:
                # LiteAgent case - is the agent itself, doesn't have task/crew
                self.agent = (
                    executor.original_agent
                    if hasattr(executor, "original_agent")
                    else executor
                )
                self.task = None
                self.crew = None
        else:
            # New path: direct LLM call with explicit parameters
            self.executor = None
            self.messages = messages or []
            self.llm = llm
            self.agent = agent
            self.task = task
            self.crew = crew
            self.iterations = 0

        self.response = response

    def request_human_input(
        self,
        prompt: str,
        default_message: str = "Press Enter to continue, or provide feedback:",
    ) -> str:
        """Request human input during LLM hook execution.

        This method pauses live console updates, displays a prompt to the user,
        waits for their input, and then resumes live updates. This is useful for
        approval gates, debugging, or getting human feedback during execution.

        Args:
            prompt: Custom message to display to the user
            default_message: Message shown after the prompt

        Returns:
            User's input as a string (empty string if just Enter pressed)

        Example:
            >>> def approval_hook(context: LLMCallHookContext) -> None:
            ...     if context.iterations > 5:
            ...         response = context.request_human_input(
            ...             prompt="Allow this LLM call?",
            ...             default_message="Type 'no' to skip, or press Enter:",
            ...         )
            ...         if response.lower() == "no":
            ...             print("LLM call skipped by user")
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


_before_llm_call_hooks: list[BeforeLLMCallHookType] = []
_after_llm_call_hooks: list[AfterLLMCallHookType] = []


def register_before_llm_call_hook(
    hook: BeforeLLMCallHookType,
) -> None:
    """Register a global before_llm_call hook.

    Global hooks are added to all executors automatically.
    This is a convenience function for registering hooks that should
    apply to all LLM calls across all executors.

    Args:
        hook: Function that receives LLMCallHookContext and can:
            - Modify context.messages directly (in-place)
            - Return False to block LLM execution
            - Return True or None to allow execution
            IMPORTANT: Modify messages in-place (append, extend, remove items).
            Do NOT replace the list (context.messages = []), as this will break execution.

    Example:
        >>> def log_llm_calls(context: LLMCallHookContext) -> None:
        ...     print(f"LLM call by {context.agent.role}")
        ...     print(f"Messages: {len(context.messages)}")
        ...     return None  # Allow execution
        >>>
        >>> register_before_llm_call_hook(log_llm_calls)
        >>>
        >>> def block_excessive_iterations(context: LLMCallHookContext) -> bool | None:
        ...     if context.iterations > 10:
        ...         print("Blocked: Too many iterations")
        ...         return False  # Block execution
        ...     return None  # Allow execution
        >>>
        >>> register_before_llm_call_hook(block_excessive_iterations)
    """
    _before_llm_call_hooks.append(hook)


def register_after_llm_call_hook(
    hook: AfterLLMCallHookType,
) -> None:
    """Register a global after_llm_call hook.

    Global hooks are added to all executors automatically.
    This is a convenience function for registering hooks that should
    apply to all LLM calls across all executors.

    Args:
        hook: Function that receives LLMCallHookContext and can modify:
            - The response: Return modified response string or None to keep original
            - The messages: Modify context.messages directly (mutable reference)
            Both modifications are supported and can be used together.
            IMPORTANT: Modify messages in-place (append, extend, remove items).
            Do NOT replace the list (context.messages = []), as this will break execution.

    Example:
        >>> def sanitize_response(context: LLMCallHookContext) -> str | None:
        ...     if context.response and "SECRET" in context.response:
        ...         return context.response.replace("SECRET", "[REDACTED]")
        ...     return None
        >>>
        >>> register_after_llm_call_hook(sanitize_response)
    """
    _after_llm_call_hooks.append(hook)


def get_before_llm_call_hooks() -> list[BeforeLLMCallHookType]:
    """Get all registered global before_llm_call hooks.

    Returns:
        List of registered before hooks
    """
    return _before_llm_call_hooks.copy()


def get_after_llm_call_hooks() -> list[AfterLLMCallHookType]:
    """Get all registered global after_llm_call hooks.

    Returns:
        List of registered after hooks
    """
    return _after_llm_call_hooks.copy()


def unregister_before_llm_call_hook(
    hook: BeforeLLMCallHookType,
) -> bool:
    """Unregister a specific global before_llm_call hook.

    Args:
        hook: The hook function to remove

    Returns:
        True if the hook was found and removed, False otherwise

    Example:
        >>> def my_hook(context: LLMCallHookContext) -> None:
        ...     print("Before LLM call")
        >>>
        >>> register_before_llm_call_hook(my_hook)
        >>> unregister_before_llm_call_hook(my_hook)
        True
    """
    try:
        _before_llm_call_hooks.remove(hook)
        return True
    except ValueError:
        return False


def unregister_after_llm_call_hook(
    hook: AfterLLMCallHookType,
) -> bool:
    """Unregister a specific global after_llm_call hook.

    Args:
        hook: The hook function to remove

    Returns:
        True if the hook was found and removed, False otherwise

    Example:
        >>> def my_hook(context: LLMCallHookContext) -> str | None:
        ...     return None
        >>>
        >>> register_after_llm_call_hook(my_hook)
        >>> unregister_after_llm_call_hook(my_hook)
        True
    """
    try:
        _after_llm_call_hooks.remove(hook)
        return True
    except ValueError:
        return False


def clear_before_llm_call_hooks() -> int:
    """Clear all registered global before_llm_call hooks.

    Returns:
        Number of hooks that were cleared

    Example:
        >>> register_before_llm_call_hook(hook1)
        >>> register_before_llm_call_hook(hook2)
        >>> clear_before_llm_call_hooks()
        2
    """
    count = len(_before_llm_call_hooks)
    _before_llm_call_hooks.clear()
    return count


def clear_after_llm_call_hooks() -> int:
    """Clear all registered global after_llm_call hooks.

    Returns:
        Number of hooks that were cleared

    Example:
        >>> register_after_llm_call_hook(hook1)
        >>> register_after_llm_call_hook(hook2)
        >>> clear_after_llm_call_hooks()
        2
    """
    count = len(_after_llm_call_hooks)
    _after_llm_call_hooks.clear()
    return count


def clear_all_llm_call_hooks() -> tuple[int, int]:
    """Clear all registered global LLM call hooks (both before and after).

    Returns:
        Tuple of (before_hooks_cleared, after_hooks_cleared)

    Example:
        >>> register_before_llm_call_hook(before_hook)
        >>> register_after_llm_call_hook(after_hook)
        >>> clear_all_llm_call_hooks()
        (1, 1)
    """
    before_count = clear_before_llm_call_hooks()
    after_count = clear_after_llm_call_hooks()
    return (before_count, after_count)
