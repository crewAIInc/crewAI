from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from crewai.agents.crew_agent_executor import CrewAgentExecutor


class LLMCallHookContext:
    """Context object passed to LLM call hooks with full executor access.

    Provides hooks with complete access to the executor state, allowing
    modification of messages, responses, and executor attributes.

    Attributes:
        executor: Full reference to the CrewAgentExecutor instance
        messages: Direct reference to executor.messages (mutable list).
            Can be modified in both before_llm_call and after_llm_call hooks.
            Modifications in after_llm_call hooks persist to the next iteration,
            allowing hooks to modify conversation history for subsequent LLM calls.
            IMPORTANT: Modify messages in-place (e.g., append, extend, remove items).
            Do NOT replace the list (e.g., context.messages = []), as this will break
            the executor. Use context.messages.append() or context.messages.extend()
            instead of assignment.
        agent: Reference to the agent executing the task
        task: Reference to the task being executed
        crew: Reference to the crew instance
        llm: Reference to the LLM instance
        iterations: Current iteration count
        response: LLM response string (only set for after_llm_call hooks).
            Can be modified by returning a new string from after_llm_call hook.
    """

    def __init__(
        self,
        executor: CrewAgentExecutor,
        response: str | None = None,
    ) -> None:
        """Initialize hook context with executor reference.

        Args:
            executor: The CrewAgentExecutor instance
            response: Optional response string (for after_llm_call hooks)
        """
        self.executor = executor
        self.messages = executor.messages
        self.agent = executor.agent
        self.task = executor.task
        self.crew = executor.crew
        self.llm = executor.llm
        self.iterations = executor.iterations
        self.response = response


# Global hook registries (optional convenience feature)
_before_llm_call_hooks: list[Callable[[LLMCallHookContext], None]] = []
_after_llm_call_hooks: list[Callable[[LLMCallHookContext], str | None]] = []


def register_before_llm_call_hook(
    hook: Callable[[LLMCallHookContext], None],
) -> None:
    """Register a global before_llm_call hook.

    Global hooks are added to all executors automatically.
    This is a convenience function for registering hooks that should
    apply to all LLM calls across all executors.

    Args:
        hook: Function that receives LLMCallHookContext and can modify
            context.messages directly. Should return None.
            IMPORTANT: Modify messages in-place (append, extend, remove items).
            Do NOT replace the list (context.messages = []), as this will break execution.
    """
    _before_llm_call_hooks.append(hook)


def register_after_llm_call_hook(
    hook: Callable[[LLMCallHookContext], str | None],
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
    """
    _after_llm_call_hooks.append(hook)


def get_before_llm_call_hooks() -> list[Callable[[LLMCallHookContext], None]]:
    """Get all registered global before_llm_call hooks.

    Returns:
        List of registered before hooks
    """
    return _before_llm_call_hooks.copy()


def get_after_llm_call_hooks() -> list[Callable[[LLMCallHookContext], str | None]]:
    """Get all registered global after_llm_call hooks.

    Returns:
        List of registered after hooks
    """
    return _after_llm_call_hooks.copy()


def clear_all_llm_call_hooks() -> None:
    """Clear all registered global hooks."""
    _before_llm_call_hooks.clear()
    _after_llm_call_hooks.clear()
