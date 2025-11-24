from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, runtime_checkable


if TYPE_CHECKING:
    from crewai.hooks.llm_hooks import LLMCallHookContext
    from crewai.hooks.tool_hooks import ToolCallHookContext


ContextT = TypeVar("ContextT", contravariant=True)
ReturnT = TypeVar("ReturnT", covariant=True)


@runtime_checkable
class Hook(Protocol, Generic[ContextT, ReturnT]):
    """Generic protocol for hook functions.

    This protocol defines the common interface for all hook types in CrewAI.
    Hooks receive a context object and optionally return a modified result.

    Type Parameters:
        ContextT: The context type (LLMCallHookContext or ToolCallHookContext)
        ReturnT: The return type (None, str | None, or bool | None)

    Example:
        >>> # Before LLM call hook: receives LLMCallHookContext, returns None
        >>> hook: Hook[LLMCallHookContext, None] = lambda ctx: print(ctx.iterations)
        >>>
        >>> # After LLM call hook: receives LLMCallHookContext, returns str | None
        >>> hook: Hook[LLMCallHookContext, str | None] = lambda ctx: ctx.response
    """

    def __call__(self, context: ContextT) -> ReturnT:
        """Execute the hook with the given context.

        Args:
            context: Context object with relevant execution state

        Returns:
            Hook-specific return value (None, str | None, or bool | None)
        """
        ...


class BeforeLLMCallHook(Hook["LLMCallHookContext", bool | None], Protocol):
    """Protocol for before_llm_call hooks.

    These hooks are called before an LLM is invoked and can modify the messages
    that will be sent to the LLM or block the execution entirely.
    """

    def __call__(self, context: LLMCallHookContext) -> bool | None:
        """Execute the before LLM call hook.

        Args:
            context: Context object with executor, messages, agent, task, etc.
                Messages can be modified in-place.

        Returns:
            False to block LLM execution, True or None to allow execution
        """
        ...


class AfterLLMCallHook(Hook["LLMCallHookContext", str | None], Protocol):
    """Protocol for after_llm_call hooks.

    These hooks are called after an LLM returns a response and can modify
    the response or the message history.
    """

    def __call__(self, context: LLMCallHookContext) -> str | None:
        """Execute the after LLM call hook.

        Args:
            context: Context object with executor, messages, agent, task, response, etc.
                Messages can be modified in-place. Response is available in context.response.

        Returns:
            Modified response string, or None to keep the original response
        """
        ...


class BeforeToolCallHook(Hook["ToolCallHookContext", bool | None], Protocol):
    """Protocol for before_tool_call hooks.

    These hooks are called before a tool is executed and can modify the tool
    input or block the execution entirely.
    """

    def __call__(self, context: ToolCallHookContext) -> bool | None:
        """Execute the before tool call hook.

        Args:
            context: Context object with tool_name, tool_input, tool, agent, task, etc.
                Tool input can be modified in-place.

        Returns:
            False to block tool execution, True or None to allow execution
        """
        ...


class AfterToolCallHook(Hook["ToolCallHookContext", str | None], Protocol):
    """Protocol for after_tool_call hooks.

    These hooks are called after a tool executes and can modify the result.
    """

    def __call__(self, context: ToolCallHookContext) -> str | None:
        """Execute the after tool call hook.

        Args:
            context: Context object with tool_name, tool_input, tool_result, etc.
                Tool result is available in context.tool_result.

        Returns:
            Modified tool result string, or None to keep the original result
        """
        ...


# - All before hooks: bool | None (False = block execution, True/None = allow)
# - All after hooks: str | None (str = modified result, None = keep original)
BeforeLLMCallHookType = Hook["LLMCallHookContext", bool | None]
AfterLLMCallHookType = Hook["LLMCallHookContext", str | None]
BeforeToolCallHookType = Hook["ToolCallHookContext", bool | None]
AfterToolCallHookType = Hook["ToolCallHookContext", str | None]

# Alternative Callable-based type aliases for compatibility
BeforeLLMCallHookCallable = Callable[["LLMCallHookContext"], bool | None]
AfterLLMCallHookCallable = Callable[["LLMCallHookContext"], str | None]
BeforeToolCallHookCallable = Callable[["ToolCallHookContext"], bool | None]
AfterToolCallHookCallable = Callable[["ToolCallHookContext"], str | None]
