from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol


if TYPE_CHECKING:
    from crewai.hooks.llm_hooks import LLMCallHookContext
    from crewai.hooks.tool_hooks import ToolCallHookContext


class BeforeLLMCallHook(Protocol):
    """Protocol for before_llm_call hooks.

    These hooks are called before an LLM is invoked and can modify the messages
    that will be sent to the LLM.
    """

    def __call__(self, context: LLMCallHookContext) -> None:
        """Execute the before LLM call hook.

        Args:
            context: Context object with executor, messages, agent, task, etc.
                Messages can be modified in-place.
        """
        ...


class AfterLLMCallHook(Protocol):
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


class BeforeToolCallHook(Protocol):
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


class AfterToolCallHook(Protocol):
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


# Type aliases for hook functions
BeforeLLMCallHookType = Callable[[LLMCallHookContext], None]
AfterLLMCallHookType = Callable[[LLMCallHookContext], str | None]
BeforeToolCallHookType = Callable[[ToolCallHookContext], bool | None]
AfterToolCallHookType = Callable[[ToolCallHookContext], str | None]
