from __future__ import annotations

from collections.abc import Callable
from functools import wraps
import inspect
from typing import TYPE_CHECKING, Any, TypeVar, overload


if TYPE_CHECKING:
    from crewai.hooks.llm_hooks import LLMCallHookContext
    from crewai.hooks.tool_hooks import ToolCallHookContext

F = TypeVar("F", bound=Callable[..., Any])


def _create_hook_decorator(
    hook_type: str,
    register_function: Callable[..., Any],
    marker_attribute: str,
) -> Callable[..., Any]:
    """Create a hook decorator with filtering support.

    This factory function eliminates code duplication across the four hook decorators.

    Args:
        hook_type: Type of hook ("llm" or "tool")
        register_function: Function to call for registration (e.g., register_before_llm_call_hook)
        marker_attribute: Attribute name to mark functions (e.g., "is_before_llm_call_hook")

    Returns:
        A decorator function that supports filters and auto-registration
    """

    def decorator_factory(
        func: Callable[..., Any] | None = None,
        *,
        tools: list[str] | None = None,
        agents: list[str] | None = None,
    ) -> Callable[..., Any]:
        def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
            setattr(f, marker_attribute, True)

            sig = inspect.signature(f)
            params = list(sig.parameters.keys())
            is_method = len(params) >= 2 and params[0] == "self"

            if tools:
                f._filter_tools = tools  # type: ignore[attr-defined]
            if agents:
                f._filter_agents = agents  # type: ignore[attr-defined]

            if tools or agents:

                @wraps(f)
                def filtered_hook(context: Any) -> Any:
                    if tools and hasattr(context, "tool_name"):
                        if context.tool_name not in tools:
                            return None

                    if agents and hasattr(context, "agent"):
                        if context.agent and context.agent.role not in agents:
                            return None

                    return f(context)

                if not is_method:
                    register_function(filtered_hook)

                return f

            if not is_method:
                register_function(f)

            return f

        if func is None:
            return decorator
        return decorator(func)

    return decorator_factory


@overload
def before_llm_call(
    func: Callable[[LLMCallHookContext], None],
) -> Callable[[LLMCallHookContext], None]: ...


@overload
def before_llm_call(
    *,
    agents: list[str] | None = None,
) -> Callable[
    [Callable[[LLMCallHookContext], None]], Callable[[LLMCallHookContext], None]
]: ...


def before_llm_call(
    func: Callable[[LLMCallHookContext], None] | None = None,
    *,
    agents: list[str] | None = None,
) -> (
    Callable[[LLMCallHookContext], None]
    | Callable[
        [Callable[[LLMCallHookContext], None]], Callable[[LLMCallHookContext], None]
    ]
):
    """Decorator to register a function as a before_llm_call hook.

    Example:
        Simple usage::

            @before_llm_call
            def log_calls(context):
                print(f"LLM call by {context.agent.role}")

        With agent filter::

            @before_llm_call(agents=["Researcher", "Analyst"])
            def log_specific_agents(context):
                print(f"Filtered LLM call: {context.agent.role}")
    """
    from crewai.hooks.llm_hooks import register_before_llm_call_hook

    return _create_hook_decorator(  # type: ignore[return-value]
        hook_type="llm",
        register_function=register_before_llm_call_hook,
        marker_attribute="is_before_llm_call_hook",
    )(func=func, agents=agents)


@overload
def after_llm_call(
    func: Callable[[LLMCallHookContext], str | None],
) -> Callable[[LLMCallHookContext], str | None]: ...


@overload
def after_llm_call(
    *,
    agents: list[str] | None = None,
) -> Callable[
    [Callable[[LLMCallHookContext], str | None]],
    Callable[[LLMCallHookContext], str | None],
]: ...


def after_llm_call(
    func: Callable[[LLMCallHookContext], str | None] | None = None,
    *,
    agents: list[str] | None = None,
) -> (
    Callable[[LLMCallHookContext], str | None]
    | Callable[
        [Callable[[LLMCallHookContext], str | None]],
        Callable[[LLMCallHookContext], str | None],
    ]
):
    """Decorator to register a function as an after_llm_call hook.

    Example:
        Simple usage::

            @after_llm_call
            def sanitize(context):
                if "SECRET" in context.response:
                    return context.response.replace("SECRET", "[REDACTED]")
                return None

        With agent filter::

            @after_llm_call(agents=["Researcher"])
            def log_researcher_responses(context):
                print(f"Response length: {len(context.response)}")
                return None
    """
    from crewai.hooks.llm_hooks import register_after_llm_call_hook

    return _create_hook_decorator(  # type: ignore[return-value]
        hook_type="llm",
        register_function=register_after_llm_call_hook,
        marker_attribute="is_after_llm_call_hook",
    )(func=func, agents=agents)


@overload
def before_tool_call(
    func: Callable[[ToolCallHookContext], bool | None],
) -> Callable[[ToolCallHookContext], bool | None]: ...


@overload
def before_tool_call(
    *,
    tools: list[str] | None = None,
    agents: list[str] | None = None,
) -> Callable[
    [Callable[[ToolCallHookContext], bool | None]],
    Callable[[ToolCallHookContext], bool | None],
]: ...


def before_tool_call(
    func: Callable[[ToolCallHookContext], bool | None] | None = None,
    *,
    tools: list[str] | None = None,
    agents: list[str] | None = None,
) -> (
    Callable[[ToolCallHookContext], bool | None]
    | Callable[
        [Callable[[ToolCallHookContext], bool | None]],
        Callable[[ToolCallHookContext], bool | None],
    ]
):
    """Decorator to register a function as a before_tool_call hook.

    Example:
        Simple usage::

            @before_tool_call
            def log_all_tools(context):
                print(f"Tool: {context.tool_name}")
                return None

        With tool filter::

            @before_tool_call(tools=["delete_file", "execute_code"])
            def approve_dangerous(context):
                response = context.request_human_input(prompt="Approve?")
                return None if response == "yes" else False

        With combined filters::

            @before_tool_call(tools=["write_file"], agents=["Developer"])
            def approve_dev_writes(context):
                return None  # Only for Developer writing files
    """
    from crewai.hooks.tool_hooks import register_before_tool_call_hook

    return _create_hook_decorator(  # type: ignore[return-value]
        hook_type="tool",
        register_function=register_before_tool_call_hook,
        marker_attribute="is_before_tool_call_hook",
    )(func=func, tools=tools, agents=agents)


@overload
def after_tool_call(
    func: Callable[[ToolCallHookContext], str | None],
) -> Callable[[ToolCallHookContext], str | None]: ...


@overload
def after_tool_call(
    *,
    tools: list[str] | None = None,
    agents: list[str] | None = None,
) -> Callable[
    [Callable[[ToolCallHookContext], str | None]],
    Callable[[ToolCallHookContext], str | None],
]: ...


def after_tool_call(
    func: Callable[[ToolCallHookContext], str | None] | None = None,
    *,
    tools: list[str] | None = None,
    agents: list[str] | None = None,
) -> (
    Callable[[ToolCallHookContext], str | None]
    | Callable[
        [Callable[[ToolCallHookContext], str | None]],
        Callable[[ToolCallHookContext], str | None],
    ]
):
    """Decorator to register a function as an after_tool_call hook.

    Example:
        Simple usage::

            @after_tool_call
            def log_results(context):
                print(f"Result: {len(context.tool_result)} chars")
                return None

        With tool filter::

            @after_tool_call(tools=["web_search", "ExaSearchTool"])
            def sanitize_search_results(context):
                if "SECRET" in context.tool_result:
                    return context.tool_result.replace("SECRET", "[REDACTED]")
                return None
    """
    from crewai.hooks.tool_hooks import register_after_tool_call_hook

    return _create_hook_decorator(  # type: ignore[return-value]
        hook_type="tool",
        register_function=register_after_tool_call_hook,
        marker_attribute="is_after_tool_call_hook",
    )(func=func, tools=tools, agents=agents)
