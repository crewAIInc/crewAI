from __future__ import annotations

from collections.abc import Callable
from functools import wraps
import inspect
from typing import TYPE_CHECKING, Any, TypeVar, overload

from crewai.hooks.dispatch import OnErrorPolicy
from crewai.utilities.string_utils import sanitize_tool_name


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
        on_error: OnErrorPolicy = "ignore",
    ) -> Callable[..., Any]:
        if tools:
            tools = [sanitize_tool_name(t) for t in tools]

        def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
            setattr(f, marker_attribute, True)

            sig = inspect.signature(f)
            params = list(sig.parameters.keys())
            is_method = len(params) >= 2 and params[0] == "self"

            if tools:
                f._filter_tools = tools  # type: ignore[attr-defined]
            if agents:
                f._filter_agents = agents  # type: ignore[attr-defined]
            if on_error != "ignore":
                f._hook_on_error = on_error  # type: ignore[attr-defined]

            register_kwargs: dict[str, Any] = (
                {"on_error": on_error} if on_error != "ignore" else {}
            )

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
                    register_function(filtered_hook, **register_kwargs)
                    f._registered_hook = getattr(  # type: ignore[attr-defined]
                        filtered_hook, "_registered_hook", filtered_hook
                    )

                return f

            if not is_method:
                register_function(f, **register_kwargs)

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
    """Decorator to register a function as a before_llm_call hook."""
    from crewai.hooks.llm_hooks import register_before_llm_call_hook

    return _create_hook_decorator(  # type: ignore[no-any-return]
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
    """Decorator to register a function as an after_llm_call hook."""
    from crewai.hooks.llm_hooks import register_after_llm_call_hook

    return _create_hook_decorator(  # type: ignore[no-any-return]
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
    on_error: OnErrorPolicy = "ignore",
) -> Callable[
    [Callable[[ToolCallHookContext], bool | None]],
    Callable[[ToolCallHookContext], bool | None],
]: ...


def before_tool_call(
    func: Callable[[ToolCallHookContext], bool | None] | None = None,
    *,
    tools: list[str] | None = None,
    agents: list[str] | None = None,
    on_error: OnErrorPolicy = "ignore",
) -> (
    Callable[[ToolCallHookContext], bool | None]
    | Callable[
        [Callable[[ToolCallHookContext], bool | None]],
        Callable[[ToolCallHookContext], bool | None],
    ]
):
    """Decorator to register a function as a before_tool_call hook."""
    from crewai.hooks.tool_hooks import register_before_tool_call_hook

    return _create_hook_decorator(  # type: ignore[no-any-return]
        hook_type="tool",
        register_function=register_before_tool_call_hook,
        marker_attribute="is_before_tool_call_hook",
    )(func=func, tools=tools, agents=agents, on_error=on_error)


@overload
def after_tool_call(
    func: Callable[[ToolCallHookContext], str | None],
) -> Callable[[ToolCallHookContext], str | None]: ...


@overload
def after_tool_call(
    *,
    tools: list[str] | None = None,
    agents: list[str] | None = None,
    on_error: OnErrorPolicy = "ignore",
) -> Callable[
    [Callable[[ToolCallHookContext], str | None]],
    Callable[[ToolCallHookContext], str | None],
]: ...


def after_tool_call(
    func: Callable[[ToolCallHookContext], str | None] | None = None,
    *,
    tools: list[str] | None = None,
    agents: list[str] | None = None,
    on_error: OnErrorPolicy = "ignore",
) -> (
    Callable[[ToolCallHookContext], str | None]
    | Callable[
        [Callable[[ToolCallHookContext], str | None]],
        Callable[[ToolCallHookContext], str | None],
    ]
):
    """Decorator to register a function as an after_tool_call hook."""
    from crewai.hooks.tool_hooks import register_after_tool_call_hook

    return _create_hook_decorator(  # type: ignore[no-any-return]
        hook_type="tool",
        register_function=register_after_tool_call_hook,
        marker_attribute="is_after_tool_call_hook",
    )(func=func, tools=tools, agents=agents, on_error=on_error)
