from __future__ import annotations

from typing import TYPE_CHECKING, Any

from crewai.hooks.decorators import (
    after_llm_call,
    after_tool_call,
    before_llm_call,
    before_tool_call,
)
from crewai.hooks.dispatch import (
    HookAborted,
    InterceptionPoint,
    clear as clear_hooks,
    clear_all as clear_all_hooks,
    dispatch,
    get_hooks,
    on,
    register as register_hook,
    unregister as unregister_hook,
)
from crewai.hooks.llm_hooks import (
    LLMCallHookContext,
    clear_after_llm_call_hooks,
    clear_all_llm_call_hooks,
    clear_before_llm_call_hooks,
    get_after_llm_call_hooks,
    get_before_llm_call_hooks,
    register_after_llm_call_hook,
    register_before_llm_call_hook,
    unregister_after_llm_call_hook,
    unregister_before_llm_call_hook,
)
from crewai.hooks.tool_hooks import (
    ToolCallHookContext,
    clear_after_tool_call_hooks,
    clear_all_tool_call_hooks,
    clear_before_tool_call_hooks,
    get_after_tool_call_hooks,
    get_before_tool_call_hooks,
    register_after_tool_call_hook,
    register_before_tool_call_hook,
    unregister_after_tool_call_hook,
    unregister_before_tool_call_hook,
)


def clear_all_global_hooks() -> dict[str, tuple[int, int]]:
    """Clear all global hooks across all hook types (LLM and Tool).

    This is a convenience function that clears all registered hooks in one call.
    Useful for testing, resetting state, or cleaning up between different
    execution contexts.

    Returns:
        Dictionary with counts of cleared hooks:
        {
            "llm_hooks": (before_count, after_count),
            "tool_hooks": (before_count, after_count),
            "total": (total_before_count, total_after_count)
        }

    Example:
        >>> # Register various hooks
        >>> register_before_llm_call_hook(llm_hook1)
        >>> register_after_llm_call_hook(llm_hook2)
        >>> register_before_tool_call_hook(tool_hook1)
        >>> register_after_tool_call_hook(tool_hook2)
        >>>
        >>> # Clear all hooks at once
        >>> result = clear_all_global_hooks()
        >>> print(result)
        {
            'llm_hooks': (1, 1),
            'tool_hooks': (1, 1),
            'total': (2, 2)
        }
    """
    llm_counts = clear_all_llm_call_hooks()
    tool_counts = clear_all_tool_call_hooks()

    return {
        "llm_hooks": llm_counts,
        "tool_hooks": tool_counts,
        "total": (llm_counts[0] + tool_counts[0], llm_counts[1] + tool_counts[1]),
    }


if TYPE_CHECKING:
    from crewai.hooks.agent_hooks_engine import (
        AgentHooksEngine as AgentHooksEngine,
        active_engine as active_engine,
        disable_agent_hooks as disable_agent_hooks,
        use_agent_hooks as use_agent_hooks,
    )


_LAZY_ENGINE_EXPORTS = frozenset(
    {
        "AgentHooksEngine",
        "HAS_AGENT_HOOKS",
        "active_engine",
        "disable_agent_hooks",
        "use_agent_hooks",
    }
)


def __getattr__(name: str) -> Any:
    """Lazily expose the optional agent-hooks control engine front-end.

    Keeps ``import crewai.hooks`` free of the optional agent-hooks dependency;
    the engine module (and its probe import) loads only on first access.
    """
    if name in _LAZY_ENGINE_EXPORTS:
        from crewai.hooks import agent_hooks_engine

        return getattr(agent_hooks_engine, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "HAS_AGENT_HOOKS",
    "AgentHooksEngine",
    "HookAborted",
    "InterceptionPoint",
    "LLMCallHookContext",
    "ToolCallHookContext",
    "active_engine",
    "after_llm_call",
    "after_tool_call",
    "before_llm_call",
    "before_tool_call",
    "clear_after_llm_call_hooks",
    "clear_after_tool_call_hooks",
    "clear_all_global_hooks",
    "clear_all_hooks",
    "clear_all_llm_call_hooks",
    "clear_all_tool_call_hooks",
    "clear_before_llm_call_hooks",
    "clear_before_tool_call_hooks",
    "clear_hooks",
    "disable_agent_hooks",
    "dispatch",
    "get_after_llm_call_hooks",
    "get_after_tool_call_hooks",
    "get_before_llm_call_hooks",
    "get_before_tool_call_hooks",
    "get_hooks",
    "on",
    "register_after_llm_call_hook",
    "register_after_tool_call_hook",
    "register_before_llm_call_hook",
    "register_before_tool_call_hook",
    "register_hook",
    "unregister_after_llm_call_hook",
    "unregister_after_tool_call_hook",
    "unregister_before_llm_call_hook",
    "unregister_before_tool_call_hook",
    "unregister_hook",
    "use_agent_hooks",
]
