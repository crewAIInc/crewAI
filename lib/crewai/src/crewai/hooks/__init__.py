from __future__ import annotations

from crewai.hooks.decorators import (
    after_llm_call,
    after_tool_call,
    before_llm_call,
    before_tool_call,
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


__all__ = [
    # Context classes
    "LLMCallHookContext",
    "ToolCallHookContext",
    # Decorators
    "after_llm_call",
    "after_tool_call",
    "before_llm_call",
    "before_tool_call",
    "clear_after_llm_call_hooks",
    "clear_after_tool_call_hooks",
    "clear_all_global_hooks",
    "clear_all_llm_call_hooks",
    "clear_all_tool_call_hooks",
    # Clear hooks
    "clear_before_llm_call_hooks",
    "clear_before_tool_call_hooks",
    "get_after_llm_call_hooks",
    "get_after_tool_call_hooks",
    # Get hooks
    "get_before_llm_call_hooks",
    "get_before_tool_call_hooks",
    "register_after_llm_call_hook",
    "register_after_tool_call_hook",
    # LLM Hook registration
    "register_before_llm_call_hook",
    # Tool Hook registration
    "register_before_tool_call_hook",
    "unregister_after_llm_call_hook",
    "unregister_after_tool_call_hook",
    "unregister_before_llm_call_hook",
    "unregister_before_tool_call_hook",
]
