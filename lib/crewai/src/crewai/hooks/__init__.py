from __future__ import annotations

from crewai.hooks.decorators import (
    after_llm_call,
    after_tool_call,
    before_llm_call,
    before_tool_call,
)

# LLM Hooks
from crewai.hooks.llm_hooks import (
    LLMCallHookContext,
    get_after_llm_call_hooks,
    get_before_llm_call_hooks,
    register_after_llm_call_hook,
    register_before_llm_call_hook,
)

# Tool Hooks
from crewai.hooks.tool_hooks import (
    ToolCallHookContext,
    get_after_tool_call_hooks,
    get_before_tool_call_hooks,
    register_after_tool_call_hook,
    register_before_tool_call_hook,
)


__all__ = [
    "LLMCallHookContext",
    "ToolCallHookContext",
    "after_llm_call",
    "after_tool_call",
    "before_llm_call",
    "before_tool_call",
    "get_after_llm_call_hooks",
    "get_after_tool_call_hooks",
    "get_before_llm_call_hooks",
    "get_before_tool_call_hooks",
    "register_after_llm_call_hook",
    "register_after_tool_call_hook",
    "register_before_llm_call_hook",
    "register_before_tool_call_hook",
]
