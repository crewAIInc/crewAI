"""Provider-agnostic prompt-cache breakpoint marker.

Application code (prompt builders, agent executors) marks messages where a
stable prefix ends. Provider adapters then translate the marker into the
cache directive their API expects, or strip it for providers that cache
implicitly (OpenAI, Gemini) or do not cache at all.

Usage:

    from crewai.llms.cache import mark_cache_breakpoint

    messages = [
        mark_cache_breakpoint({"role": "system", "content": stable_system}),
        mark_cache_breakpoint({"role": "user", "content": stable_user_prefix}),
        {"role": "user", "content": volatile_query},
    ]
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from crewai.llms.base_llm import BaseLLM


CACHE_BREAKPOINT_KEY = "cache_breakpoint"

_ANTHROPIC_MODEL_PREFIXES: tuple[str, ...] = ("anthropic/", "claude-", "claude/")


def mark_cache_breakpoint(message: dict[str, Any]) -> dict[str, Any]:
    """Return ``message`` with the cache-breakpoint flag set.

    Returns a new dict so callers can safely pass literal dicts.
    """
    return {**message, CACHE_BREAKPOINT_KEY: True}


def strip_cache_breakpoint(message: dict[str, Any]) -> None:
    """Remove the breakpoint flag from a message in place."""
    message.pop(CACHE_BREAKPOINT_KEY, None)


def supports_cache_breakpoint(llm: BaseLLM | str | None) -> bool:
    """Check if the given LLM supports Anthropic-style prompt caching.

    Args:
        llm: The LLM instance, model string, or None.

    Returns:
        True if the LLM is Anthropic-compatible and supports cache_breakpoint.
    """
    if llm is None:
        return False

    if hasattr(llm, "model"):
        model = llm.model
    elif isinstance(llm, str):
        model = llm
    else:
        return False

    model_lower = model.lower()
    return any(prefix in model_lower for prefix in _ANTHROPIC_MODEL_PREFIXES)
