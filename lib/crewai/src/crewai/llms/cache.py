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

from typing import Any


CACHE_BREAKPOINT_KEY = "cache_breakpoint"


def mark_cache_breakpoint(message: dict[str, Any]) -> dict[str, Any]:
    """Return ``message`` with the cache-breakpoint flag set.

    Returns a new dict so callers can safely pass literal dicts.
    """
    return {**message, CACHE_BREAKPOINT_KEY: True}


def extract_cache_breakpoint_indices(
    messages: str | list[dict[str, Any]] | None,
) -> set[int]:
    """Return the indices of messages flagged as cache breakpoints.

    Reads the flag without mutating the input. The base LLM formatter strips
    the flag during validation, so providers that need to honor breakpoints
    must read them from the raw input first.
    """
    if not messages or isinstance(messages, str):
        return set()
    return {
        i
        for i, m in enumerate(messages)
        if isinstance(m, dict) and m.get(CACHE_BREAKPOINT_KEY)
    }


def strip_cache_breakpoint(message: dict[str, Any]) -> None:
    """Remove the breakpoint flag from a message in place."""
    message.pop(CACHE_BREAKPOINT_KEY, None)
