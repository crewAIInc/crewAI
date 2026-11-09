"""ContextVar-based execution collector for tool results.

Provides execution-level isolation for tool results (including result_as_answer)
when tasks are scheduled concurrently using a shared Agent.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any


_active_tool_results: ContextVar[list[dict[str, Any]] | None] = ContextVar(
    "crewai_tool_results", default=None
)


@contextmanager
def tool_results_collector() -> Generator[list[dict[str, Any]], None, None]:
    """Collect the tool results of one execution, isolated from concurrent ones.

    An agent may be shared by tasks running concurrently, so accumulating on
    the agent lets one execution erase or inherit another's records. The
    collector is a ContextVar, which asyncio tasks and threads copy, so each
    execution sees only its own.
    """
    records: list[dict[str, Any]] = []
    token = _active_tool_results.set(records)
    try:
        yield records
    finally:
        _active_tool_results.reset(token)


def active_tool_results() -> list[dict[str, Any]] | None:
    """Tool results for the execution in progress, or None outside a collector."""
    return _active_tool_results.get()
