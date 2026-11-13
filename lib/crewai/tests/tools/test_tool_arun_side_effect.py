"""Regression tests for arun() on a sync decorated tool."""

from __future__ import annotations

import pytest

from crewai.tools import tool


@pytest.mark.asyncio
async def test_sync_tool_is_not_executed_before_arun_fails() -> None:
    """A sync tool must fail before running, not after its side effect."""
    calls: list[str] = []

    @tool("send_email")
    def send_email(value: str) -> str:
        """Record the value that would be sent."""
        calls.append(value)
        return "sent"

    with pytest.raises(NotImplementedError):
        await send_email.arun(value="hello")

    assert calls == []
    assert send_email.current_usage_count == 0
