"""arun must not execute a sync tool before raising NotImplementedError."""

import asyncio

import pytest

from crewai.tools import tool

calls: list[str] = []


@tool("send_email")
def send_email(value: str) -> str:
    """Record the value that would be sent."""
    calls.append(value)
    return "sent"


@tool("send_email_async")
async def send_email_async(value: str) -> str:
    """Async version."""
    calls.append(value)
    return "sent"


def test_arun_of_sync_tool_does_not_run_it():
    calls.clear()

    with pytest.raises(NotImplementedError):
        asyncio.run(send_email.arun("nobody@example.com"))

    assert calls == []


def test_arun_of_async_tool_still_runs():
    calls.clear()

    result = asyncio.run(send_email_async.arun("someone@example.com"))

    assert result == "sent"
    assert calls == ["someone@example.com"]
