"""Tests for guardrail retry handling when agent returns non-string results.

When a guardrail fails and the agent retries, the result from
``agent.execute_task`` / ``agent.aexecute_task`` may be a Pydantic
BaseModel (e.g. when ``output_pydantic`` is set). ``TaskOutput.raw``
requires a string, so the retry path must coerce the result.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, Mock

import pytest
from pydantic import BaseModel, Field

from crewai import Agent, Task
from crewai.tasks.task_output import TaskOutput


class MovieReview(BaseModel):
    title: str = Field(description="Movie title")
    rating: int = Field(description="Rating out of 10")


def _make_agent(side_effects: list[object]) -> Mock:
    """Build a minimal mock agent whose execute_task returns values from *side_effects*."""
    agent = Mock()
    agent.role = "reviewer"
    agent.execute_task.side_effect = side_effects
    agent.crew = None
    agent.last_messages = []
    agent.agent_executor = None
    return agent


def _make_async_agent(side_effects: list[object]) -> Mock:
    """Build a minimal mock agent whose aexecute_task returns values from *side_effects*."""
    agent = Mock()
    agent.role = "reviewer"
    agent.aexecute_task = AsyncMock(side_effect=side_effects)
    agent.crew = None
    agent.last_messages = []
    agent.agent_executor = None
    return agent


def _failing_then_passing_guardrail():
    """Return a guardrail that fails once then passes."""
    call_count = {"n": 0}

    def guardrail(output: TaskOutput):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return (False, "Not good enough")
        return (True, output.raw)

    return guardrail


def _create_task_with_guardrail(guardrail):
    """Create a task with a guardrail using a real Agent placeholder."""
    placeholder_agent = Agent(
        role="reviewer",
        goal="review movies",
        backstory="a film critic",
    )
    return Task(
        description="Review a movie",
        expected_output="A movie review",
        guardrail=guardrail,
        agent=placeholder_agent,
    )


class TestGuardrailNonStringRaw:
    """Guardrail retry must not crash when agent returns a BaseModel."""

    def test_sync_guardrail_retry_with_pydantic_result(self) -> None:
        """Sync retry path coerces BaseModel to string for TaskOutput.raw."""
        review = MovieReview(title="Inception", rating=9)

        # First call returns a plain string (guardrail fails on it).
        # Second call (retry) returns a BaseModel.
        agent = _make_agent(["initial output", review])

        task = _create_task_with_guardrail(_failing_then_passing_guardrail())
        result = task.execute_sync(agent=agent)

        assert isinstance(result, TaskOutput)
        # raw must be a string, not a BaseModel
        assert isinstance(result.raw, str)
        assert "Inception" in result.raw

    @pytest.mark.asyncio
    async def test_async_guardrail_retry_with_pydantic_result(self) -> None:
        """Async retry path coerces BaseModel to string for TaskOutput.raw."""
        review = MovieReview(title="Inception", rating=9)

        agent = _make_async_agent(["initial output", review])

        task = _create_task_with_guardrail(_failing_then_passing_guardrail())
        result = await task.aexecute_sync(agent=agent)

        assert isinstance(result, TaskOutput)
        assert isinstance(result.raw, str)
        assert "Inception" in result.raw
