"""Tests for Agent kickoff and kickoff_async guardrail retry handling."""

from __future__ import annotations

from typing import Any

import pytest

pytestmark = pytest.mark.block_network(allowed_hosts=["127.0.0.1", "localhost"])
from pydantic import BaseModel

from crewai import Agent
from crewai.llms.base_llm import BaseLLM


class _CountedLLM(BaseLLM):
    """Returns responses with sequential call counts."""

    def __init__(self, responses: list[str] | None = None) -> None:
        """Initialize CountedLLM with optional canned responses."""
        super().__init__(model="counted_llm")
        object.__setattr__(self, "call_count", 0)
        object.__setattr__(self, "responses", responses or [])

    def call(self, messages: Any, **kwargs: Any) -> str:
        """Return next response or default response string."""
        count = self.call_count
        object.__setattr__(self, "call_count", count + 1)
        if self.responses and count < len(self.responses):
            return self.responses[count]
        return f"Response {count + 1}"

    async def acall(self, messages: Any, **kwargs: Any) -> str:
        """Async call delegating to synchronous call."""
        return self.call(messages, **kwargs)

    def supports_function_calling(self) -> bool:
        """Return whether function calling is supported."""
        return False

    def supports_stop_words(self) -> bool:
        """Return whether stop words are supported."""
        return False

    def get_context_window_size(self) -> int:
        """Return simulated context window size."""
        return 8192


class _CustomPydanticResult(BaseModel):
    """Sample Pydantic result model for guardrail output transformations."""

    summary: str


@pytest.mark.asyncio
async def test_agent_kickoff_async_guardrail_retry_success() -> None:
    """A failing guardrail in kickoff_async retries asynchronously and returns on success."""
    llm = _CountedLLM(responses=["Initial draft", "Refined draft"])
    attempts: list[str] = []

    def fail_first_guardrail(output: Any) -> tuple[bool, str]:
        """Fail on first attempt and pass on subsequent attempts."""
        attempts.append(output.raw)
        if len(attempts) == 1:
            return (False, "Draft lacked required detail, please refine.")
        return (True, output.raw)

    agent = Agent(
        role="Researcher",
        goal="Test guardrails",
        backstory="Tester",
        llm=llm,
        guardrail=fail_first_guardrail,
        guardrail_max_retries=2,
    )

    result = await agent.kickoff_async("Provide report")

    assert result.raw == "Refined draft"
    assert len(attempts) == 2
    assert attempts == ["Initial draft", "Refined draft"]


@pytest.mark.asyncio
async def test_agent_kickoff_async_guardrail_max_retries_exceeded() -> None:
    """When a guardrail fails repeatedly, kickoff_async raises ValueError after max retries."""
    llm = _CountedLLM()

    def always_fail_guardrail(output: Any) -> tuple[bool, str]:
        """Always reject output to trigger retry exhaustion."""
        return (False, "Unconditional guardrail rejection")

    agent = Agent(
        role="Researcher",
        goal="Test guardrails",
        backstory="Tester",
        llm=llm,
        guardrail=always_fail_guardrail,
        guardrail_max_retries=2,
    )

    with pytest.raises(
        ValueError,
        match="Agent's guardrail failed validation after 2 retries. Last error: Unconditional guardrail rejection",
    ):
        await agent.kickoff_async("Provide report")


@pytest.mark.asyncio
async def test_agent_kickoff_async_guardrail_transforms_output() -> None:
    """A passing guardrail in kickoff_async can transform raw and pydantic outputs."""
    llm = _CountedLLM()

    def transform_guardrail(output: Any) -> tuple[bool, _CustomPydanticResult]:
        """Transform output into a structured Pydantic model."""
        return (True, _CustomPydanticResult(summary="Transformed Summary"))

    agent = Agent(
        role="Researcher",
        goal="Test guardrails",
        backstory="Tester",
        llm=llm,
        guardrail=transform_guardrail,
        guardrail_max_retries=1,
    )

    result = await agent.kickoff_async("Provide report")

    assert result.pydantic is not None
    assert isinstance(result.pydantic, _CustomPydanticResult)
    assert result.pydantic.summary == "Transformed Summary"


def test_agent_kickoff_sync_guardrail_retry_parity() -> None:
    """Synchronous kickoff continues to process guardrail retries cleanly."""
    llm = _CountedLLM(responses=["Draft 1", "Draft 2"])
    attempts: list[str] = []

    def fail_first_guardrail(output: Any) -> tuple[bool, str]:
        """Fail on first attempt and pass on second for sync kickoff."""
        attempts.append(output.raw)
        if len(attempts) == 1:
            return (False, "Needs revision.")
        return (True, "Draft 2 Approved")

    agent = Agent(
        role="Researcher",
        goal="Test guardrails",
        backstory="Tester",
        llm=llm,
        guardrail=fail_first_guardrail,
        guardrail_max_retries=2,
    )

    result = agent.kickoff("Provide report")

    assert result.raw == "Draft 2 Approved"
    assert len(attempts) == 2
