"""Guardrail retries inside kickoff_async must stay on the async path (#7252)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai import Agent
from crewai.lite_agent_output import LiteAgentOutput


def _agent_with_fail_once_guardrail():
    calls = {"n": 0}

    def fail_once(output):
        calls["n"] += 1
        if calls["n"] == 1:
            return (False, "not good enough, retry")
        return (True, output)

    agent = Agent(
        role="Test Agent",
        goal="Answer",
        backstory="Test backstory.",
        guardrail=fail_once,
        guardrail_max_retries=2,
    )
    return agent, calls


def _canned_output(raw="final answer"):
    return LiteAgentOutput(raw=raw, agent_role="Test Agent")


@pytest.mark.asyncio
async def test_kickoff_async_guardrail_retry_uses_async_execution():
    """A guardrail failure on kickoff_async must retry via invoke_async.

    Before the fix the retry went through the sync executor.invoke(), which
    under a running loop hands back an unawaited coroutine instead of a result
    dict and crashes on .get("output"). The sync execute path must not run at
    all here.
    """
    agent, calls = _agent_with_fail_once_guardrail()

    with (
        patch.object(
            Agent, "_prepare_kickoff", return_value=(MagicMock(), {}, {}, [])
        ),
        patch.object(Agent, "_current_usage_summary", return_value=MagicMock()),
        patch.object(
            Agent,
            "_execute_and_build_output_async",
            new=AsyncMock(side_effect=[_canned_output("first"), _canned_output("final answer")]),
        ),
        patch.object(
            Agent,
            "_execute_and_build_output",
            side_effect=AssertionError("sync execution path taken in async context"),
        ),
    ):
        result = await agent.kickoff_async("answer me")

    assert calls["n"] == 2
    assert result.raw == "final answer"


@pytest.mark.asyncio
async def test_kickoff_async_guardrail_retry_exhaustion_still_raises():
    """Retry limits are honored on the async path too."""
    agent, _ = _agent_with_fail_once_guardrail()
    agent.guardrail_max_retries = 0

    with (
        patch.object(
            Agent, "_prepare_kickoff", return_value=(MagicMock(), {}, {}, [])
        ),
        patch.object(Agent, "_current_usage_summary", return_value=MagicMock()),
        patch.object(
            Agent,
            "_execute_and_build_output_async",
            new=AsyncMock(return_value=_canned_output("first")),
        ),
    ):
        with pytest.raises(ValueError, match="guardrail failed validation"):
            await agent.kickoff_async("answer me")
