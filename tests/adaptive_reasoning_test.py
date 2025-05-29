from unittest.mock import MagicMock, patch

import pytest

from crewai import Agent, Crew, Task
from crewai.agents.crew_agent_executor import CrewAgentExecutor


def _create_executor(agent):  # noqa: D401,E501
    """Utility to build a minimal CrewAgentExecutor with the given agent.

    A real LLM call is not required for these unit-tests, so we stub it with
    MagicMock to avoid any network interaction.
    """
    return CrewAgentExecutor(
        llm=MagicMock(),
        task=MagicMock(),
        crew=MagicMock(),
        agent=agent,
        prompt={},
        max_iter=5,
        tools=[],
        tools_names="",
        stop_words=[],
        tools_description="",
        tools_handler=MagicMock(),
    )


def test_agent_adaptive_reasoning_default():
    """Agent.adaptive_reasoning should be False by default."""
    agent = Agent(role="Test", goal="Goal", backstory="Backstory")
    assert agent.adaptive_reasoning is False


@pytest.mark.parametrize("adaptive_decision,expected", [(True, True), (False, False)])
def test_should_trigger_reasoning_with_adaptive_reasoning(adaptive_decision, expected):
    """Verify _should_trigger_reasoning defers to _should_adaptive_reason when
    adaptive_reasoning is enabled and reasoning_interval is None."""
    # Use a lightweight mock instead of a full Agent instance to isolate the logic
    agent = MagicMock()
    agent.reasoning = True
    agent.reasoning_interval = None
    agent.adaptive_reasoning = True

    executor = _create_executor(agent)

    # Ensure the helper returns the desired decision
    with patch.object(executor, "_should_adaptive_reason", return_value=adaptive_decision) as mock_adaptive:
        assert executor._should_trigger_reasoning() is expected
        mock_adaptive.assert_called_once()


@pytest.mark.vcr(filter_headers=["authorization"])
def test_adaptive_reasoning_full_execution():
    """End-to-end test that triggers adaptive reasoning in a real execution flow.

    The task description intentionally contains the word "error" to activate the
    simple error-based heuristic inside `_should_adaptive_reason`, guaranteeing
    that the agent reasons mid-execution without relying on patched internals.
    """
    agent = Agent(
        role="Math Analyst",
        goal="Solve arithmetic problems flawlessly",
        backstory="You excel at basic calculations and always double-check your steps.",
        llm="gpt-4o-mini",
        reasoning=True,
        adaptive_reasoning=True,
        verbose=False,
    )

    task = Task(
        description="There was an unexpected error earlier. Now, please calculate 3 + 5 and return only the number.",
        expected_output="The result of the calculation (a single number).",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task])

    result = crew.kickoff()

    # Validate the answer is correct and numeric
    assert result.raw.strip() == "8"

    # Confirm that an adaptive reasoning message (Updated plan) was injected
    assert any(
        "updated plan" in msg.get("content", "").lower()
        for msg in agent.agent_executor.messages
    )