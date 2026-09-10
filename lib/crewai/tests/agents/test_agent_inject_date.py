"""Tests for the agent ``inject_date`` flag.

These assert against the messages the LLM actually receives, so they fail if the
date stops reaching the wire for either execution entry point: crew/task
execution via ``execute_task`` and standalone execution via ``kickoff``.
"""

from datetime import datetime
from typing import Any
from unittest.mock import patch

from crewai.agent import Agent
from crewai.llms.base_llm import BaseLLM
from crewai.task import Task

MOCK_TARGET = "crewai.utilities.prompts.datetime"
FROZEN_NOW = datetime(2025, 1, 1)


class _RecordingLLM(BaseLLM):
    """Deterministic LLM that captures every message list it is handed."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.calls: list[Any] = []

    def call(self, messages: Any, **kwargs: Any) -> str:
        self.calls.append(messages)
        return "Thought: Done.\nFinal Answer: done"

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return False

    def get_context_window_size(self) -> int:
        return 8_192

    def contents_for(self, role: str) -> list[str]:
        """Every message body sent under ``role`` across all calls."""
        return [
            str(message.get("content", ""))
            for messages in self.calls
            if not isinstance(messages, str)
            for message in messages
            if message.get("role") == role
        ]

    @property
    def everything_sent(self) -> str:
        return "\n".join(
            str(message.get("content", ""))
            for messages in self.calls
            if not isinstance(messages, str)
            for message in messages
        )


def _agent(llm: BaseLLM, **kwargs: Any) -> Agent:
    return Agent(
        role="test_agent",
        goal="test_goal",
        backstory="test_backstory",
        llm=llm,
        max_iter=2,
        **kwargs,
    )


def _task(agent: Agent) -> Task:
    return Task(
        description="What is the date today?",
        expected_output="The date.",
        agent=agent,
    )


def test_inject_date_reaches_system_prompt_on_kickoff() -> None:
    """Standalone ``kickoff`` must put the date in the system message."""
    llm = _RecordingLLM(model="date-test")
    agent = _agent(llm, inject_date=True)

    with patch(MOCK_TARGET) as mock_datetime:
        mock_datetime.now.return_value = FROZEN_NOW
        agent.kickoff("What is the date today?")

    system_prompts = llm.contents_for("system")
    assert system_prompts
    assert all("Current Date: 2025-01-01" in prompt for prompt in system_prompts)


def test_inject_date_reaches_system_prompt_on_task_execution() -> None:
    """Crew/task execution must keep putting the date in front of the model."""
    llm = _RecordingLLM(model="date-test")
    agent = _agent(llm, inject_date=True)

    with patch(MOCK_TARGET) as mock_datetime:
        mock_datetime.now.return_value = FROZEN_NOW
        agent.execute_task(_task(agent))

    system_prompts = llm.contents_for("system")
    assert system_prompts
    assert all("Current Date: 2025-01-01" in prompt for prompt in system_prompts)


def test_inject_date_reaches_prompt_without_system_prompt() -> None:
    """With ``use_system_prompt=False`` the whole prompt is one user message."""
    llm = _RecordingLLM(model="date-test")
    agent = _agent(llm, inject_date=True, use_system_prompt=False)

    with patch(MOCK_TARGET) as mock_datetime:
        mock_datetime.now.return_value = FROZEN_NOW
        agent.kickoff("What is the date today?")

    assert not llm.contents_for("system")
    assert "Current Date: 2025-01-01" in llm.contents_for("user")[0]


def test_inject_date_custom_format() -> None:
    llm = _RecordingLLM(model="date-test")
    agent = _agent(llm, inject_date=True, date_format="%d/%m/%Y")

    with patch(MOCK_TARGET) as mock_datetime:
        mock_datetime.now.return_value = FROZEN_NOW
        agent.kickoff("What is the date today?")

    assert "Current Date: 01/01/2025" in llm.contents_for("system")[0]


def test_without_inject_date_no_date_is_sent() -> None:
    llm = _RecordingLLM(model="date-test")
    agent = _agent(llm)

    agent.kickoff("What is the date today?")

    assert llm.calls
    assert "Current Date:" not in llm.everything_sent


def test_inject_date_invalid_format_is_skipped() -> None:
    """An unusable format should drop the date, not break execution."""
    llm = _RecordingLLM(model="date-test")
    agent = _agent(llm, inject_date=True, date_format="invalid")

    output = agent.kickoff("What is the date today?")

    assert output.raw == "done"
    assert "Current Date:" not in llm.everything_sent


def test_inject_date_does_not_accumulate_across_runs() -> None:
    """Re-running the same task must not stack up repeated date lines."""
    llm = _RecordingLLM(model="date-test")
    agent = _agent(llm, inject_date=True)
    task = _task(agent)

    with patch(MOCK_TARGET) as mock_datetime:
        mock_datetime.now.return_value = FROZEN_NOW
        agent.execute_task(task)
        agent.execute_task(task)

    assert all(
        prompt.count("Current Date:") == 1 for prompt in llm.contents_for("system")
    )
