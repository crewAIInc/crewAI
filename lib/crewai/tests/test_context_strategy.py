"""Tests for context_strategy='summarized' feature."""

from unittest.mock import MagicMock, patch

import pytest

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.process import Process
from crewai.task import Task
from crewai.tasks.task_output import TaskOutput
from crewai.utilities.formatter import (
    aggregate_summarized_outputs_from_task_outputs,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent(llm_response: str = "summary text") -> Agent:
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
        allow_delegation=False,
    )
    agent.llm = MagicMock()
    agent.llm.call.return_value = llm_response
    return agent


def _make_task_output(raw: str, context_summary: str | None = None) -> TaskOutput:
    return TaskOutput(
        description="test task",
        raw=raw,
        agent="Test Agent",
        context_summary=context_summary,
    )


# ---------------------------------------------------------------------------
# formatter tests
# ---------------------------------------------------------------------------


def test_aggregate_summarized_uses_context_summary_when_present():
    outputs = [
        _make_task_output("long raw 1", context_summary="short summary 1"),
        _make_task_output("long raw 2", context_summary="short summary 2"),
    ]
    result = aggregate_summarized_outputs_from_task_outputs(outputs)
    assert "short summary 1" in result
    assert "short summary 2" in result
    assert "long raw" not in result


def test_aggregate_summarized_falls_back_to_raw_when_no_summary():
    outputs = [
        _make_task_output("raw output A", context_summary=None),
        _make_task_output("raw output B", context_summary="summary B"),
    ]
    result = aggregate_summarized_outputs_from_task_outputs(outputs)
    assert "raw output A" in result
    assert "summary B" in result


def test_aggregate_summarized_empty_list():
    assert aggregate_summarized_outputs_from_task_outputs([]) == ""


# ---------------------------------------------------------------------------
# TaskOutput tests
# ---------------------------------------------------------------------------


def test_task_output_context_summary_defaults_to_none():
    output = _make_task_output("some raw output")
    assert output.context_summary is None


def test_task_output_context_summary_can_be_set():
    output = _make_task_output("some raw output", context_summary="condensed")
    assert output.context_summary == "condensed"


# ---------------------------------------------------------------------------
# Crew.context_strategy field tests
# ---------------------------------------------------------------------------


def test_crew_context_strategy_defaults_to_full():
    agent = _make_agent()
    task = Task(description="t", expected_output="o", agent=agent)
    crew = Crew(agents=[agent], tasks=[task])
    assert crew.context_strategy == "full"


def test_crew_context_strategy_accepts_summarized():
    agent = _make_agent()
    task = Task(description="t", expected_output="o", agent=agent)
    crew = Crew(agents=[agent], tasks=[task], context_strategy="summarized")
    assert crew.context_strategy == "summarized"


def test_task_context_strategy_defaults_to_none():
    agent = _make_agent()
    task = Task(description="t", expected_output="o", agent=agent)
    assert task.context_strategy is None


# ---------------------------------------------------------------------------
# Crew._generate_context_summary tests
# ---------------------------------------------------------------------------


def _make_minimal_crew(context_strategy: str = "full") -> Crew:
    agent = _make_agent()
    task = Task(description="t", expected_output="o", agent=agent)
    return Crew(agents=[agent], tasks=[task], context_strategy=context_strategy)


def test_generate_context_summary_sets_field():
    agent = _make_agent(llm_response="concise summary")
    task = Task(description="t", expected_output="o", agent=agent)
    task_mock = MagicMock(spec=task)
    task_mock.agent = agent

    output = _make_task_output("very long raw output")
    crew = _make_minimal_crew()
    crew._generate_context_summary(task_mock, output)

    assert output.context_summary == "concise summary"
    agent.llm.call.assert_called_once()


def test_generate_context_summary_ignores_empty_llm_response():
    agent = _make_agent(llm_response="   ")
    task_mock = MagicMock()
    task_mock.agent = agent

    output = _make_task_output("raw output")
    crew = _make_minimal_crew()
    crew._generate_context_summary(task_mock, output)

    assert output.context_summary is None


def test_generate_context_summary_swallows_llm_errors():
    agent = _make_agent()
    agent.llm.call.side_effect = RuntimeError("LLM unavailable")
    task_mock = MagicMock()
    task_mock.agent = agent

    output = _make_task_output("raw output")
    crew = _make_minimal_crew()

    # Should not raise
    crew._generate_context_summary(task_mock, output)
    assert output.context_summary is None


def test_generate_context_summary_skips_when_no_agent():
    task_mock = MagicMock()
    task_mock.agent = None

    output = _make_task_output("raw output")
    crew = _make_minimal_crew()
    crew._generate_context_summary(task_mock, output)

    assert output.context_summary is None


# ---------------------------------------------------------------------------
# Crew._process_task_result — summary generation gated by strategy
# ---------------------------------------------------------------------------


def test_process_task_result_calls_summary_when_strategy_summarized():
    agent = _make_agent()
    task = Task(description="t", expected_output="o", agent=agent)
    task.context_strategy = None  # inherit from crew

    output = _make_task_output("raw")
    crew = Crew(agents=[agent], tasks=[task], context_strategy="summarized")

    with patch.object(crew, "_generate_context_summary") as mock_summary:
        crew._process_task_result(task, output)
        mock_summary.assert_called_once_with(task, output)


def test_process_task_result_skips_summary_when_strategy_full():
    agent = _make_agent()
    task = Task(description="t", expected_output="o", agent=agent)

    output = _make_task_output("raw")
    crew = Crew(agents=[agent], tasks=[task], context_strategy="full")

    with patch.object(crew, "_generate_context_summary") as mock_summary:
        crew._process_task_result(task, output)
        mock_summary.assert_not_called()


def test_process_task_result_task_level_override_wins():
    """Task-level context_strategy='summarized' triggers summary even if crew is 'full'."""
    agent = _make_agent()
    task = Task(description="t", expected_output="o", agent=agent)
    task.context_strategy = "summarized"

    output = _make_task_output("raw")
    crew = Crew(agents=[agent], tasks=[task], context_strategy="full")

    with patch.object(crew, "_generate_context_summary") as mock_summary:
        crew._process_task_result(task, output)
        mock_summary.assert_called_once_with(task, output)


# ---------------------------------------------------------------------------
# Crew._get_context — routes to correct aggregator
# ---------------------------------------------------------------------------


def test_get_context_full_strategy_uses_raw():
    agent = _make_agent()
    task = Task(description="t", expected_output="o", agent=agent)

    outputs = [
        _make_task_output("raw A", context_summary="summary A"),
        _make_task_output("raw B", context_summary="summary B"),
    ]
    crew = Crew(agents=[agent], tasks=[task], context_strategy="full")
    result = crew._get_context(task, outputs)

    assert "raw A" in result
    assert "raw B" in result


def test_get_context_summarized_strategy_uses_summaries():
    agent = _make_agent()
    task = Task(description="t", expected_output="o", agent=agent)

    outputs = [
        _make_task_output("long raw A", context_summary="summary A"),
        _make_task_output("long raw B", context_summary="summary B"),
    ]
    crew = Crew(agents=[agent], tasks=[task], context_strategy="summarized")
    result = crew._get_context(task, outputs)

    assert "summary A" in result
    assert "summary B" in result
    assert "long raw" not in result


def test_get_context_returns_empty_when_no_context():
    agent = _make_agent()
    task = Task(description="t", expected_output="o", agent=agent, context=None)

    crew = Crew(agents=[agent], tasks=[task])
    result = crew._get_context(task, [])
    assert result == ""
