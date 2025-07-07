from unittest.mock import Mock, patch

import pytest

from crewai import Agent, Task
from crewai.llm import LLM
from crewai.tasks.hallucination_guardrail import HallucinationGuardrail
from crewai.tasks.llm_guardrail import LLMGuardrail
from crewai.tasks.task_output import TaskOutput
from crewai.utilities.events import (
    LLMGuardrailCompletedEvent,
    LLMGuardrailStartedEvent,
)
from crewai.utilities.events.crewai_event_bus import crewai_event_bus


def test_task_without_guardrail():
    """Test that tasks work normally without guardrails (backward compatibility)."""
    agent = Mock()
    agent.role = "test_agent"
    agent.execute_task.return_value = "test result"
    agent.crew = None

    task = Task(description="Test task", expected_output="Output")

    result = task.execute_sync(agent=agent)
    assert isinstance(result, TaskOutput)
    assert result.raw == "test result"


def test_task_with_successful_guardrail_func():
    """Test that successful guardrail validation passes transformed result."""

    def guardrail(result: TaskOutput):
        return (True, result.raw.upper())

    agent = Mock()
    agent.role = "test_agent"
    agent.execute_task.return_value = "test result"
    agent.crew = None

    task = Task(description="Test task", expected_output="Output", guardrail=guardrail)

    result = task.execute_sync(agent=agent)
    assert isinstance(result, TaskOutput)
    assert result.raw == "TEST RESULT"


def test_task_with_failing_guardrail():
    """Test that failing guardrail triggers retry with error context."""

    def guardrail(result: TaskOutput):
        return (False, "Invalid format")

    agent = Mock()
    agent.role = "test_agent"
    agent.execute_task.side_effect = ["bad result", "good result"]
    agent.crew = None

    task = Task(
        description="Test task",
        expected_output="Output",
        guardrail=guardrail,
        max_retries=1,
    )

    # First execution fails guardrail, second succeeds
    agent.execute_task.side_effect = ["bad result", "good result"]
    with pytest.raises(Exception) as exc_info:
        task.execute_sync(agent=agent)

    assert "Task failed guardrail validation" in str(exc_info.value)
    assert task.retry_count == 1


def test_task_with_guardrail_retries():
    """Test that guardrail respects max_retries configuration."""

    def guardrail(result: TaskOutput):
        return (False, "Invalid format")

    agent = Mock()
    agent.role = "test_agent"
    agent.execute_task.return_value = "bad result"
    agent.crew = None

    task = Task(
        description="Test task",
        expected_output="Output",
        guardrail=guardrail,
        max_retries=2,
    )

    with pytest.raises(Exception) as exc_info:
        task.execute_sync(agent=agent)

    assert task.retry_count == 2
    assert "Task failed guardrail validation after 2 retries" in str(exc_info.value)
    assert "Invalid format" in str(exc_info.value)


def test_guardrail_error_in_context():
    """Test that guardrail error is passed in context for retry."""

    def guardrail(result: TaskOutput):
        return (False, "Expected JSON, got string")

    agent = Mock()
    agent.role = "test_agent"
    agent.crew = None

    task = Task(
        description="Test task",
        expected_output="Output",
        guardrail=guardrail,
        max_retries=1,
    )

    # Mock execute_task to succeed on second attempt
    first_call = True

    def execute_task(task, context, tools):
        nonlocal first_call
        if first_call:
            first_call = False
            return "invalid"
        return '{"valid": "json"}'

    agent.execute_task.side_effect = execute_task

    with pytest.raises(Exception) as exc_info:
        task.execute_sync(agent=agent)

    assert "Task failed guardrail validation" in str(exc_info.value)
    assert "Expected JSON, got string" in str(exc_info.value)


@pytest.fixture
def sample_agent():
    return Agent(role="Test Agent", goal="Test Goal", backstory="Test Backstory")


@pytest.fixture
def task_output():
    return TaskOutput(
        raw="""
        Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever
        """,
        description="Test task",
        expected_output="Output",
        agent="Test Agent",
    )


@pytest.mark.vcr(filter_headers=["authorization"])
def test_task_guardrail_process_output(task_output):
    guardrail = LLMGuardrail(
        description="Ensure the result has less than 10 words", llm=LLM(model="gpt-4o")
    )

    result = guardrail(task_output)
    assert result[0] is False

    assert "exceeding the guardrail limit of fewer than" in result[1].lower()

    guardrail = LLMGuardrail(
        description="Ensure the result has less than 500 words", llm=LLM(model="gpt-4o")
    )

    result = guardrail(task_output)
    assert result[0] is True
    assert result[1] == task_output.raw


@pytest.mark.vcr(filter_headers=["authorization"])
def test_guardrail_emits_events(sample_agent):
    started_guardrail = []
    completed_guardrail = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(LLMGuardrailStartedEvent)
        def handle_guardrail_started(source, event):
            started_guardrail.append(
                {"guardrail": event.guardrail, "retry_count": event.retry_count}
            )

        @crewai_event_bus.on(LLMGuardrailCompletedEvent)
        def handle_guardrail_completed(source, event):
            completed_guardrail.append(
                {
                    "success": event.success,
                    "result": event.result,
                    "error": event.error,
                    "retry_count": event.retry_count,
                }
            )

        task = Task(
            description="Gather information about available books on the First World War",
            agent=sample_agent,
            expected_output="A list of available books on the First World War",
            guardrail="Ensure the authors are from Italy",
        )

        result = task.execute_sync(agent=sample_agent)

        def custom_guardrail(result: TaskOutput):
            return (True, "good result from callable function")

        task = Task(
            description="Test task",
            expected_output="Output",
            guardrail=custom_guardrail,
        )

        task.execute_sync(agent=sample_agent)

        expected_started_events = [
            {"guardrail": "Ensure the authors are from Italy", "retry_count": 0},
            {"guardrail": "Ensure the authors are from Italy", "retry_count": 1},
            {
                "guardrail": """def custom_guardrail(result: TaskOutput):
            return (True, "good result from callable function")""",
                "retry_count": 0,
            },
        ]

        expected_completed_events = [
            {
                "success": False,
                "result": None,
                "error": "The task result does not comply with the guardrail because none of "
                "the listed authors are from Italy. All authors mentioned are from "
                "different countries, including Germany, the UK, the USA, and others, "
                "which violates the requirement that authors must be Italian.",
                "retry_count": 0,
            },
            {"success": True, "result": result.raw, "error": None, "retry_count": 1},
            {
                "success": True,
                "result": "good result from callable function",
                "error": None,
                "retry_count": 0,
            },
        ]
        assert started_guardrail == expected_started_events
        assert completed_guardrail == expected_completed_events


@pytest.mark.vcr(filter_headers=["authorization"])
def test_guardrail_when_an_error_occurs(sample_agent, task_output):
    with (
        patch(
            "crewai.Agent.kickoff",
            side_effect=Exception("Unexpected error"),
        ),
        pytest.raises(
            Exception,
            match="Error while validating the task output: Unexpected error",
        ),
    ):
        task = Task(
            description="Gather information about available books on the First World War",
            agent=sample_agent,
            expected_output="A list of available books on the First World War",
            guardrail="Ensure the authors are from Italy",
            max_retries=0,
        )
        task.execute_sync(agent=sample_agent)


def test_hallucination_guardrail_integration():
    """Test that HallucinationGuardrail integrates properly with the task system."""
    agent = Mock()
    agent.role = "test_agent"
    agent.execute_task.return_value = "test result"
    agent.crew = None

    mock_llm = Mock(spec=LLM)
    guardrail = HallucinationGuardrail(
        context="Test reference context for validation", llm=mock_llm, threshold=8.0
    )

    task = Task(
        description="Test task with hallucination guardrail",
        expected_output="Valid output",
        guardrail=guardrail,
    )

    result = task.execute_sync(agent=agent)
    assert isinstance(result, TaskOutput)
    assert result.raw == "test result"


def test_hallucination_guardrail_description_in_events():
    """Test that HallucinationGuardrail description appears correctly in events."""
    mock_llm = Mock(spec=LLM)
    guardrail = HallucinationGuardrail(context="Test context", llm=mock_llm)

    assert guardrail.description == "HallucinationGuardrail (no-op)"

    event = LLMGuardrailStartedEvent(guardrail=guardrail, retry_count=0)
    assert event.guardrail == "HallucinationGuardrail (no-op)"
