from unittest.mock import ANY, Mock, patch

import pytest

from crewai import Agent, Task
from crewai.llm import LLM
from crewai.tasks.task_guardrail import TaskGuardrail
from crewai.tasks.task_output import TaskOutput
from crewai.utilities.events import (
    TaskGuardrailCompletedEvent,
    TaskGuardrailStartedEvent,
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


@pytest.mark.vcr(filter_headers=["authorization"])
def test_guardrail_using_llm(sample_agent):
    task = Task(
        description="Test task",
        expected_output="Output",
        guardrail="Ensure the output is equal to 'good result'",
    )

    with patch(
        "crewai.tasks.task_guardrail.TaskGuardrail.__call__",
        side_effect=[(False, "bad result"), (True, "good result")],
    ) as mock_guardrail:
        task.execute_sync(agent=sample_agent)

    assert mock_guardrail.call_count == 2

    task.guardrail = TaskGuardrail(
        description="Ensure the output is equal to 'good result'",
        llm=LLM(model="gpt-4o-mini"),
    )

    with patch(
        "crewai.tasks.task_guardrail.TaskGuardrail.__call__",
        side_effect=[(False, "bad result"), (True, "good result")],
    ) as mock_guardrail:
        task.execute_sync(agent=sample_agent)

    assert mock_guardrail.call_count == 2


@pytest.fixture
def task_output():
    return TaskOutput(
        raw="Test output",
        description="Test task",
        expected_output="Output",
        agent="Test Agent",
    )


def test_task_guardrail_initialization_no_llm(task_output):
    """Test TaskGuardrail initialization fails without LLM"""
    with pytest.raises(ValueError, match="Provide a valid LLM to the TaskGuardrail"):
        TaskGuardrail(description="Test")(task_output)


@pytest.fixture
def mock_llm():
    llm = Mock(spec=LLM)
    llm.call.return_value = """
output = 'Sample book data'
if isinstance(output, str):
    result = (True, output)
else:
    result = (False, 'Invalid output format')
print(result)
"""
    return llm


@pytest.mark.parametrize(
    "tool_run_output",
    [
        {
            "output": "(True, 'Valid output')",
            "expected_result": True,
            "expected_output": "Valid output",
        },
        {
            "output": "(False, 'Invalid output format')",
            "expected_result": False,
            "expected_output": "Invalid output format",
        },
        {
            "output": "Something went wrong while running the code, Invalid output format",
            "expected_result": False,
            "expected_output": "Something went wrong while running the code, Invalid output format",
        },
        {
            "output": "No result variable found",
            "expected_result": False,
            "expected_output": "No result variable found",
        },
        {
            "output": (False, "Invalid output format"),
            "expected_result": False,
            "expected_output": "Invalid output format",
        },
        {
            "output": "bla-bla-bla",
            "expected_result": False,
            "expected_output": "Error parsing result: malformed node or string on line 1",
        },
    ],
)
@patch("crewai_tools.CodeInterpreterTool.run")
def test_task_guardrail_execute_code(mock_run, mock_llm, tool_run_output, task_output):
    mock_run.return_value = tool_run_output["output"]

    guardrail = TaskGuardrail(description="Test validation", llm=mock_llm)

    result = guardrail(task_output)
    assert result[0] == tool_run_output["expected_result"]
    assert tool_run_output["expected_output"] in result[1]


@patch("crewai_tools.CodeInterpreterTool.run")
def test_guardrail_using_additional_instructions(mock_run, mock_llm, task_output):
    mock_run.return_value = "(True, 'Valid output')"
    additional_instructions = (
        "This is an additional instruction created by the user follow it strictly"
    )
    guardrail = TaskGuardrail(
        description="Test validation",
        llm=mock_llm,
        additional_instructions=additional_instructions,
    )

    guardrail(task_output)

    assert additional_instructions in str(mock_llm.call.call_args)


# TODO: missing a test to cover callable func guardrail
@pytest.mark.vcr(filter_headers=["authorization"])
def test_guardrail_emits_events(sample_agent):
    started_guardrail = []
    completed_guardrail = []

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(TaskGuardrailStartedEvent)
        def handle_guardrail_started(source, event):
            started_guardrail.append(
                {"guardrail": event.guardrail, "retry_count": event.retry_count}
            )

        @crewai_event_bus.on(TaskGuardrailCompletedEvent)
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
            description="Test task",
            expected_output="Output",
            guardrail="Ensure the output is equal to 'good result'",
        )

        with (
            patch(
                "crewai_tools.CodeInterpreterTool.run",
                side_effect=[
                    "Something went wrong while running the code",
                    (True, "good result"),
                ],
            ),
            patch(
                "crewai.tasks.task_guardrail.TaskGuardrail.generate_code",
                return_value="""def guardrail(result: TaskOutput):
    return (True, result.raw.upper())""",
            ),
        ):
            task.execute_sync(agent=sample_agent)

        def custom_guardrail(result: TaskOutput):
            return (True, "good result from callable function")

        task = Task(
            description="Test task",
            expected_output="Output",
            guardrail=custom_guardrail,
        )

        task.execute_sync(agent=sample_agent)

        expected_started_events = [
            {
                "guardrail": """def guardrail(result: TaskOutput):
    return (True, result.raw.upper())""",
                "retry_count": 0,
            },
            {
                "guardrail": """def guardrail(result: TaskOutput):
    return (True, result.raw.upper())""",
                "retry_count": 1,
            },
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
                "error": "Something went wrong while running the code",
                "retry_count": 0,
            },
            {
                "success": True,
                "result": "good result",
                "error": None,
                "retry_count": 1,
            },
            {
                "success": True,
                "result": "good result from callable function",
                "error": None,
                "retry_count": 0,
            },
        ]
        assert started_guardrail == expected_started_events
        assert completed_guardrail == expected_completed_events


def test_task_guardrail_when_docker_is_not_available(mock_llm, task_output):
    guardrail = TaskGuardrail(description="Test validation", llm=mock_llm)
    with (
        patch(
            "crewai_tools.CodeInterpreterTool.__init__", return_value=None
        ) as mock_init,
        patch(
            "crewai_tools.CodeInterpreterTool.run", return_value=(True, "Valid output")
        ),
        patch(
            "subprocess.run",
            side_effect=FileNotFoundError,
        ),
    ):
        guardrail(task_output)

    mock_init.assert_called_once_with(code=ANY, unsafe_mode=True)


def test_task_guardrail_when_docker_is_available(mock_llm, task_output):
    guardrail = TaskGuardrail(description="Test validation", llm=mock_llm)
    with (
        patch(
            "crewai_tools.CodeInterpreterTool.__init__", return_value=None
        ) as mock_init,
        patch(
            "crewai_tools.CodeInterpreterTool.run", return_value=(True, "Valid output")
        ) as mock_run,
        patch(
            "subprocess.run",
            return_value=True,
        ),
    ):
        guardrail(task_output)

    mock_init.assert_called_once_with(code=ANY, unsafe_mode=False)


def test_task_guardrail_when_tool_output_is_not_valid(mock_llm, task_output):
    guardrail = TaskGuardrail(description="Test validation", llm=mock_llm)
    with (
        patch(
            "crewai_tools.CodeInterpreterTool.__init__", return_value=None
        ) as mock_init,
        patch(
            "crewai_tools.CodeInterpreterTool.run", return_value=(True, "Valid output")
        ) as mock_run,
        patch(
            "subprocess.run",
            return_value=True,
        ) as docker_check,
    ):
        guardrail(task_output)

    mock_init.assert_called_once_with(code=ANY, unsafe_mode=False)
    docker_check.assert_called_once()


@pytest.mark.parametrize("unsafe_mode", [True, False])
def test_task_guardrail_force_code_tool_unsafe_mode(mock_llm, task_output, unsafe_mode):
    guardrail = TaskGuardrail(
        description="Test validation", llm=mock_llm, unsafe_mode=unsafe_mode
    )
    with (
        patch(
            "crewai_tools.CodeInterpreterTool.__init__", return_value=None
        ) as mock_init,
        patch(
            "crewai_tools.CodeInterpreterTool.run", return_value=(True, "Valid output")
        ),
        patch(
            "subprocess.run",
            side_effect=FileNotFoundError,
        ) as docker_check,
    ):
        guardrail(task_output)

    docker_check.assert_not_called()
    mock_init.assert_called_once_with(code=ANY, unsafe_mode=unsafe_mode)
