from unittest.mock import Mock, patch

import pytest

from crewai import Agent, Task
from crewai.events.event_bus import crewai_event_bus
from crewai.events.event_types import (
    LLMGuardrailCompletedEvent,
    LLMGuardrailStartedEvent,
)
from crewai.llm import LLM
from crewai.tasks.hallucination_guardrail import HallucinationGuardrail
from crewai.tasks.llm_guardrail import LLMGuardrail
from crewai.tasks.task_output import TaskOutput


def create_smart_task(**kwargs):
    """
    Smart task factory that automatically assigns a mock agent when guardrails are present.
    This maintains backward compatibility while handling the agent requirement for guardrails.
    """
    guardrails_list = kwargs.get("guardrails")
    has_guardrails = kwargs.get("guardrail") is not None or (
        guardrails_list is not None and len(guardrails_list) > 0
    )

    if has_guardrails and kwargs.get("agent") is None:
        kwargs["agent"] = Agent(
            role="test_agent", goal="test_goal", backstory="test_backstory"
        )

    return Task(**kwargs)


def test_task_without_guardrail():
    """Test that tasks work normally without guardrails (backward compatibility)."""
    agent = Mock()
    agent.role = "test_agent"
    agent.execute_task.return_value = "test result"
    agent.crew = None
    agent.last_messages = []

    task = create_smart_task(description="Test task", expected_output="Output")

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
    agent.last_messages = []

    task = create_smart_task(
        description="Test task", expected_output="Output", guardrail=guardrail
    )

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
    agent.last_messages = []

    task = create_smart_task(
        description="Test task",
        expected_output="Output",
        guardrail=guardrail,
        guardrail_max_retries=1,
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
    agent.last_messages = []

    task = create_smart_task(
        description="Test task",
        expected_output="Output",
        guardrail=guardrail,
        guardrail_max_retries=2,
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
    agent.last_messages = []

    task = create_smart_task(
        description="Test task",
        expected_output="Output",
        guardrail=guardrail,
        guardrail_max_retries=1,
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


@pytest.mark.vcr()
def test_task_guardrail_process_output(task_output):
    guardrail = LLMGuardrail(
        description="Ensure the result has less than 10 words", llm=LLM(model="gpt-4o")
    )

    result = guardrail(task_output)
    assert result[0] is False

    assert result[1] == "The task result contains more than 10 words, violating the guardrail. The text provided contains about 21 words."

    guardrail = LLMGuardrail(
        description="Ensure the result has less than 500 words", llm=LLM(model="gpt-4o")
    )

    result = guardrail(task_output)
    assert result[0] is True
    assert result[1] == task_output.raw


@pytest.mark.vcr()
def test_guardrail_emits_events(sample_agent):
    import threading

    started_guardrail = []
    completed_guardrail = []
    condition = threading.Condition()

    @crewai_event_bus.on(LLMGuardrailStartedEvent)
    def handle_guardrail_started(source, event):
        with condition:
            started_guardrail.append(
                {"guardrail": event.guardrail, "retry_count": event.retry_count}
            )
            condition.notify()

    @crewai_event_bus.on(LLMGuardrailCompletedEvent)
    def handle_guardrail_completed(source, event):
        with condition:
            completed_guardrail.append(
                {
                    "success": event.success,
                    "result": event.result,
                    "error": event.error,
                    "retry_count": event.retry_count,
                }
            )
            condition.notify()

    task = create_smart_task(
        description="Gather information about available books on the First World War",
        agent=sample_agent,
        expected_output="A list of available books on the First World War",
        guardrail="Ensure the authors are from Italy",
    )

    result = task.execute_sync(agent=sample_agent)

    with condition:
        success = condition.wait_for(
            lambda: len(started_guardrail) >= 2 and len(completed_guardrail) >= 2,
            timeout=5
        )
    assert success, f"Timeout waiting for first task events. Started: {len(started_guardrail)}, Completed: {len(completed_guardrail)}"

    def custom_guardrail(result: TaskOutput):
        return (True, "good result from callable function")

    task = create_smart_task(
        description="Test task",
        expected_output="Output",
        guardrail=custom_guardrail,
    )

    task.execute_sync(agent=sample_agent)

    with condition:
        success = condition.wait_for(
            lambda: len(started_guardrail) >= 3 and len(completed_guardrail) >= 3,
            timeout=5
        )
    assert success, f"Timeout waiting for second task events. Started: {len(started_guardrail)}, Completed: {len(completed_guardrail)}"

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
            "error": "The output indicates that none of the authors mentioned are from Italy, while the guardrail requires authors to be from Italy. Therefore, the output does not comply with the guardrail.",
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


@pytest.mark.vcr()
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
        task = create_smart_task(
            description="Gather information about available books on the First World War",
            agent=sample_agent,
            expected_output="A list of available books on the First World War",
            guardrail="Ensure the authors are from Italy",
            guardrail_max_retries=0,
        )
        task.execute_sync(agent=sample_agent)


def test_hallucination_guardrail_integration():
    """Test that HallucinationGuardrail integrates properly with the task system."""
    agent = Mock()
    agent.role = "test_agent"
    agent.execute_task.return_value = "test result"
    agent.crew = None
    agent.last_messages = []

    mock_llm = Mock(spec=LLM)
    guardrail = HallucinationGuardrail(
        context="Test reference context for validation", llm=mock_llm, threshold=8.0
    )

    task = create_smart_task(
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


def test_multiple_guardrails_sequential_processing():
    """Test that multiple guardrails are processed sequentially."""

    def first_guardrail(result: TaskOutput) -> tuple[bool, str]:
        """First guardrail adds prefix."""
        return (True, f"[FIRST] {result.raw}")

    def second_guardrail(result: TaskOutput) -> tuple[bool, str]:
        """Second guardrail adds suffix."""
        return (True, f"{result.raw} [SECOND]")

    def third_guardrail(result: TaskOutput) -> tuple[bool, str]:
        """Third guardrail converts to uppercase."""
        return (True, result.raw.upper())

    agent = Mock()
    agent.role = "sequential_agent"
    agent.execute_task.return_value = "original text"
    agent.crew = None
    agent.last_messages = []

    task = create_smart_task(
        description="Test sequential guardrails",
        expected_output="Processed text",
        guardrails=[first_guardrail, second_guardrail, third_guardrail],
    )

    result = task.execute_sync(agent=agent)
    assert result.raw == "[FIRST] ORIGINAL TEXT [SECOND]"


def test_multiple_guardrails_with_validation_failure():
    """Test multiple guardrails where one fails validation."""

    def length_guardrail(result: TaskOutput) -> tuple[bool, str]:
        """Ensure minimum length."""
        if len(result.raw) < 10:
            return (False, "Text too short")
        return (True, result.raw)

    def format_guardrail(result: TaskOutput) -> tuple[bool, str]:
        """Add formatting only if not already formatted."""
        if not result.raw.startswith("Formatted:"):
            return (True, f"Formatted: {result.raw}")
        return (True, result.raw)

    def validation_guardrail(result: TaskOutput) -> tuple[bool, str]:
        """Final validation."""
        if "Formatted:" not in result.raw:
            return (False, "Missing formatting")
        return (True, result.raw)

    # Use a callable that tracks calls and returns appropriate values
    call_count = 0

    def mock_execute_task(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        result = (
            "short"
            if call_count == 1
            else "this is a longer text that meets requirements"
        )
        return result

    agent = Mock()
    agent.role = "validation_agent"
    agent.execute_task = mock_execute_task
    agent.crew = None
    agent.last_messages = []

    task = create_smart_task(
        description="Test guardrails with validation",
        expected_output="Valid formatted text",
        guardrails=[length_guardrail, format_guardrail, validation_guardrail],
        guardrail_max_retries=2,
    )

    result = task.execute_sync(agent=agent)
    # The second call should be processed through all guardrails
    assert result.raw == "Formatted: this is a longer text that meets requirements"
    assert task._guardrail_retry_counts.get(0, 0) == 1


def test_multiple_guardrails_with_mixed_string_and_taskoutput():
    """Test guardrails that return both strings and TaskOutput objects."""

    def string_guardrail(result: TaskOutput) -> tuple[bool, str]:
        """Returns a string."""
        return (True, f"String: {result.raw}")

    def taskoutput_guardrail(result: TaskOutput) -> tuple[bool, TaskOutput]:
        """Returns a TaskOutput object."""
        new_output = TaskOutput(
            name=result.name,
            description=result.description,
            expected_output=result.expected_output,
            raw=f"TaskOutput: {result.raw}",
            agent=result.agent,
            output_format=result.output_format,
        )
        return (True, new_output)

    def final_string_guardrail(result: TaskOutput) -> tuple[bool, str]:
        """Final string transformation."""
        return (True, f"Final: {result.raw}")

    agent = Mock()
    agent.role = "mixed_agent"
    agent.execute_task.return_value = "original"
    agent.crew = None
    agent.last_messages = []

    task = create_smart_task(
        description="Test mixed return types",
        expected_output="Mixed processing",
        guardrails=[string_guardrail, taskoutput_guardrail, final_string_guardrail],
    )

    result = task.execute_sync(agent=agent)
    assert result.raw == "Final: TaskOutput: String: original"


def test_multiple_guardrails_with_retry_on_middle_guardrail():
    """Test that retry works correctly when a middle guardrail fails."""

    call_count = {"first": 0, "second": 0, "third": 0}

    def first_guardrail(result: TaskOutput) -> tuple[bool, str]:
        """Always succeeds."""
        call_count["first"] += 1
        return (True, f"First({call_count['first']}): {result.raw}")

    def second_guardrail(result: TaskOutput) -> tuple[bool, str]:
        """Fails on first attempt, succeeds on second."""
        call_count["second"] += 1
        if call_count["second"] == 1:
            return (False, "Second guardrail failed on first attempt")
        return (True, f"Second({call_count['second']}): {result.raw}")

    def third_guardrail(result: TaskOutput) -> tuple[bool, str]:
        """Always succeeds."""
        call_count["third"] += 1
        return (True, f"Third({call_count['third']}): {result.raw}")

    agent = Mock()
    agent.role = "retry_agent"
    agent.execute_task.return_value = "base"
    agent.crew = None
    agent.last_messages = []

    task = create_smart_task(
        description="Test retry in middle guardrail",
        expected_output="Retry handling",
        guardrails=[first_guardrail, second_guardrail, third_guardrail],
        guardrail_max_retries=2,
    )

    result = task.execute_sync(agent=agent)
    assert task._guardrail_retry_counts.get(1, 0) == 1
    assert call_count["first"] == 1
    assert call_count["second"] == 2
    assert call_count["third"] == 1
    assert "Second(2)" in result.raw


def test_multiple_guardrails_with_max_retries_exceeded():
    """Test that exception is raised when max retries exceeded with multiple guardrails."""

    def passing_guardrail(result: TaskOutput) -> tuple[bool, str]:
        """Always passes."""
        return (True, f"Passed: {result.raw}")

    def failing_guardrail(result: TaskOutput) -> tuple[bool, str]:
        """Always fails."""
        return (False, "This guardrail always fails")

    agent = Mock()
    agent.role = "failing_agent"
    agent.execute_task.return_value = "test"
    agent.crew = None
    agent.last_messages = []

    task = create_smart_task(
        description="Test max retries with multiple guardrails",
        expected_output="Will fail",
        guardrails=[passing_guardrail, failing_guardrail],
        guardrail_max_retries=1,
    )

    with pytest.raises(Exception) as exc_info:
        task.execute_sync(agent=agent)

    assert "Task failed guardrail 1 validation after 1 retries" in str(exc_info.value)
    assert "This guardrail always fails" in str(exc_info.value)
    assert task._guardrail_retry_counts.get(1, 0) == 1


def test_multiple_guardrails_empty_list():
    """Test that empty guardrails list works correctly."""

    agent = Mock()
    agent.role = "empty_agent"
    agent.execute_task.return_value = "no guardrails"
    agent.crew = None
    agent.last_messages = []

    task = create_smart_task(
        description="Test empty guardrails list",
        expected_output="No processing",
        guardrails=[],
    )

    result = task.execute_sync(agent=agent)
    assert result.raw == "no guardrails"


def test_multiple_guardrails_with_llm_guardrails():
    """Test mixing callable and LLM guardrails."""

    def callable_guardrail(result: TaskOutput) -> tuple[bool, str]:
        """Callable guardrail."""
        return (True, f"Callable: {result.raw}")

    # Create a proper mock agent without config issues
    from crewai import Agent

    agent = Agent(
        role="mixed_guardrail_agent", goal="Test goal", backstory="Test backstory"
    )

    task = create_smart_task(
        description="Test mixed guardrail types",
        expected_output="Mixed processing",
        guardrails=[callable_guardrail, "Ensure the output is professional"],
        agent=agent,
    )

    # The LLM guardrail will be converted to LLMGuardrail internally
    assert len(task._guardrails) == 2
    assert callable(task._guardrails[0])
    assert callable(task._guardrails[1])  # LLMGuardrail is callable


def test_multiple_guardrails_processing_order():
    """Test that guardrails are processed in the correct order."""

    processing_order = []

    def first_guardrail(result: TaskOutput) -> tuple[bool, str]:
        processing_order.append("first")
        return (True, f"1-{result.raw}")

    def second_guardrail(result: TaskOutput) -> tuple[bool, str]:
        processing_order.append("second")
        return (True, f"2-{result.raw}")

    def third_guardrail(result: TaskOutput) -> tuple[bool, str]:
        processing_order.append("third")
        return (True, f"3-{result.raw}")

    agent = Mock()
    agent.role = "order_agent"
    agent.execute_task.return_value = "base"
    agent.crew = None
    agent.last_messages = []

    task = create_smart_task(
        description="Test processing order",
        expected_output="Ordered processing",
        guardrails=[first_guardrail, second_guardrail, third_guardrail],
    )

    result = task.execute_sync(agent=agent)
    assert processing_order == ["first", "second", "third"]
    assert result.raw == "3-2-1-base"


def test_multiple_guardrails_with_pydantic_output():
    """Test multiple guardrails with Pydantic output model."""
    from pydantic import BaseModel, Field

    class TestModel(BaseModel):
        content: str = Field(description="The content")
        processed: bool = Field(description="Whether it was processed")

    def json_guardrail(result: TaskOutput) -> tuple[bool, str]:
        """Convert to JSON format."""
        import json

        data = {"content": result.raw, "processed": True}
        return (True, json.dumps(data))

    def validation_guardrail(result: TaskOutput) -> tuple[bool, str]:
        """Validate JSON structure."""
        import json

        try:
            data = json.loads(result.raw)
            if "content" not in data or "processed" not in data:
                return (False, "Missing required fields")
            return (True, result.raw)
        except json.JSONDecodeError:
            return (False, "Invalid JSON format")

    agent = Mock()
    agent.role = "pydantic_agent"
    agent.execute_task.return_value = "test content"
    agent.crew = None
    agent.last_messages = []

    task = create_smart_task(
        description="Test guardrails with Pydantic",
        expected_output="Structured output",
        guardrails=[json_guardrail, validation_guardrail],
        output_pydantic=TestModel,
    )

    result = task.execute_sync(agent=agent)

    # Verify the result is valid JSON and can be parsed
    import json

    parsed = json.loads(result.raw)
    assert parsed["content"] == "test content"
    assert parsed["processed"] is True


def test_guardrails_vs_single_guardrail_mutual_exclusion():
    """Test that guardrails list nullifies single guardrail."""

    def single_guardrail(result: TaskOutput) -> tuple[bool, str]:
        """Single guardrail - should be ignored."""
        return (True, f"Single: {result.raw}")

    def list_guardrail(result: TaskOutput) -> tuple[bool, str]:
        """List guardrail - should be used."""
        return (True, f"List: {result.raw}")

    agent = Mock()
    agent.role = "exclusion_agent"
    agent.execute_task.return_value = "test"
    agent.crew = None
    agent.last_messages = []

    task = create_smart_task(
        description="Test mutual exclusion",
        expected_output="Exclusion test",
        guardrail=single_guardrail,  # This should be ignored
        guardrails=[list_guardrail],  # This should be used
    )

    result = task.execute_sync(agent=agent)
    # Should only use the guardrails list, not the single guardrail
    assert result.raw == "List: test"
    assert task._guardrail is None  # Single guardrail should be nullified


def test_per_guardrail_independent_retry_tracking():
    """Test that each guardrail has independent retry tracking."""

    call_counts = {"g1": 0, "g2": 0, "g3": 0}

    def guardrail_1(result: TaskOutput) -> tuple[bool, str]:
        """Fails twice, then succeeds."""
        call_counts["g1"] += 1
        if call_counts["g1"] <= 2:
            return (False, "Guardrail 1 not ready yet")
        return (True, f"G1({call_counts['g1']}): {result.raw}")

    def guardrail_2(result: TaskOutput) -> tuple[bool, str]:
        """Fails once, then succeeds."""
        call_counts["g2"] += 1
        if call_counts["g2"] == 1:
            return (False, "Guardrail 2 not ready yet")
        return (True, f"G2({call_counts['g2']}): {result.raw}")

    def guardrail_3(result: TaskOutput) -> tuple[bool, str]:
        """Always succeeds."""
        call_counts["g3"] += 1
        return (True, f"G3({call_counts['g3']}): {result.raw}")

    agent = Mock()
    agent.role = "independent_retry_agent"
    agent.execute_task.return_value = "base"
    agent.crew = None
    agent.last_messages = []

    task = create_smart_task(
        description="Test independent retry tracking",
        expected_output="Independent retries",
        guardrails=[guardrail_1, guardrail_2, guardrail_3],
        guardrail_max_retries=3,
    )

    result = task.execute_sync(agent=agent)

    assert task._guardrail_retry_counts.get(0, 0) == 2
    assert task._guardrail_retry_counts.get(1, 0) == 1
    assert task._guardrail_retry_counts.get(2, 0) == 0

    assert call_counts["g1"] == 3
    assert call_counts["g2"] == 2
    assert call_counts["g3"] == 1

    assert "G3(1)" in result.raw
