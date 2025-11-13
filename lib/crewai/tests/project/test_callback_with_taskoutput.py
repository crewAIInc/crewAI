"""Test callback decorator with TaskOutput arguments."""

from unittest.mock import MagicMock, patch

from crewai import Agent, Crew, Task
from crewai.project import CrewBase, callback, task
from crewai.tasks.output_format import OutputFormat
from crewai.tasks.task_output import TaskOutput


def test_callback_decorator_with_taskoutput() -> None:
    """Test that @callback decorator works with TaskOutput arguments."""

    @CrewBase
    class TestCrew:
        """Test crew with callback."""

        callback_called = False
        callback_output = None

        @callback
        def task_callback(self, output: TaskOutput) -> None:
            """Test callback that receives TaskOutput."""
            self.callback_called = True
            self.callback_output = output

        @task
        def test_task(self) -> Task:
            """Test task with callback."""
            return Task(
                description="Test task",
                expected_output="Test output",
                callback=self.task_callback,
            )

    test_crew = TestCrew()
    task_instance = test_crew.test_task()

    test_output = TaskOutput(
        description="Test task",
        agent="Test Agent",
        raw="test result",
        output_format=OutputFormat.RAW,
    )

    task_instance.callback(test_output)

    assert test_crew.callback_called
    assert test_crew.callback_output == test_output


def test_callback_decorator_with_taskoutput_integration() -> None:
    """Integration test for callback with actual task execution."""

    @CrewBase
    class TestCrew:
        """Test crew with callback integration."""

        callback_called = False
        received_output: TaskOutput | None = None

        @callback
        def task_callback(self, output: TaskOutput) -> None:
            """Callback executed after task completion."""
            self.callback_called = True
            self.received_output = output

        @task
        def test_task(self) -> Task:
            """Test task."""
            return Task(
                description="Test task",
                expected_output="Test output",
                callback=self.task_callback,
            )

    test_crew = TestCrew()

    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory",
    )

    task_instance = test_crew.test_task()
    task_instance.agent = agent

    with patch.object(Agent, "execute_task") as mock_execute:
        mock_execute.return_value = "test result"
        task_instance.execute_sync()

        assert test_crew.callback_called
        assert test_crew.received_output is not None
        assert test_crew.received_output.raw == "test result"