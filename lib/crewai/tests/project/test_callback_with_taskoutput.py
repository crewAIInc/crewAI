"""Test callback decorator with TaskOutput arguments."""

from collections.abc import Callable
from pathlib import Path
from typing import Any
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


def _sample_output() -> TaskOutput:
    return TaskOutput(
        description="d",
        agent="Writer",
        raw="done",
        output_format=OutputFormat.RAW,
    )


def _crew_from_task_yaml(
    tmp_path: Path, task_fields: str, *, factory: bool
) -> Any:
    """Build a crew whose task callback comes from tasks.yaml."""
    agents = tmp_path / "agents.yaml"
    tasks = tmp_path / "tasks.yaml"
    agents.write_text(
        "writer:\n  role: Writer\n  goal: Write\n  backstory: Writes\n",
        encoding="utf-8",
    )
    tasks.write_text(
        f"write:\n  description: d\n  expected_output: e\n{task_fields}",
        encoding="utf-8",
    )

    if factory:

        @CrewBase
        class FactoryCrew:
            """Crew whose callback method returns the task callable."""

            agents_config = str(agents)
            tasks_config = str(tasks)

            def __init__(self) -> None:
                self.seen: list[TaskOutput] = []

            @callback
            def on_done(self) -> Callable[[TaskOutput], None]:
                def inner(output: TaskOutput) -> None:
                    self.seen.append(output)

                return inner

            @task
            def write(self) -> Task:
                return Task(config=self.tasks_config["write"])

        return FactoryCrew

    @CrewBase
    class DirectCrew:
        """Crew whose callback method accepts the task output."""

        agents_config = str(agents)
        tasks_config = str(tasks)

        def __init__(self) -> None:
            self.seen: list[TaskOutput] = []

        @callback
        def on_done(self, output: TaskOutput) -> None:
            self.seen.append(output)

        @callback
        def on_also(self, output: TaskOutput) -> None:
            self.seen.append(output)

        @task
        def write(self) -> Task:
            return Task(config=self.tasks_config["write"])

    return DirectCrew


def test_yaml_callback_receives_task_output(tmp_path: Path) -> None:
    """tasks.yaml callback: name becomes Task.callback and receives output."""
    crew = _crew_from_task_yaml(tmp_path, "  callback: on_done\n", factory=False)()
    task = crew.write()
    output = _sample_output()

    assert crew.seen == []
    assert task.callback is not None
    task.callback(output)
    assert crew.seen == [output]


def test_yaml_callbacks_list_is_not_invoked_during_build(tmp_path: Path) -> None:
    """tasks.yaml callbacks are attached, not called while the crew is built."""
    crew = _crew_from_task_yaml(
        tmp_path, "  callbacks:\n    - on_done\n", factory=False
    )()
    task = crew.write()
    output = _sample_output()

    assert crew.seen == []
    assert task.callback is not None
    task.callback(output)
    assert crew.seen == [output]


def test_yaml_callback_factory_is_stored_on_task(tmp_path: Path) -> None:
    """A callback factory's returned callable is Task.callback."""
    crew = _crew_from_task_yaml(
        tmp_path, "  callbacks:\n    - on_done\n", factory=True
    )()
    task = crew.write()
    output = _sample_output()

    assert crew.seen == []
    assert task.callback is not None
    task.callback(output)
    assert crew.seen == [output]


def test_yaml_callbacks_list_runs_every_named_callback(tmp_path: Path) -> None:
    """Every name in callbacks runs with the task output."""
    crew = _crew_from_task_yaml(
        tmp_path,
        "  callbacks:\n    - on_done\n    - on_also\n",
        factory=False,
    )()
    task = crew.write()
    output = _sample_output()

    assert crew.seen == []
    assert task.callback is not None
    task.callback(output)
    assert crew.seen == [output, output]