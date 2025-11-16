"""Tests for crew_base._map_task_variables handling context: None."""

import pytest

from crewai import Agent, Task
from crewai.project.crew_base import CrewBase


@CrewBase
class TestCrew:
    """Test crew for context: None handling."""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"


class TestMapTaskVariablesContextNone:
    """Test suite for _map_task_variables handling context: None."""

    def test_map_task_variables_with_context_none(self):
        """Test that context: None in task_info is preserved in tasks_config."""
        from crewai.project.crew_base import _map_task_variables

        class MockCrewInstance:
            def __init__(self):
                self.tasks_config = {
                    "test_task": {
                        "description": "Test task",
                        "expected_output": "Test output",
                        "context": None
                    }
                }

        instance = MockCrewInstance()
        task_info = {"context": None}
        
        _map_task_variables(
            instance,
            task_name="test_task",
            task_info=task_info,
            agents={},
            tasks={},
            output_json_functions={},
            tool_functions={},
            callback_functions={},
            output_pydantic_functions={}
        )

        assert instance.tasks_config["test_task"]["context"] is None

    def test_map_task_variables_with_context_empty_list(self):
        """Test that context: [] in task_info is preserved as empty list."""
        from crewai.project.crew_base import _map_task_variables

        class MockCrewInstance:
            def __init__(self):
                self.tasks_config = {
                    "test_task": {
                        "description": "Test task",
                        "expected_output": "Test output"
                    }
                }

        instance = MockCrewInstance()
        task_info = {"context": []}
        
        _map_task_variables(
            instance,
            task_name="test_task",
            task_info=task_info,
            agents={},
            tasks={},
            output_json_functions={},
            tool_functions={},
            callback_functions={},
            output_pydantic_functions={}
        )

        assert instance.tasks_config["test_task"]["context"] == []

    def test_map_task_variables_with_context_list(self):
        """Test that context with task names is resolved to Task instances."""
        from crewai.project.crew_base import _map_task_variables

        class MockCrewInstance:
            def __init__(self):
                self.tasks_config = {
                    "test_task": {
                        "description": "Test task",
                        "expected_output": "Test output"
                    }
                }

        instance = MockCrewInstance()
        
        task1 = Task(description="Task 1", expected_output="Output 1")
        task2 = Task(description="Task 2", expected_output="Output 2")
        
        tasks = {
            "task1": lambda: task1,
            "task2": lambda: task2
        }
        
        task_info = {"context": ["task1", "task2"]}
        
        _map_task_variables(
            instance,
            task_name="test_task",
            task_info=task_info,
            agents={},
            tasks=tasks,
            output_json_functions={},
            tool_functions={},
            callback_functions={},
            output_pydantic_functions={}
        )

        assert len(instance.tasks_config["test_task"]["context"]) == 2
        assert instance.tasks_config["test_task"]["context"][0] is task1
        assert instance.tasks_config["test_task"]["context"][1] is task2

    def test_map_task_variables_without_context_key(self):
        """Test that missing context key doesn't add context to tasks_config."""
        from crewai.project.crew_base import _map_task_variables

        class MockCrewInstance:
            def __init__(self):
                self.tasks_config = {
                    "test_task": {
                        "description": "Test task",
                        "expected_output": "Test output"
                    }
                }

        instance = MockCrewInstance()
        task_info = {}
        
        _map_task_variables(
            instance,
            task_name="test_task",
            task_info=task_info,
            agents={},
            tasks={},
            output_json_functions={},
            tool_functions={},
            callback_functions={},
            output_pydantic_functions={}
        )

        assert "context" not in instance.tasks_config["test_task"]


class TestTaskWithContextNoneFromConfig:
    """Integration tests for Task creation with context: None from config."""

    def test_task_with_context_none_from_config(self):
        """Test that Task can be created with config containing context: None."""
        task = Task(
            description="Test task",
            expected_output="Test output",
            config={"context": None}
        )

        assert task.context is None
        assert task.description == "Test task"
        assert task.expected_output == "Test output"

    def test_task_with_context_none_direct(self):
        """Test that Task can be created with context=None directly."""
        task = Task(
            description="Test task",
            expected_output="Test output",
            context=None
        )

        assert task.context is None
        assert task.description == "Test task"
        assert task.expected_output == "Test output"

    def test_task_with_context_empty_list_from_config(self):
        """Test that Task can be created with config containing context: []."""
        task = Task(
            description="Test task",
            expected_output="Test output",
            config={"context": []}
        )

        assert task.context == []
        assert task.description == "Test task"
        assert task.expected_output == "Test output"

    def test_task_without_context_uses_default(self):
        """Test that Task without context uses NOT_SPECIFIED default."""
        from crewai.utilities.constants import NOT_SPECIFIED
        
        task = Task(
            description="Test task",
            expected_output="Test output"
        )

        assert task.context is NOT_SPECIFIED
        assert task.description == "Test task"
        assert task.expected_output == "Test output"

    def test_task_with_context_list_from_config(self):
        """Test that Task can be created with config containing context list."""
        context_task = Task(
            description="Context task",
            expected_output="Context output"
        )
        
        task = Task(
            description="Test task",
            expected_output="Test output",
            config={"context": [context_task]}
        )

        assert isinstance(task.context, list)
        assert len(task.context) == 1
        assert task.context[0] is context_task
