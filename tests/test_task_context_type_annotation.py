"""
Test for issue #3019: Type annotation for `context` in `Task` is `Optional[List["Task"]]`, but default is `NOT_SPECIFIED`

This test reproduces the type annotation issue and verifies the fix.
"""

import pytest
from typing import get_type_hints, get_origin, get_args
from pydantic import ValidationError

from crewai.task import Task
from crewai.utilities.constants import NOT_SPECIFIED, _NotSpecified


class TestTaskContextTypeAnnotation:
    """Test cases for Task context field type annotation issue."""

    def test_task_context_default_value_is_not_specified(self):
        """Test that Task.context default value is NOT_SPECIFIED sentinel."""
        task = Task(description="Test task", expected_output="Test output")
        assert task.context is NOT_SPECIFIED
        assert isinstance(task.context, _NotSpecified)

    def test_task_context_can_be_set_to_none(self):
        """Test that Task.context can be explicitly set to None."""
        task = Task(description="Test task", expected_output="Test output", context=None)
        assert task.context is None

    def test_task_context_can_be_set_to_empty_list(self):
        """Test that Task.context can be set to an empty list."""
        task = Task(description="Test task", expected_output="Test output", context=[])
        assert task.context == []
        assert isinstance(task.context, list)

    def test_task_context_can_be_set_to_task_list(self):
        """Test that Task.context can be set to a list of tasks."""
        task1 = Task(description="Task 1", expected_output="Output 1")
        task2 = Task(description="Task 2", expected_output="Output 2")
        task3 = Task(description="Task 3", expected_output="Output 3", context=[task1, task2])
        
        assert task3.context == [task1, task2]
        assert isinstance(task3.context, list)
        assert len(task3.context) == 2

    def test_task_context_type_annotation_includes_not_specified(self):
        """Test that the type annotation for context includes _NotSpecified type."""
        type_hints = get_type_hints(Task)
        context_type = type_hints.get('context')
        
        assert context_type is not None
        
        origin = get_origin(context_type)
        if origin is not None:
            args = get_args(context_type)
            
            assert any('_NotSpecified' in str(arg) or arg is _NotSpecified for arg in args), \
                f"Type annotation should include _NotSpecified, got: {args}"

    def test_task_context_distinguishes_not_passed_from_none(self):
        """Test that NOT_SPECIFIED distinguishes between not passed and None."""
        task_not_passed = Task(description="Test task", expected_output="Test output")
        
        task_explicit_none = Task(description="Test task", expected_output="Test output", context=None)
        
        task_empty_list = Task(description="Test task", expected_output="Test output", context=[])
        
        assert task_not_passed.context is NOT_SPECIFIED
        assert task_explicit_none.context is None
        assert task_empty_list.context == []
        
        assert task_not_passed.context is not task_explicit_none.context
        assert task_not_passed.context != task_empty_list.context
        assert task_explicit_none.context != task_empty_list.context

    def test_task_context_usage_in_crew_logic(self):
        """Test that the context field works correctly with crew logic."""
        from crewai.utilities.constants import NOT_SPECIFIED
        
        task_with_not_specified = Task(description="Task 1", expected_output="Output 1")
        task_with_none = Task(description="Task 2", expected_output="Output 2", context=None)
        task_with_empty_list = Task(description="Task 3", expected_output="Output 3", context=[])
        
        assert task_with_not_specified.context is NOT_SPECIFIED
        assert not (task_with_none.context is NOT_SPECIFIED)
        assert not (task_with_empty_list.context is NOT_SPECIFIED)

    def test_task_context_repr_shows_not_specified(self):
        """Test that NOT_SPECIFIED has a proper string representation."""
        task = Task(description="Test task", expected_output="Test output")
        assert str(task.context) == "NOT_SPECIFIED"
        assert repr(task.context) == "NOT_SPECIFIED"

    def test_task_context_validation_accepts_valid_types(self):
        """Test that Task validation accepts all valid context types."""
        try:
            Task(description="Test 1", expected_output="Output 1")
            Task(description="Test 2", expected_output="Output 2", context=None)
            Task(description="Test 3", expected_output="Output 3", context=[])
            
            task1 = Task(description="Task 1", expected_output="Output 1")
            Task(description="Test 4", expected_output="Output 4", context=[task1])
        except ValidationError as e:
            pytest.fail(f"Valid context types should not raise ValidationError: {e}")

    def test_task_context_validation_rejects_invalid_types(self):
        """Test that Task validation rejects invalid context types."""
        with pytest.raises(ValidationError):
            Task(description="Test", expected_output="Output", context="invalid")
            
        with pytest.raises(ValidationError):
            Task(description="Test", expected_output="Output", context=123)
            
        with pytest.raises(ValidationError):
            Task(description="Test", expected_output="Output", context=["not", "tasks"])
