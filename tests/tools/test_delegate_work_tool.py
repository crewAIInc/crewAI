from unittest.mock import MagicMock

from crewai.tools.agent_tools.delegate_work_tool import (
    DelegateWorkTool,
    DelegateWorkToolSchema,
)


def test_delegate_work_tool_with_string_inputs():
    """Test that DelegateWorkTool works with string inputs (original behavior)."""
    # Create a mock for _get_coworker and _execute
    delegate_tool = DelegateWorkTool()
    delegate_tool._get_coworker = MagicMock(return_value="Researcher")
    delegate_tool._execute = MagicMock(return_value="Task delegated successfully")

    # Test with string inputs
    task_str = "Research AI history"
    context_str = "Need comprehensive information about AI development"

    result = delegate_tool._run(
        task=task_str, context=context_str, coworker="Researcher"
    )

    # Verify the correct values were passed to _execute
    delegate_tool._get_coworker.assert_called_once_with("Researcher", **{})
    delegate_tool._execute.assert_called_once_with("Researcher", task_str, context_str)
    assert result == "Task delegated successfully"


def test_delegate_work_tool_with_dict_inputs():
    """Test that DelegateWorkTool works with dictionary inputs (new behavior)."""
    # Create a mock for _get_coworker and _execute
    delegate_tool = DelegateWorkTool()
    delegate_tool._get_coworker = MagicMock(return_value="Writer")
    delegate_tool._execute = MagicMock(return_value="Task delegated successfully")

    # Test with dictionary inputs
    task_dict = {"description": "Write summary of research", "type": "str"}
    context_dict = {
        "description": "Use the AI research to create a concise summary",
        "type": "str",
    }

    result = delegate_tool._run(task=task_dict, context=context_dict, coworker="Writer")

    # Verify the correct values were passed to _execute
    delegate_tool._get_coworker.assert_called_once_with("Writer", **{})
    delegate_tool._execute.assert_called_once_with(
        "Writer",
        "Write summary of research",
        "Use the AI research to create a concise summary",
    )
    assert result == "Task delegated successfully"


def test_delegate_work_tool_with_mixed_inputs():
    """Test that DelegateWorkTool works with mixed inputs (string and dictionary)."""
    # Create a mock for _get_coworker and _execute
    delegate_tool = DelegateWorkTool()
    delegate_tool._get_coworker = MagicMock(return_value="Reviewer")
    delegate_tool._execute = MagicMock(return_value="Task delegated successfully")

    # Test with mixed inputs
    task_str = "Review the AI summary"
    context_dict = {
        "description": "Check for accuracy and clarity in the summary",
        "type": "str",
    }

    result = delegate_tool._run(
        task=task_str, context=context_dict, coworker="Reviewer"
    )

    # Verify the correct values were passed to _execute
    delegate_tool._get_coworker.assert_called_once_with("Reviewer", **{})
    delegate_tool._execute.assert_called_once_with(
        "Reviewer",
        "Review the AI summary",
        "Check for accuracy and clarity in the summary",
    )
    assert result == "Task delegated successfully"


def test_delegate_work_tool_with_dict_without_description():
    """Test that DelegateWorkTool handles dictionaries without 'description' field."""
    # Create a mock for _get_coworker and _execute
    delegate_tool = DelegateWorkTool()
    delegate_tool._get_coworker = MagicMock(return_value="Analyst")
    delegate_tool._execute = MagicMock(return_value="Task delegated successfully")

    # Test with dictionary missing description
    task_dict = {"other_field": "Some value", "type": "str"}
    context_dict = {"description": "Context information", "type": "str"}

    result = delegate_tool._run(
        task=task_dict, context=context_dict, coworker="Analyst"
    )

    # Verify the original dictionary was passed to _execute
    delegate_tool._get_coworker.assert_called_once_with("Analyst", **{})
    delegate_tool._execute.assert_called_once_with(
        "Analyst", task_dict, "Context information"
    )
    assert result == "Task delegated successfully"


def test_delegate_work_tool_schema_validation():
    """Test that the updated DelegateWorkToolSchema accepts both string and dict inputs."""
    # Test validation with string inputs
    string_data = {
        "task": "Research task",
        "context": "Research context",
        "coworker": "Researcher",
    }
    model = DelegateWorkToolSchema(**string_data)
    assert model.task == "Research task"
    assert model.context == "Research context"
    assert model.coworker == "Researcher"

    # Test validation with dictionary inputs
    dict_data = {
        "task": {"description": "Write task", "type": "str"},
        "context": {"description": "Writing context", "type": "str"},
        "coworker": "Writer",
    }
    model = DelegateWorkToolSchema(**dict_data)
    assert model.task == {"description": "Write task", "type": "str"}
    assert model.context == {"description": "Writing context", "type": "str"}
    assert model.coworker == "Writer"

    # Test validation with mixed inputs
    mixed_data = {
        "task": "Review task",
        "context": {"description": "Review context", "type": "str"},
        "coworker": "Reviewer",
    }
    model = DelegateWorkToolSchema(**mixed_data)
    assert model.task == "Review task"
    assert model.context == {"description": "Review context", "type": "str"}
    assert model.coworker == "Reviewer"
