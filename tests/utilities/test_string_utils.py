from typing import Any

import pytest

from crewai.utilities.string_utils import interpolate_only


class TestInterpolateOnly:
    """Tests for the interpolate_only function in string_utils.py."""

    def test_basic_variable_interpolation(self) -> None:
        """Test basic variable interpolation works correctly."""
        template = "Hello, {name}! Welcome to {company}."
        inputs: dict[str, str | int | float | dict[str, Any] | list[Any]] = {
            "name": "Alice",
            "company": "CrewAI",
        }

        result = interpolate_only(template, inputs)

        assert result == "Hello, Alice! Welcome to CrewAI."

    def test_multiple_occurrences_of_same_variable(self) -> None:
        """Test that multiple occurrences of the same variable are replaced."""
        template = "{name} is using {name}'s account."
        inputs: dict[str, str | int | float | dict[str, Any] | list[Any]] = {
            "name": "Bob",
        }

        result = interpolate_only(template, inputs)

        assert result == "Bob is using Bob's account."

    def test_json_structure_preservation(self) -> None:
        """Test that JSON structures are preserved and not interpolated incorrectly."""
        template = """
        Instructions for {agent}:

        Please return the following object:

        {"name": "person's name", "age": 25, "skills": ["coding", "testing"]}
        """
        inputs: dict[str, str | int | float | dict[str, Any] | list[Any]] = {
            "agent": "DevAgent",
        }

        result = interpolate_only(template, inputs)

        assert "Instructions for DevAgent:" in result
        assert (
            '{"name": "person\'s name", "age": 25, "skills": ["coding", "testing"]}'
            in result
        )

    def test_complex_nested_json(self) -> None:
        """Test with complex JSON structures containing curly braces."""
        template = """
        {agent} needs to process:
        {
          "config": {
            "nested": {
              "value": 42
            },
            "arrays": [1, 2, {"inner": "value"}]
          }
        }
        """
        inputs: dict[str, str | int | float | dict[str, Any] | list[Any]] = {
            "agent": "DataProcessor",
        }

        result = interpolate_only(template, inputs)

        assert "DataProcessor needs to process:" in result
        assert '"nested": {' in result
        assert '"value": 42' in result
        assert '[1, 2, {"inner": "value"}]' in result

    def test_missing_variable(self) -> None:
        """Test that an error is raised when a required variable is missing."""
        template = "Hello, {name}!"
        inputs: dict[str, str | int | float | dict[str, Any] | list[Any]] = {
            "not_name": "Alice",
        }

        with pytest.raises(KeyError) as excinfo:
            interpolate_only(template, inputs)

        assert "template variable" in str(excinfo.value).lower()
        assert "name" in str(excinfo.value)

    def test_invalid_input_types(self) -> None:
        """Test that an error is raised with invalid input types."""
        template = "Hello, {name}!"
        # Using Any for this test since we're intentionally testing an invalid type
        inputs: dict[str, Any] = {"name": object()}  # Object is not a valid input type

        with pytest.raises(ValueError) as excinfo:
            interpolate_only(template, inputs)

        assert "unsupported type" in str(excinfo.value).lower()

    def test_empty_input_string(self) -> None:
        """Test handling of empty or None input string."""
        inputs: dict[str, str | int | float | dict[str, Any] | list[Any]] = {
            "name": "Alice",
        }

        assert interpolate_only("", inputs) == ""
        assert interpolate_only(None, inputs) == ""

    def test_no_variables_in_template(self) -> None:
        """Test a template with no variables to replace."""
        template = "This is a static string with no variables."
        inputs: dict[str, str | int | float | dict[str, Any] | list[Any]] = {
            "name": "Alice",
        }

        result = interpolate_only(template, inputs)

        assert result == template

    def test_variable_name_starting_with_underscore(self) -> None:
        """Test variables starting with underscore are replaced correctly."""
        template = "Variable: {_special_var}"
        inputs: dict[str, str | int | float | dict[str, Any] | list[Any]] = {
            "_special_var": "Special Value",
        }

        result = interpolate_only(template, inputs)

        assert result == "Variable: Special Value"

    def test_preserves_non_matching_braces(self) -> None:
        """Test that non-matching braces patterns are preserved."""
        template = (
            "This {123} and {!var} should not be replaced but {valid_var} should."
        )
        inputs: dict[str, str | int | float | dict[str, Any] | list[Any]] = {
            "valid_var": "works",
        }

        result = interpolate_only(template, inputs)

        assert (
            result == "This {123} and {!var} should not be replaced but works should."
        )

    def test_complex_mixed_scenario(self) -> None:
        """Test a complex scenario with both valid variables and JSON structures."""
        template = """
        {agent_name} is working on task {task_id}.

        Instructions:
        1. Process the data
        2. Return results as:

        {
          "taskId": "{task_id}",
          "results": {
            "processed_by": "agent_name",
            "status": "complete",
            "values": [1, 2, 3]
          }
        }
        """
        inputs: dict[str, str | int | float | dict[str, Any] | list[Any]] = {
            "agent_name": "AnalyticsAgent",
            "task_id": "T-12345",
        }

        result = interpolate_only(template, inputs)

        assert "AnalyticsAgent is working on task T-12345" in result
        assert '"taskId": "T-12345"' in result
        assert '"processed_by": "agent_name"' in result  # This shouldn't be replaced
        assert '"values": [1, 2, 3]' in result

    def test_empty_inputs_dictionary(self) -> None:
        """Test that an error is raised with empty inputs dictionary."""
        template = "Hello, {name}!"
        inputs: dict[str, Any] = {}

        with pytest.raises(ValueError) as excinfo:
            interpolate_only(template, inputs)

        assert "inputs dictionary cannot be empty" in str(excinfo.value).lower()
