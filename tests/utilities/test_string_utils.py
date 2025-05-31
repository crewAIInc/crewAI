from typing import Any, Dict, List, Union

import pytest
import pandas as pd

from crewai.utilities.string_utils import interpolate_only


class TestInterpolateOnly:
    """Tests for the interpolate_only function in string_utils.py."""

    def test_basic_variable_interpolation(self):
        """Test basic variable interpolation works correctly."""
        template = "Hello, {name}! Welcome to {company}."
        inputs: Dict[str, Union[str, int, float, Dict[str, Any], List[Any]]] = {
            "name": "Alice",
            "company": "CrewAI",
        }

        result = interpolate_only(template, inputs)

        assert result == "Hello, Alice! Welcome to CrewAI."

    def test_multiple_occurrences_of_same_variable(self):
        """Test that multiple occurrences of the same variable are replaced."""
        template = "{name} is using {name}'s account."
        inputs: Dict[str, Union[str, int, float, Dict[str, Any], List[Any]]] = {
            "name": "Bob"
        }

        result = interpolate_only(template, inputs)

        assert result == "Bob is using Bob's account."

    def test_json_structure_preservation(self):
        """Test that JSON structures are preserved and not interpolated incorrectly."""
        template = """
        Instructions for {agent}:

        Please return the following object:

        {"name": "person's name", "age": 25, "skills": ["coding", "testing"]}
        """
        inputs: Dict[str, Union[str, int, float, Dict[str, Any], List[Any]]] = {
            "agent": "DevAgent"
        }

        result = interpolate_only(template, inputs)

        assert "Instructions for DevAgent:" in result
        assert (
            '{"name": "person\'s name", "age": 25, "skills": ["coding", "testing"]}'
            in result
        )

    def test_complex_nested_json(self):
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
        inputs: Dict[str, Union[str, int, float, Dict[str, Any], List[Any]]] = {
            "agent": "DataProcessor"
        }

        result = interpolate_only(template, inputs)

        assert "DataProcessor needs to process:" in result
        assert '"nested": {' in result
        assert '"value": 42' in result
        assert '[1, 2, {"inner": "value"}]' in result

    def test_missing_variable(self):
        """Test that an error is raised when a required variable is missing."""
        template = "Hello, {name}!"
        inputs: Dict[str, Union[str, int, float, Dict[str, Any], List[Any]]] = {
            "not_name": "Alice"
        }

        with pytest.raises(KeyError) as excinfo:
            interpolate_only(template, inputs)

        assert "template variable" in str(excinfo.value).lower()
        assert "name" in str(excinfo.value)

    def test_invalid_input_types(self):
        """Test that an error is raised with invalid input types."""
        template = "Hello, {name}!"
        # Using Any for this test since we're intentionally testing an invalid type
        inputs: Dict[str, Any] = {"name": object()}  # Object is not a valid input type

        with pytest.raises(ValueError) as excinfo:
            interpolate_only(template, inputs)

        assert "unsupported type" in str(excinfo.value).lower()

    def test_empty_input_string(self):
        """Test handling of empty or None input string."""
        inputs: Dict[str, Union[str, int, float, Dict[str, Any], List[Any]]] = {
            "name": "Alice"
        }

        assert interpolate_only("", inputs) == ""
        assert interpolate_only(None, inputs) == ""

    def test_no_variables_in_template(self):
        """Test a template with no variables to replace."""
        template = "This is a static string with no variables."
        inputs: Dict[str, Union[str, int, float, Dict[str, Any], List[Any]]] = {
            "name": "Alice"
        }

        result = interpolate_only(template, inputs)

        assert result == template

    def test_variable_name_starting_with_underscore(self):
        """Test variables starting with underscore are replaced correctly."""
        template = "Variable: {_special_var}"
        inputs: Dict[str, Union[str, int, float, Dict[str, Any], List[Any]]] = {
            "_special_var": "Special Value"
        }

        result = interpolate_only(template, inputs)

        assert result == "Variable: Special Value"

    def test_preserves_non_matching_braces(self):
        """Test that non-matching braces patterns are preserved."""
        template = (
            "This {123} and {!var} should not be replaced but {valid_var} should."
        )
        inputs: Dict[str, Union[str, int, float, Dict[str, Any], List[Any]]] = {
            "valid_var": "works"
        }

        result = interpolate_only(template, inputs)

        assert (
            result == "This {123} and {!var} should not be replaced but works should."
        )

    def test_complex_mixed_scenario(self):
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
        inputs: Dict[str, Union[str, int, float, Dict[str, Any], List[Any]]] = {
            "agent_name": "AnalyticsAgent",
            "task_id": "T-12345",
        }

        result = interpolate_only(template, inputs)

        assert "AnalyticsAgent is working on task T-12345" in result
        assert '"taskId": "T-12345"' in result
        assert '"processed_by": "agent_name"' in result  # This shouldn't be replaced
        assert '"values": [1, 2, 3]' in result

    def test_empty_inputs_dictionary(self):
        """Test that an error is raised with empty inputs dictionary."""
        template = "Hello, {name}!"
        inputs: Dict[str, Any] = {}

        with pytest.raises(ValueError) as excinfo:
            interpolate_only(template, inputs)


    def test_interpolate_only_with_dataframe(self):
        """Test that interpolate_only handles pandas DataFrames correctly."""
        df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
        
        result = interpolate_only("Data: {data}", {"data": df})
        
        assert "Alice" in result
        assert "Bob" in result
        assert "25" in result
        assert "30" in result

    def test_interpolate_only_mixed_types_with_dataframe(self):
        """Test interpolate_only with mixed input types including DataFrame."""
        df = pd.DataFrame({"col": [1, 2, 3]})
        
        inputs = {
            "text": "hello",
            "number": 42,
            "flag": True,
            "data": df,
            "items": [1, 2, 3]
        }
        
        template = "Text: {text}, Number: {number}, Flag: {flag}, Data: {data}, Items: {items}"
        result = interpolate_only(template, inputs)
        
        assert "hello" in result
        assert "42" in result
        assert "True" in result
        assert "col" in result
        assert "[1, 2, 3]" in result

    def test_interpolate_only_unsupported_type_error(self):
        """Test that interpolate_only handles unsupported types gracefully."""
        class CustomObject:
            def __str__(self):
                raise Exception("Cannot serialize")
        
        with pytest.raises(ValueError, match="Unable to serialize CustomObject"):
            interpolate_only("Value: {obj}", {"obj": CustomObject()})

    def test_interpolate_only_complex_dataframe(self):
        """Test interpolate_only with more complex DataFrame structures."""
        df = pd.DataFrame({
            "product": ["Widget A", "Widget B", "Widget C"],
            "sales": [100, 150, 200],
            "region": ["North", "South", "East"]
        })
        
        result = interpolate_only("Sales report: {report}", {"report": df})
        
        assert "Widget A" in result
        assert "100" in result
        assert "North" in result
        assert "sales" in result
        assert "product" in result

    def test_interpolate_only_backward_compatibility(self):
        """Test that existing supported types still work correctly."""
        inputs = {
            "text": "hello",
            "number": 42,
            "float_val": 3.14,
            "flag": True,
            "nested": {"key": "value"},
            "items": [1, 2, 3]
        }
        
        template = "Text: {text}, Number: {number}, Float: {float_val}, Flag: {flag}, Nested: {nested}, Items: {items}"
        result = interpolate_only(template, inputs)
        
        assert "hello" in result
        assert "42" in result
        assert "3.14" in result
        assert "True" in result
        assert "key" in result
        assert "[1, 2, 3]" in result
