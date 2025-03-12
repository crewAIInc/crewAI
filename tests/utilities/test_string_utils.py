import unittest
from typing import Any, Dict, List, Union

import pytest

from crewai.utilities import is_ipv4_pattern, sanitize_collection_name
from crewai.utilities.string_utils import (
    MAX_LENGTH,
    MIN_LENGTH,
    interpolate_only,
)


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

        assert "inputs dictionary cannot be empty" in str(excinfo.value).lower()


class TestStringUtils(unittest.TestCase):
    def test_sanitize_collection_name_long_name(self):
        """Test sanitizing a very long collection name."""
        long_name = "This is an extremely long role name that will definitely exceed the ChromaDB collection name limit of 63 characters and cause an error when used as a collection name"
        sanitized = sanitize_collection_name(long_name)
        self.assertLessEqual(len(sanitized), MAX_LENGTH)
        self.assertTrue(sanitized[0].isalnum())
        self.assertTrue(sanitized[-1].isalnum())
        self.assertTrue(all(c.isalnum() or c in ["_", "-"] for c in sanitized))

    def test_sanitize_collection_name_special_chars(self):
        """Test sanitizing a name with special characters."""
        special_chars = "Agent@123!#$%^&*()"
        sanitized = sanitize_collection_name(special_chars)
        self.assertTrue(sanitized[0].isalnum())
        self.assertTrue(sanitized[-1].isalnum())
        self.assertTrue(all(c.isalnum() or c in ["_", "-"] for c in sanitized))

    def test_sanitize_collection_name_short_name(self):
        """Test sanitizing a very short name."""
        short_name = "A"
        sanitized = sanitize_collection_name(short_name)
        self.assertGreaterEqual(len(sanitized), MIN_LENGTH)
        self.assertTrue(sanitized[0].isalnum())
        self.assertTrue(sanitized[-1].isalnum())

    def test_sanitize_collection_name_bad_ends(self):
        """Test sanitizing a name with non-alphanumeric start/end."""
        bad_ends = "_Agent_"
        sanitized = sanitize_collection_name(bad_ends)
        self.assertTrue(sanitized[0].isalnum())
        self.assertTrue(sanitized[-1].isalnum())

    def test_sanitize_collection_name_none(self):
        """Test sanitizing a None value."""
        sanitized = sanitize_collection_name(None)
        self.assertEqual(sanitized, "default_collection")

    def test_sanitize_collection_name_ipv4_pattern(self):
        """Test sanitizing an IPv4 address."""
        ipv4 = "192.168.1.1"
        sanitized = sanitize_collection_name(ipv4)
        self.assertTrue(sanitized.startswith("ip_"))
        self.assertTrue(sanitized[0].isalnum())
        self.assertTrue(sanitized[-1].isalnum())
        self.assertTrue(all(c.isalnum() or c in ["_", "-"] for c in sanitized))

    def test_is_ipv4_pattern(self):
        """Test IPv4 pattern detection."""
        self.assertTrue(is_ipv4_pattern("192.168.1.1"))
        self.assertFalse(is_ipv4_pattern("not.an.ip.address"))

    def test_sanitize_collection_name_properties(self):
        """Test that sanitized collection names always meet ChromaDB requirements."""
        test_cases = [
            "A" * 100,  # Very long name
            "_start_with_underscore",
            "end_with_underscore_",
            "contains@special#characters",
            "192.168.1.1",  # IPv4 address
            "a" * 2,  # Too short
        ]
        for test_case in test_cases:
            sanitized = sanitize_collection_name(test_case)
            self.assertGreaterEqual(len(sanitized), MIN_LENGTH)
            self.assertLessEqual(len(sanitized), MAX_LENGTH)
            self.assertTrue(sanitized[0].isalnum())
            self.assertTrue(sanitized[-1].isalnum())


if __name__ == "__main__":
    unittest.main()
