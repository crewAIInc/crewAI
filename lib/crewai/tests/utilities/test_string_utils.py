import re
from typing import Any, Dict, List, Union

import pytest
from crewai.tools import BaseTool
from crewai.utilities.agent_utils import convert_tools_to_openai_schema
from crewai.utilities.string_utils import interpolate_only, sanitize_tool_name


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
        assert '"processed_by": "agent_name"' in result
        assert '"values": [1, 2, 3]' in result

    def test_empty_inputs_dictionary(self):
        """Test that an error is raised with empty inputs dictionary."""
        template = "Hello, {name}!"
        inputs: Dict[str, Any] = {}

        with pytest.raises(ValueError) as excinfo:
            interpolate_only(template, inputs)

        assert "inputs dictionary cannot be empty" in str(excinfo.value).lower()


class ChineseSearchTool(BaseTool):
    """Tool whose name has no ASCII letters or digits."""

    name: str = "搜索工具"
    description: str = "搜索网络获取信息"

    def _run(self, query: str) -> str:
        return f"results for {query}"


class ChineseFetchTool(BaseTool):
    """Second non-ASCII-named tool, to exercise collision handling."""

    name: str = "抓取网页"
    description: str = "抓取网页内容"

    def _run(self, query: str) -> str:
        return f"page {query}"


class TestSanitizeToolName:
    """Tests for the sanitize_tool_name empty-result fallback."""

    def test_non_ascii_only_name_never_sanitizes_to_empty(self):
        """A name with no ASCII letters/digits still yields a provider-valid name."""
        result = sanitize_tool_name("搜索工具")

        assert result
        assert re.fullmatch(r"[a-z0-9_]{1,64}", result)

    def test_punctuation_only_name_never_sanitizes_to_empty(self):
        """A name erased entirely by the cleaning pipeline gets a fallback."""
        result = sanitize_tool_name("!!!")

        assert result
        assert re.fullmatch(r"[a-z0-9_]{1,64}", result)

    def test_distinct_non_ascii_names_get_distinct_results(self):
        """Two different erased names must not collapse onto one fallback."""
        assert sanitize_tool_name("搜索工具") != sanitize_tool_name("抓取网页")

    def test_fallback_is_deterministic(self):
        """The same input always produces the same fallback name."""
        first = sanitize_tool_name("搜索工具")
        sanitize_tool_name("抓取网页")
        second = sanitize_tool_name("搜索工具")

        assert first == second

    def test_ascii_and_empty_names_are_unchanged(self):
        """Names that already sanitize cleanly keep their existing output."""
        assert sanitize_tool_name("My Tool") == "my_tool"
        assert sanitize_tool_name("searchTool") == "search_tool"
        assert sanitize_tool_name("") == ""

    def test_openai_schema_has_no_empty_function_names(self):
        """Function-calling schemas never carry an empty or duplicated name."""
        tools: list[BaseTool] = [ChineseSearchTool(), ChineseFetchTool()]

        schemas, available_functions, _ = convert_tools_to_openai_schema(tools)

        names = [schema["function"]["name"] for schema in schemas]
        assert all(names)
        assert len(set(names)) == len(tools)
        assert set(names) == set(available_functions)
