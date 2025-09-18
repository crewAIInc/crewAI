"""Tests for agent utility functions, particularly JSON formatting functionality."""

from unittest.mock import Mock

from crewai.agents.parser import AgentAction, AgentFinish
from crewai.utilities import Printer
from crewai.utilities.agent_utils import (
    _apply_json_colors,
    _format_tool_input_json,
    show_agent_logs,
)


class TestFormatToolInputJson:
    """Tests for the _format_tool_input_json function."""

    def test_empty_input(self):
        """Test that empty input returns colored empty dict."""
        result = _format_tool_input_json({})
        assert result == "\033[93m{}\033[00m"  # Yellow empty dict

    def test_simple_dict_formatting(self):
        """Test basic dictionary formatting with improved readability."""
        tool_input = {"query": "test search", "limit": 10}
        result = _format_tool_input_json(tool_input)

        # Check that content is present
        assert "query" in result
        assert "test search" in result
        assert "limit" in result
        assert "10" in result

        # Check for color codes (cyan keys)
        assert "\033[96m" in result  # Cyan color for keys

    def test_complex_nested_dict(self):
        """Test formatting of complex nested dictionaries."""
        tool_input = {
            "search_params": {
                "query": "AI research",
                "filters": {"date": "2024", "category": "tech"},
            },
            "options": {"max_results": 50, "sort_by": "relevance"},
        }

        result = _format_tool_input_json(tool_input)

        # Check formatting elements
        assert "search_params" in result
        assert "AI research" in result
        assert "max_results" in result

        # Check for color codes
        assert "\033[96m" in result  # Cyan for keys
        assert "\033[92m" in result  # Green for string values

    def test_sorted_keys(self):
        """Test that keys are sorted for consistency."""
        tool_input = {"zebra": 1, "alpha": 2, "beta": 3}
        result = _format_tool_input_json(tool_input)

        # Find positions of keys in the result
        alpha_pos = result.find("alpha")
        beta_pos = result.find("beta")
        zebra_pos = result.find("zebra")

        # Should be in alphabetical order
        assert alpha_pos < beta_pos < zebra_pos

    def test_non_serializable_object_error_handling(self):
        """Test error handling for non-serializable objects."""

        # Create a non-serializable object
        class NonSerializable:
            pass

        tool_input = {"data": NonSerializable()}
        result = _format_tool_input_json(tool_input)

        assert "Error formatting tool input" in result
        assert "\033[91m" in result  # Red color for error

    def test_special_characters_handling(self):
        """Test proper handling of special characters and unicode."""
        tool_input = {
            "text": 'Hello 世界! Special chars: \n\t"quotes"',
            "emoji": "🔧🎯📝",
            "numbers": [1.5, -10, 3.14159],
        }

        result = _format_tool_input_json(tool_input)

        # Should preserve unicode and special characters
        assert "世界" in result
        assert "🔧🎯📝" in result
        assert "3.14159" in result

    def test_ensure_ascii_false(self):
        """Test that ensure_ascii=False is properly used."""
        tool_input = {"unicode_text": "Café naïve résumé"}
        result = _format_tool_input_json(tool_input)

        # Unicode characters should be preserved, not escaped
        assert "Café" in result
        assert "naïve" in result
        assert "résumé" in result
        # Should not have escaped unicode like \u00e9
        assert "\\u00" not in result


class TestApplyJsonColors:
    """Tests for the _apply_json_colors function."""

    def test_key_coloring(self):
        """Test that JSON keys are properly colored."""
        json_str = '{"name": "test", "value": 42}'
        result = _apply_json_colors(json_str)

        # Keys should be bright cyan (\033[96m)
        assert '\033[96m"name"\033[00m' in result
        assert '\033[96m"value"\033[00m' in result

    def test_string_value_coloring(self):
        """Test that string values are properly colored."""
        json_str = '{"name": "test"}'
        result = _apply_json_colors(json_str)

        # String values should be green (\033[92m)
        assert '\033[92m"test"\033[00m' in result

    def test_number_coloring(self):
        """Test that numbers are properly colored."""
        json_str = '{"count": 42, "price": 19.99}'
        result = _apply_json_colors(json_str)

        # Numbers should be yellow (\033[93m)
        assert "\033[93m42\033[00m" in result
        assert "\033[93m19.99\033[00m" in result

    def test_boolean_and_null_coloring(self):
        """Test that booleans and null values are properly colored."""
        json_str = '{"active": true, "deleted": false, "data": null}'
        result = _apply_json_colors(json_str)

        # Booleans and null should be magenta (\033[95m)
        assert "\033[95mtrue\033[00m" in result
        assert "\033[95mfalse\033[00m" in result
        assert "\033[95mnull\033[00m" in result

    def test_structural_characters_coloring(self):
        """Test that brackets and braces are properly colored."""
        json_str = '{"items": [1, 2, 3]}'
        result = _apply_json_colors(json_str)

        # Structural characters should be white (\033[37m)
        assert "\033[37m{\033[00m" in result
        assert "\033[37m}\033[00m" in result
        assert "\033[37m[\033[00m" in result
        assert "\033[37m]\033[00m" in result

    def test_no_indentation_guides(self):
        """Test that the basic color formatting works without complex guides."""
        json_str = """{\n    "nested": {\n        "value": 42\n    }\n}"""
        result = _apply_json_colors(json_str)

        # Should have colored keys and values but no complex guides
        assert "\033[96m" in result  # Cyan keys
        assert "nested" in result


class TestShowAgentLogsIntegration:
    """Integration tests for show_agent_logs with improved JSON formatting."""

    def setup_method(self):
        """Set up test fixtures."""
        self.printer = Mock(spec=Printer)
        self.agent_role = "Test Agent"

    def test_agent_action_with_json_formatting(self):
        """Test that AgentAction logs use improved JSON formatting."""
        tool_input = {"search_query": "AI tools", "max_results": 5}
        agent_action = Mock(spec=AgentAction)
        agent_action.thought = "I need to search for AI tools"
        agent_action.tool = "search_tool"
        agent_action.tool_input = tool_input
        agent_action.result = "Found 5 results"

        show_agent_logs(
            printer=self.printer,
            agent_role=self.agent_role,
            formatted_answer=agent_action,
            verbose=True,
        )

        # Check that printer.print was called with formatted JSON
        calls = self.printer.print.call_args_list

        # Should have calls for thought, tool, input, and output
        assert len(calls) >= 4

        # Find the JSON input call
        json_calls = [
            call for call in calls if "search_query" in str(call) or "{" in str(call)
        ]
        assert len(json_calls) >= 1

        # The JSON should be properly formatted (contains the formatted result)
        json_call_content = str(json_calls[0])
        assert "search_query" in json_call_content

    def test_agent_action_with_empty_tool_input(self):
        """Test handling of empty tool input."""
        agent_action = Mock(spec=AgentAction)
        agent_action.thought = "Simple action"
        agent_action.tool = "simple_tool"
        agent_action.tool_input = {}
        agent_action.result = "Done"

        show_agent_logs(
            printer=self.printer,
            agent_role=self.agent_role,
            formatted_answer=agent_action,
            verbose=True,
        )

        # Should handle empty input gracefully
        calls = self.printer.print.call_args_list
        assert len(calls) >= 4  # Should still make all the expected calls

    def test_agent_finish_unchanged(self):
        """Test that AgentFinish behavior is unchanged."""
        agent_finish = Mock(spec=AgentFinish)
        agent_finish.output = "Task completed successfully"

        show_agent_logs(
            printer=self.printer,
            agent_role=self.agent_role,
            formatted_answer=agent_finish,
            verbose=True,
        )

        # Should have a call with the final answer
        calls = self.printer.print.call_args_list
        final_call = [
            call for call in calls if "Task completed successfully" in str(call)
        ]
        assert len(final_call) == 1

    def test_verbose_false_no_output(self):
        """Test that no output occurs when verbose=False."""
        agent_action = Mock(spec=AgentAction)
        agent_action.tool_input = {"test": "data"}

        show_agent_logs(
            printer=self.printer,
            agent_role=self.agent_role,
            formatted_answer=agent_action,
            verbose=False,
        )

        # Should not call printer when verbose=False
        self.printer.print.assert_not_called()

    def test_start_logs_with_task_description(self):
        """Test start logs with task description."""
        task_description = "Research AI trends in 2024"

        show_agent_logs(
            printer=self.printer,
            agent_role=self.agent_role,
            task_description=task_description,
            verbose=True,
        )

        # Should have calls for agent and task
        calls = self.printer.print.call_args_list
        assert len(calls) >= 2

        # Check that task description is included
        task_calls = [call for call in calls if task_description in str(call)]
        assert len(task_calls) == 1


class TestJsonFormattingEdgeCases:
    """Tests for edge cases in JSON formatting."""

    def test_very_large_json_structure(self):
        """Test handling of large JSON structures."""
        # Create a large nested structure
        large_input = {
            f"key_{i}": {"nested": [f"item_{j}" for j in range(10)]} for i in range(10)
        }

        result = _format_tool_input_json(large_input)

        # Should still format properly without errors
        assert isinstance(result, str)
        assert len(result) > 0
        assert "{" in result
        assert "key_0" in result

    def test_deeply_nested_structure(self):
        """Test deeply nested JSON structures."""
        nested_input = {
            "level1": {"level2": {"level3": {"level4": {"deep_value": "found"}}}}
        }

        result = _format_tool_input_json(nested_input)

        # Should handle deep nesting
        assert "deep_value" in result
        assert "found" in result
        assert result.count("{") >= 5  # Multiple opening braces for nesting

    def test_mixed_data_types(self):
        """Test handling of mixed data types."""
        mixed_input = {
            "string": "text",
            "integer": 42,
            "float": 3.14159,
            "boolean": True,
            "null_value": None,
            "list": [1, "two", 3.0, True, None],
            "nested_dict": {"inner": "value"},
        }

        result = _format_tool_input_json(mixed_input)

        # All data types should be represented
        assert "text" in result
        assert "42" in result
        assert "3.14159" in result
        assert "true" in result
        assert "null" in result
        assert "two" in result

    def test_special_json_characters(self):
        """Test handling of special JSON characters that need escaping."""
        special_input = {
            "quotes": 'He said "Hello"',
            "backslash": "path\\to\\file",
            "newlines": "line1\nline2\nline3",
            "tabs": "col1\tcol2\tcol3",
        }

        result = _format_tool_input_json(special_input)

        # Should properly escape special characters
        assert '\\"' in result  # Escaped quotes
        assert "\\\\" in result  # Escaped backslashes
        assert "\\n" in result  # Escaped newlines
        assert "\\t" in result  # Escaped tabs
