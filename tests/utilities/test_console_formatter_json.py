"""Tests for ConsoleFormatter JSON formatting functionality."""

import json
from unittest.mock import patch

from crewai.events.utils.console_formatter import ConsoleFormatter


class TestConsoleFormatterJsonMethods:
    """Tests for the _format_tool_input_json method in ConsoleFormatter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.formatter = ConsoleFormatter()

    def test_empty_input_returns_empty_json(self):
        """Test that empty input returns empty JSON object."""
        result = self.formatter._format_tool_input_json({})
        assert result == "{}"

    def test_simple_dict_pretty_formatting(self):
        """Test pretty formatting of simple dictionary."""
        tool_input = {"query": "test search", "limit": 10}
        result = self.formatter._format_tool_input_json(tool_input, pretty=True)

        # Should have 4-space indentation (pretty=True)
        assert "    " in result
        # Should sort keys
        assert result.find("limit") < result.find("query")
        # Should have clean separators
        assert ": " in result

    def test_simple_dict_standard_formatting(self):
        """Test standard formatting of simple dictionary."""
        tool_input = {"query": "test search", "limit": 10}
        result = self.formatter._format_tool_input_json(tool_input, pretty=False)

        # Should have 2-space indentation (pretty=False)
        lines = result.split("\n")
        indented_lines = [
            line for line in lines if line.strip() and line.startswith("  ")
        ]
        if indented_lines:
            # Check that indented lines use 2 spaces, not 4
            assert indented_lines[0].startswith("  ")
            assert not indented_lines[0].startswith("    ")

    def test_complex_nested_dict_formatting(self):
        """Test formatting of complex nested dictionary."""
        tool_input = {
            "search_params": {
                "query": "AI research",
                "filters": {"date": "2024", "category": "tech"},
            },
            "options": {"max_results": 50, "sort_by": "relevance"},
        }

        result = self.formatter._format_tool_input_json(tool_input, pretty=True)

        # Should contain all nested elements
        assert "search_params" in result
        assert "AI research" in result
        assert "filters" in result
        assert "max_results" in result

        # Should be properly formatted JSON
        parsed = json.loads(result)
        assert parsed["search_params"]["query"] == "AI research"
        assert parsed["options"]["max_results"] == 50

    def test_keys_sorted_for_consistency(self):
        """Test that keys are sorted for consistent output."""
        tool_input = {"zebra": 1, "alpha": 2, "beta": 3}
        result = self.formatter._format_tool_input_json(tool_input)

        # Should be in alphabetical order
        alpha_pos = result.find("alpha")
        beta_pos = result.find("beta")
        zebra_pos = result.find("zebra")

        assert alpha_pos < beta_pos < zebra_pos

    def test_ensure_ascii_false_preserves_unicode(self):
        """Test that unicode characters are preserved."""
        tool_input = {"message": "Hello 世界! Café naïve"}
        result = self.formatter._format_tool_input_json(tool_input)

        # Unicode should be preserved, not escaped
        assert "世界" in result
        assert "Café" in result
        assert "naïve" in result
        # Should not have escaped unicode sequences
        assert "\\u" not in result

    def test_clean_separators_in_pretty_mode(self):
        """Test that pretty mode uses clean separators."""
        tool_input = {"key1": "value1", "key2": "value2"}
        result = self.formatter._format_tool_input_json(tool_input, pretty=True)

        # Should use clean separators: comma-space for items, colon-space for key-value
        assert ": " in result  # colon-space for key-value pairs
        # Check that it's not using compact separators like {"key1":"value1","key2":"value2"}
        lines = result.split("\n")
        key_value_lines = [line for line in lines if ": " in line]
        assert (
            len(key_value_lines) >= 2
        )  # Should have properly formatted key-value pairs

    def test_error_handling_non_serializable_objects(self):
        """Test error handling for non-serializable objects."""

        class NonSerializable:
            def __repr__(self):
                return "NonSerializable()"

        tool_input = {"data": NonSerializable()}
        result = self.formatter._format_tool_input_json(tool_input)

        assert "Error formatting tool input" in result
        # Should be a string, not raise an exception
        assert isinstance(result, str)

    def test_type_error_handling(self):
        """Test handling of TypeError during serialization."""
        # Create a mock that raises TypeError when json.dumps is called
        with patch("json.dumps", side_effect=TypeError("Test type error")):
            result = self.formatter._format_tool_input_json({"test": "value"})

            assert "Error formatting tool input: Test type error" in result

    def test_value_error_handling(self):
        """Test handling of ValueError during serialization."""
        # Create a mock that raises ValueError when json.dumps is called
        with patch("json.dumps", side_effect=ValueError("Test value error")):
            result = self.formatter._format_tool_input_json({"test": "value"})

            assert "Error formatting tool input: Test value error" in result

    def test_different_indentation_levels(self):
        """Test that pretty and standard modes use different indentation."""
        tool_input = {"nested": {"inner": "value"}}

        pretty_result = self.formatter._format_tool_input_json(tool_input, pretty=True)
        standard_result = self.formatter._format_tool_input_json(
            tool_input, pretty=False
        )

        # Pretty should have more indentation (4 spaces vs 2)
        pretty_lines = pretty_result.split("\n")
        standard_lines = standard_result.split("\n")

        # Find lines with indentation
        pretty_indented = [line for line in pretty_lines if line.startswith("    ")]
        standard_indented = [
            line
            for line in standard_lines
            if line.startswith("  ") and not line.startswith("    ")
        ]

        # Pretty mode should have 4-space indented lines
        assert len(pretty_indented) > 0
        # Standard mode should have 2-space indented lines
        assert len(standard_indented) > 0

    def test_mixed_data_types_formatting(self):
        """Test formatting with mixed data types."""
        tool_input = {
            "string_val": "hello",
            "int_val": 42,
            "float_val": 3.14159,
            "bool_val": True,
            "null_val": None,
            "list_val": [1, "two", 3.0, False],
            "dict_val": {"nested": "value"},
        }

        result = self.formatter._format_tool_input_json(tool_input)

        # Should be valid JSON
        parsed = json.loads(result)

        # All values should be correctly represented
        assert parsed["string_val"] == "hello"
        assert parsed["int_val"] == 42
        assert parsed["float_val"] == 3.14159
        assert parsed["bool_val"] is True
        assert parsed["null_val"] is None
        assert parsed["list_val"] == [1, "two", 3.0, False]
        assert parsed["dict_val"]["nested"] == "value"

    def test_large_data_structure_handling(self):
        """Test handling of large data structures."""
        # Create a reasonably large structure
        large_input = {
            f"section_{i}": {
                "items": [f"item_{j}" for j in range(20)],
                "metadata": {f"field_{k}": f"value_{k}" for k in range(10)},
            }
            for i in range(5)
        }

        result = self.formatter._format_tool_input_json(large_input)

        # Should complete without error and return valid JSON
        assert isinstance(result, str)
        assert len(result) > 0

        # Should be parseable
        parsed = json.loads(result)
        assert len(parsed) == 5
        assert "section_0" in parsed
        assert len(parsed["section_0"]["items"]) == 20

    def test_special_characters_handling(self):
        """Test proper handling of special JSON characters."""
        tool_input = {
            "quotes": 'Text with "quoted" content',
            "backslashes": "path\\to\\file",
            "newlines": "line1\nline2\nline3",
            "tabs": "column1\tcolumn2",
            "unicode": "Emoji 🔧 and unicode 世界",
        }

        result = self.formatter._format_tool_input_json(tool_input)

        # Should be valid JSON
        parsed = json.loads(result)

        # Special characters should be properly handled
        assert '"quoted"' in parsed["quotes"]
        assert "path\\to\\file" == parsed["backslashes"]
        assert "line1\nline2\nline3" == parsed["newlines"]
        assert "column1\tcolumn2" == parsed["tabs"]
        assert "🔧" in parsed["unicode"]
        assert "世界" in parsed["unicode"]
