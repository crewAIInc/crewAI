"""Tests for structured tool error formatting."""

import json

import pytest

from crewai.utilities.tool_errors import RETRYABLE_EXCEPTIONS, format_tool_error


class TestFormatToolError:
    """Tests for the format_tool_error utility function."""

    def test_returns_string_with_prefix(self):
        err = ValueError("bad input")
        result = format_tool_error(err)
        assert result.startswith("Error executing tool: ")

    def test_contains_valid_json_after_prefix(self):
        err = ValueError("bad input")
        result = format_tool_error(err)
        json_str = result[len("Error executing tool: "):]
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

    def test_error_flag_is_true(self):
        err = RuntimeError("something broke")
        result = format_tool_error(err)
        parsed = json.loads(result[len("Error executing tool: "):])
        assert parsed["error"] is True

    def test_preserves_exception_type(self):
        err = KeyError("missing_key")
        result = format_tool_error(err)
        parsed = json.loads(result[len("Error executing tool: "):])
        assert parsed["type"] == "KeyError"

    def test_preserves_exception_message(self):
        err = ValueError("count must be positive")
        result = format_tool_error(err)
        parsed = json.loads(result[len("Error executing tool: "):])
        assert parsed["message"] == "count must be positive"

    def test_retryable_true_for_timeout(self):
        err = TimeoutError("connection timed out")
        result = format_tool_error(err)
        parsed = json.loads(result[len("Error executing tool: "):])
        assert parsed["retryable"] is True

    def test_retryable_true_for_connection_error(self):
        err = ConnectionError("refused")
        result = format_tool_error(err)
        parsed = json.loads(result[len("Error executing tool: "):])
        assert parsed["retryable"] is True

    def test_retryable_true_for_os_error(self):
        err = OSError("disk full")
        result = format_tool_error(err)
        parsed = json.loads(result[len("Error executing tool: "):])
        assert parsed["retryable"] is True

    def test_retryable_false_for_value_error(self):
        err = ValueError("invalid")
        result = format_tool_error(err)
        parsed = json.loads(result[len("Error executing tool: "):])
        assert parsed["retryable"] is False

    def test_retryable_false_for_type_error(self):
        err = TypeError("wrong type")
        result = format_tool_error(err)
        parsed = json.loads(result[len("Error executing tool: "):])
        assert parsed["retryable"] is False

    def test_retryable_false_for_key_error(self):
        err = KeyError("not found")
        result = format_tool_error(err)
        parsed = json.loads(result[len("Error executing tool: "):])
        assert parsed["retryable"] is False

    def test_no_traceback_by_default(self):
        err = ValueError("test")
        result = format_tool_error(err)
        parsed = json.loads(result[len("Error executing tool: "):])
        assert "traceback" not in parsed

    def test_traceback_included_when_requested(self):
        try:
            raise ValueError("deliberate error")
        except ValueError as e:
            result = format_tool_error(e, include_traceback=True)
        parsed = json.loads(result[len("Error executing tool: "):])
        assert "traceback" in parsed
        assert "ValueError" in parsed["traceback"]

    def test_handles_exception_with_special_characters(self):
        err = ValueError('path "C:\\Users\\test" not found')
        result = format_tool_error(err)
        parsed = json.loads(result[len("Error executing tool: "):])
        assert 'C:\\Users\\test' in parsed["message"]

    def test_handles_exception_with_empty_message(self):
        err = RuntimeError()
        result = format_tool_error(err)
        parsed = json.loads(result[len("Error executing tool: "):])
        assert parsed["type"] == "RuntimeError"
        assert parsed["message"] == ""

    def test_handles_custom_exception(self):
        class MyToolError(Exception):
            pass

        err = MyToolError("custom failure")
        result = format_tool_error(err)
        parsed = json.loads(result[len("Error executing tool: "):])
        assert parsed["type"] == "MyToolError"
        assert parsed["message"] == "custom failure"
        assert parsed["retryable"] is False

    def test_retryable_exceptions_tuple_contains_expected_types(self):
        assert TimeoutError in RETRYABLE_EXCEPTIONS
        assert ConnectionError in RETRYABLE_EXCEPTIONS
        assert OSError in RETRYABLE_EXCEPTIONS
