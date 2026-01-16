"""Tests for agent_utils module, specifically debug logging for OutputParserError."""

import logging
from unittest.mock import MagicMock

import pytest

from crewai.agents.parser import AgentAction, OutputParserError
from crewai.utilities.agent_utils import handle_output_parser_exception


class TestHandleOutputParserExceptionDebugLogging:
    """Tests for debug logging in handle_output_parser_exception."""

    def test_debug_logging_with_raw_output_and_agent_role(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that debug logging includes raw output and agent role when provided."""
        error = OutputParserError("Invalid Format: I missed the 'Action:' after 'Thought:'.")
        messages: list[dict[str, str]] = []
        raw_output = "Let me think about this... The answer is..."
        agent_role = "Researcher"

        with caplog.at_level(logging.DEBUG):
            result = handle_output_parser_exception(
                e=error,
                messages=messages,
                iterations=0,
                raw_output=raw_output,
                agent_role=agent_role,
            )

        assert isinstance(result, AgentAction)
        assert "Parse failed for agent 'Researcher'" in caplog.text
        assert "Raw output (truncated) for agent 'Researcher'" in caplog.text
        assert "Let me think about this... The answer is..." in caplog.text
        assert "Retry 1 initiated for agent 'Researcher'" in caplog.text

    def test_debug_logging_without_agent_role(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that debug logging works without agent role."""
        error = OutputParserError("Invalid Format: I missed the 'Action:' after 'Thought:'.")
        messages: list[dict[str, str]] = []
        raw_output = "Some raw output"

        with caplog.at_level(logging.DEBUG):
            result = handle_output_parser_exception(
                e=error,
                messages=messages,
                iterations=0,
                raw_output=raw_output,
            )

        assert isinstance(result, AgentAction)
        assert "Parse failed:" in caplog.text
        assert "for agent" not in caplog.text.split("Parse failed:")[1].split("\n")[0]
        assert "Raw output (truncated):" in caplog.text
        assert "Retry 1 initiated" in caplog.text

    def test_debug_logging_without_raw_output(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that debug logging works without raw output."""
        error = OutputParserError("Invalid Format: I missed the 'Action:' after 'Thought:'.")
        messages: list[dict[str, str]] = []

        with caplog.at_level(logging.DEBUG):
            result = handle_output_parser_exception(
                e=error,
                messages=messages,
                iterations=0,
                agent_role="Researcher",
            )

        assert isinstance(result, AgentAction)
        assert "Parse failed for agent 'Researcher'" in caplog.text
        assert "Raw output (truncated)" not in caplog.text
        assert "Retry 1 initiated for agent 'Researcher'" in caplog.text

    def test_debug_logging_truncates_long_raw_output(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that raw output is truncated when longer than 500 characters."""
        error = OutputParserError("Invalid Format")
        messages: list[dict[str, str]] = []
        long_output = "A" * 600

        with caplog.at_level(logging.DEBUG):
            handle_output_parser_exception(
                e=error,
                messages=messages,
                iterations=0,
                raw_output=long_output,
                agent_role="Researcher",
            )

        assert "A" * 500 + "..." in caplog.text
        assert "A" * 600 not in caplog.text

    def test_debug_logging_does_not_truncate_short_raw_output(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that short raw output is not truncated."""
        error = OutputParserError("Invalid Format")
        messages: list[dict[str, str]] = []
        short_output = "Short output"

        with caplog.at_level(logging.DEBUG):
            handle_output_parser_exception(
                e=error,
                messages=messages,
                iterations=0,
                raw_output=short_output,
                agent_role="Researcher",
            )

        assert "Short output" in caplog.text
        assert "..." not in caplog.text.split("Short output")[1].split("\n")[0]

    def test_debug_logging_retry_count_increments(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that retry count is correctly calculated from iterations."""
        error = OutputParserError("Invalid Format")
        messages: list[dict[str, str]] = []

        with caplog.at_level(logging.DEBUG):
            handle_output_parser_exception(
                e=error,
                messages=messages,
                iterations=4,
                raw_output="test",
                agent_role="Researcher",
            )

        assert "Retry 5 initiated" in caplog.text

    def test_debug_logging_escapes_newlines_in_raw_output(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that newlines in raw output are escaped for readability."""
        error = OutputParserError("Invalid Format")
        messages: list[dict[str, str]] = []
        output_with_newlines = "Line 1\nLine 2\nLine 3"

        with caplog.at_level(logging.DEBUG):
            handle_output_parser_exception(
                e=error,
                messages=messages,
                iterations=0,
                raw_output=output_with_newlines,
                agent_role="Researcher",
            )

        assert "Line 1\\nLine 2\\nLine 3" in caplog.text

    def test_debug_logging_extracts_first_line_of_error(self, caplog: pytest.LogCaptureFixture) -> None:
        """Test that only the first line of the error message is logged."""
        error = OutputParserError("First line of error\nSecond line\nThird line")
        messages: list[dict[str, str]] = []

        with caplog.at_level(logging.DEBUG):
            handle_output_parser_exception(
                e=error,
                messages=messages,
                iterations=0,
                agent_role="Researcher",
            )

        assert "First line of error" in caplog.text
        parse_failed_line = [line for line in caplog.text.split("\n") if "Parse failed" in line][0]
        assert "Second line" not in parse_failed_line

    def test_messages_updated_with_error(self) -> None:
        """Test that messages list is updated with the error."""
        error = OutputParserError("Test error message")
        messages: list[dict[str, str]] = []

        handle_output_parser_exception(
            e=error,
            messages=messages,
            iterations=0,
        )

        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Test error message"

    def test_returns_agent_action_with_error_text(self) -> None:
        """Test that the function returns an AgentAction with the error text."""
        error = OutputParserError("Test error message")
        messages: list[dict[str, str]] = []

        result = handle_output_parser_exception(
            e=error,
            messages=messages,
            iterations=0,
        )

        assert isinstance(result, AgentAction)
        assert result.text == "Test error message"
        assert result.tool == ""
        assert result.tool_input == ""
        assert result.thought == ""

    def test_printer_logs_after_log_error_after_iterations(self) -> None:
        """Test that printer logs error after log_error_after iterations."""
        error = OutputParserError("Test error")
        messages: list[dict[str, str]] = []
        printer = MagicMock()

        handle_output_parser_exception(
            e=error,
            messages=messages,
            iterations=4,
            log_error_after=3,
            printer=printer,
        )

        printer.print.assert_called_once()
        call_args = printer.print.call_args
        assert "Error parsing LLM output" in call_args.kwargs["content"]
        assert call_args.kwargs["color"] == "red"

    def test_printer_does_not_log_before_log_error_after_iterations(self) -> None:
        """Test that printer does not log before log_error_after iterations."""
        error = OutputParserError("Test error")
        messages: list[dict[str, str]] = []
        printer = MagicMock()

        handle_output_parser_exception(
            e=error,
            messages=messages,
            iterations=2,
            log_error_after=3,
            printer=printer,
        )

        printer.print.assert_not_called()

    def test_backward_compatibility_without_new_parameters(self) -> None:
        """Test that the function works without the new optional parameters."""
        error = OutputParserError("Test error")
        messages: list[dict[str, str]] = []

        result = handle_output_parser_exception(
            e=error,
            messages=messages,
            iterations=0,
        )

        assert isinstance(result, AgentAction)
        assert len(messages) == 1
