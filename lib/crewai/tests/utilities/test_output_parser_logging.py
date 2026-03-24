"""Tests for debug logging on OutputParser error retries.

Verifies that handle_output_parser_exception emits a logger.debug() message
containing the iteration number and error details, regardless of verbose setting.
Addresses GitHub Issue #4246.
"""

import logging

from crewai.agents.parser import OutputParserError
from crewai.utilities.agent_utils import handle_output_parser_exception


def test_handle_output_parser_exception_emits_debug_log(caplog):
    """Verify debug log is emitted on OutputParserError retry."""
    error = OutputParserError(error="Invalid format: missing Final Answer")

    with caplog.at_level(logging.DEBUG, logger="crewai.utilities.agent_utils"):
        result = handle_output_parser_exception(
            e=error,
            messages=[],
            iterations=2,
        )

    assert any(
        "OutputParserError on iteration 2" in record.message
        for record in caplog.records
    ), f"Expected debug log not found in: {[r.message for r in caplog.records]}"

    # Should still return an AgentAction
    assert result.tool == ""
    assert "Invalid format" in result.text


def test_debug_log_includes_error_details(caplog):
    """Verify debug log includes the actual error message."""
    error_msg = "I just used the Read File tool"
    error = OutputParserError(error=error_msg)

    with caplog.at_level(logging.DEBUG, logger="crewai.utilities.agent_utils"):
        handle_output_parser_exception(
            e=error,
            messages=[],
            iterations=5,
        )

    debug_records = [
        r for r in caplog.records
        if r.levelno == logging.DEBUG and "OutputParserError" in r.message
    ]
    assert len(debug_records) == 1
    assert error_msg in debug_records[0].message
    assert "iteration 5" in debug_records[0].message


def test_debug_log_emitted_even_when_not_verbose(caplog):
    """Debug log should be emitted regardless of verbose setting."""
    error = OutputParserError(error="Bad format")

    with caplog.at_level(logging.DEBUG, logger="crewai.utilities.agent_utils"):
        handle_output_parser_exception(
            e=error,
            messages=[],
            iterations=1,
            verbose=False,
        )

    assert any(
        "OutputParserError on iteration 1" in record.message
        for record in caplog.records
    )


def test_messages_appended_on_parser_error():
    """Verify the error is appended to the messages list."""
    messages = []
    error = OutputParserError(error="Parse failure")

    handle_output_parser_exception(
        e=error,
        messages=messages,
        iterations=0,
    )

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Parse failure"
