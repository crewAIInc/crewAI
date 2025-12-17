"""Tests for agent_utils module.

These tests cover the format_answer() and handle_max_iterations_exceeded() functions,
specifically testing the fix for issue #4113 where OutputParserError was being
swallowed instead of being re-raised for retry logic.
"""

from unittest.mock import MagicMock, patch

import pytest

from crewai.agents.parser import (
    AgentAction,
    AgentFinish,
    OutputParserError,
)
from crewai.utilities.agent_utils import (
    format_answer,
    handle_max_iterations_exceeded,
    process_llm_response,
)


class TestFormatAnswer:
    """Tests for the format_answer function."""

    def test_format_answer_with_valid_action(self) -> None:
        """Test that format_answer correctly parses a valid action."""
        answer = "Thought: Let's search\nAction: search\nAction Input: query"
        result = format_answer(answer)
        assert isinstance(result, AgentAction)
        assert result.tool == "search"
        assert result.tool_input == "query"

    def test_format_answer_with_valid_final_answer(self) -> None:
        """Test that format_answer correctly parses a valid final answer."""
        answer = "Thought: I found the answer\nFinal Answer: The result is 42"
        result = format_answer(answer)
        assert isinstance(result, AgentFinish)
        assert result.output == "The result is 42"

    def test_format_answer_raises_output_parser_error_for_malformed_output(
        self,
    ) -> None:
        """Test that format_answer re-raises OutputParserError for malformed output.

        This is the core fix for issue #4113. Previously, format_answer would catch
        all exceptions and return AgentFinish, which broke the retry logic.
        """
        malformed_answer = """Thought
The user wants to verify something.
Action
Video Analysis Tool
Action Input:
{"query": "Is there something?"}"""

        with pytest.raises(OutputParserError):
            format_answer(malformed_answer)

    def test_format_answer_raises_output_parser_error_missing_action(self) -> None:
        """Test that format_answer re-raises OutputParserError when Action is missing."""
        answer = "Thought: Let's search\nAction Input: query"
        with pytest.raises(OutputParserError) as exc_info:
            format_answer(answer)
        assert "Action:" in str(exc_info.value)

    def test_format_answer_raises_output_parser_error_missing_action_input(
        self,
    ) -> None:
        """Test that format_answer re-raises OutputParserError when Action Input is missing."""
        answer = "Thought: Let's search\nAction: search"
        with pytest.raises(OutputParserError) as exc_info:
            format_answer(answer)
        assert "Action Input:" in str(exc_info.value)

    def test_format_answer_returns_agent_finish_for_generic_exception(self) -> None:
        """Test that format_answer returns AgentFinish for non-OutputParserError exceptions."""
        with patch(
            "crewai.utilities.agent_utils.parse",
            side_effect=ValueError("Unexpected error"),
        ):
            result = format_answer("some answer")
            assert isinstance(result, AgentFinish)
            assert result.thought == "Failed to parse LLM response"
            assert result.output == "some answer"


class TestProcessLlmResponse:
    """Tests for the process_llm_response function."""

    def test_process_llm_response_raises_output_parser_error(self) -> None:
        """Test that process_llm_response propagates OutputParserError."""
        malformed_answer = "Thought\nMissing colons\nAction\nSome Tool"
        with pytest.raises(OutputParserError):
            process_llm_response(malformed_answer, use_stop_words=True)

    def test_process_llm_response_with_valid_action(self) -> None:
        """Test that process_llm_response correctly processes a valid action."""
        answer = "Thought: Let's search\nAction: search\nAction Input: query"
        result = process_llm_response(answer, use_stop_words=True)
        assert isinstance(result, AgentAction)
        assert result.tool == "search"

    def test_process_llm_response_with_valid_final_answer(self) -> None:
        """Test that process_llm_response correctly processes a valid final answer."""
        answer = "Thought: Done\nFinal Answer: The result"
        result = process_llm_response(answer, use_stop_words=True)
        assert isinstance(result, AgentFinish)
        assert result.output == "The result"


class TestHandleMaxIterationsExceeded:
    """Tests for the handle_max_iterations_exceeded function."""

    def test_handle_max_iterations_exceeded_with_valid_final_answer(self) -> None:
        """Test that handle_max_iterations_exceeded returns AgentFinish for valid output."""
        mock_llm = MagicMock()
        mock_llm.call.return_value = "Thought: Done\nFinal Answer: The final result"
        mock_printer = MagicMock()
        mock_i18n = MagicMock()
        mock_i18n.errors.return_value = "Please provide final answer"

        result = handle_max_iterations_exceeded(
            formatted_answer=None,
            printer=mock_printer,
            i18n=mock_i18n,
            messages=[],
            llm=mock_llm,
            callbacks=[],
        )

        assert isinstance(result, AgentFinish)
        assert result.output == "The final result"

    def test_handle_max_iterations_exceeded_with_valid_action_converts_to_finish(
        self,
    ) -> None:
        """Test that handle_max_iterations_exceeded converts AgentAction to AgentFinish."""
        mock_llm = MagicMock()
        mock_llm.call.return_value = (
            "Thought: Using tool\nAction: search\nAction Input: query"
        )
        mock_printer = MagicMock()
        mock_i18n = MagicMock()
        mock_i18n.errors.return_value = "Please provide final answer"

        result = handle_max_iterations_exceeded(
            formatted_answer=None,
            printer=mock_printer,
            i18n=mock_i18n,
            messages=[],
            llm=mock_llm,
            callbacks=[],
        )

        assert isinstance(result, AgentFinish)

    def test_handle_max_iterations_exceeded_catches_output_parser_error(self) -> None:
        """Test that handle_max_iterations_exceeded catches OutputParserError and returns AgentFinish.

        This prevents infinite loops when the forced final answer is malformed.
        Without this safeguard, the OutputParserError would bubble up to _invoke_loop(),
        which would retry, hit max iterations again, and loop forever.
        """
        malformed_response = """Thought
Missing colons everywhere
Action
Some Tool
Action Input:
{"query": "test"}"""

        mock_llm = MagicMock()
        mock_llm.call.return_value = malformed_response
        mock_printer = MagicMock()
        mock_i18n = MagicMock()
        mock_i18n.errors.return_value = "Please provide final answer"

        result = handle_max_iterations_exceeded(
            formatted_answer=None,
            printer=mock_printer,
            i18n=mock_i18n,
            messages=[],
            llm=mock_llm,
            callbacks=[],
        )

        assert isinstance(result, AgentFinish)
        assert result.output == malformed_response
        assert "Failed to parse LLM response during max iterations" in result.thought
        mock_printer.print.assert_any_call(
            content="Failed to parse forced final answer. Returning raw response.",
            color="yellow",
        )

    def test_handle_max_iterations_exceeded_with_previous_formatted_answer(
        self,
    ) -> None:
        """Test that handle_max_iterations_exceeded uses previous answer text."""
        mock_llm = MagicMock()
        mock_llm.call.return_value = "Thought: Done\nFinal Answer: New result"
        mock_printer = MagicMock()
        mock_i18n = MagicMock()
        mock_i18n.errors.return_value = "Please provide final answer"

        previous_answer = AgentAction(
            thought="Previous thought",
            tool="search",
            tool_input="query",
            text="Previous text",
        )

        result = handle_max_iterations_exceeded(
            formatted_answer=previous_answer,
            printer=mock_printer,
            i18n=mock_i18n,
            messages=[],
            llm=mock_llm,
            callbacks=[],
        )

        assert isinstance(result, AgentFinish)
        assert result.output == "New result"

    def test_handle_max_iterations_exceeded_raises_on_empty_response(self) -> None:
        """Test that handle_max_iterations_exceeded raises ValueError for empty response."""
        mock_llm = MagicMock()
        mock_llm.call.return_value = ""
        mock_printer = MagicMock()
        mock_i18n = MagicMock()
        mock_i18n.errors.return_value = "Please provide final answer"

        with pytest.raises(ValueError, match="Invalid response from LLM call"):
            handle_max_iterations_exceeded(
                formatted_answer=None,
                printer=mock_printer,
                i18n=mock_i18n,
                messages=[],
                llm=mock_llm,
                callbacks=[],
            )


class TestRetryLogicIntegration:
    """Integration tests to verify the retry logic works correctly with the fix."""

    def test_malformed_output_allows_retry_in_format_answer(self) -> None:
        """Test that malformed output raises OutputParserError which can be caught for retry.

        This simulates what happens in _invoke_loop() when the LLM returns malformed output.
        The OutputParserError should be raised so the loop can catch it and retry.
        """
        malformed_outputs = [
            "Thought\nMissing colon after Thought",
            "Thought: OK\nAction\nMissing colon after Action",
            "Thought: OK\nAction: tool\nAction Input\nMissing colon",
            "Random text without any structure",
        ]

        for malformed in malformed_outputs:
            with pytest.raises(OutputParserError):
                format_answer(malformed)

    def test_valid_output_does_not_raise(self) -> None:
        """Test that valid outputs are parsed correctly without raising."""
        valid_outputs = [
            ("Thought: Let's search\nAction: search\nAction Input: query", AgentAction),
            ("Thought: Done\nFinal Answer: The result", AgentFinish),
        ]

        for output, expected_type in valid_outputs:
            result = format_answer(output)
            assert isinstance(result, expected_type)
