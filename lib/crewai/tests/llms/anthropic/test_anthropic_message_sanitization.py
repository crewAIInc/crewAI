"""Tests for Anthropic message sanitization.

Validates fixes for:
- Issue #4413: Trailing whitespace in final assistant message
- Issue #4427: Empty user message content
"""

import os
from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def mock_anthropic_api_key():
    """Ensure ANTHROPIC_API_KEY is set for all tests."""
    if "ANTHROPIC_API_KEY" not in os.environ:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            yield
    else:
        yield


def _get_sanitize_fn():
    """Import and return the static sanitization method."""
    from crewai.llms.providers.anthropic.completion import AnthropicCompletion

    return AnthropicCompletion._sanitize_messages_for_anthropic


class TestTrailingWhitespaceStripping:
    """Test that trailing whitespace is stripped from the final assistant message.

    Anthropic rejects requests where the final assistant message content ends
    with trailing whitespace (spaces, tabs, newlines).

    Regression test for: https://github.com/crewAIInc/crewAI/issues/4413
    """

    def test_trailing_space_in_final_assistant_message(self):
        sanitize = _get_sanitize_fn()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Say: "},
        ]
        result = sanitize(messages)
        assert result[-1]["content"] == "Say:"

    def test_trailing_newline_in_final_assistant_message(self):
        sanitize = _get_sanitize_fn()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Response\n\n"},
        ]
        result = sanitize(messages)
        assert result[-1]["content"] == "Response"

    def test_trailing_tabs_in_final_assistant_message(self):
        sanitize = _get_sanitize_fn()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Result\t\t"},
        ]
        result = sanitize(messages)
        assert result[-1]["content"] == "Result"

    def test_no_stripping_when_final_message_is_user(self):
        sanitize = _get_sanitize_fn()
        messages = [
            {"role": "user", "content": "Hello "},
        ]
        # User message trailing space should NOT be stripped (only assistant)
        # But it would be replaced by "." since it's whitespace-only? No,
        # "Hello " is not whitespace-only, so it stays.
        result = sanitize(messages)
        assert result[-1]["content"] == "Hello "

    def test_trailing_whitespace_in_structured_content_blocks(self):
        """Test stripping in list-based content (e.g., thinking + text blocks)."""
        sanitize = _get_sanitize_fn()
        messages = [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me think..."},
                    {"type": "text", "text": "Answer: "},
                ],
            },
        ]
        result = sanitize(messages)
        text_block = result[-1]["content"][-1]
        assert text_block["text"] == "Answer:"

    def test_no_stripping_on_non_final_assistant_message(self):
        """Trailing whitespace on non-final assistant messages is left alone."""
        sanitize = _get_sanitize_fn()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Step 1 "},
            {"role": "user", "content": "Continue"},
        ]
        result = sanitize(messages)
        # Non-final assistant message keeps its trailing space
        assert result[1]["content"] == "Step 1 "

    def test_assistant_message_without_trailing_whitespace_unchanged(self):
        sanitize = _get_sanitize_fn()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Clean response"},
        ]
        result = sanitize(messages)
        assert result[-1]["content"] == "Clean response"


class TestEmptyContentReplacement:
    """Test that empty content is replaced with a placeholder.

    Anthropic rejects requests where any message has empty content (except
    the optional final assistant message).

    Regression test for: https://github.com/crewAIInc/crewAI/issues/4427
    """

    def test_empty_string_user_message(self):
        sanitize = _get_sanitize_fn()
        messages = [
            {"role": "user", "content": ""},
        ]
        result = sanitize(messages)
        assert result[0]["content"] == "."

    def test_whitespace_only_user_message(self):
        sanitize = _get_sanitize_fn()
        messages = [
            {"role": "user", "content": "   "},
        ]
        result = sanitize(messages)
        assert result[0]["content"] == "."

    def test_empty_assistant_message_in_middle(self):
        sanitize = _get_sanitize_fn()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "Continue"},
        ]
        result = sanitize(messages)
        assert result[1]["content"] == "."

    def test_non_empty_content_unchanged(self):
        sanitize = _get_sanitize_fn()
        messages = [
            {"role": "user", "content": "Hello world"},
            {"role": "assistant", "content": "Hi there"},
        ]
        result = sanitize(messages)
        assert result[0]["content"] == "Hello world"
        assert result[1]["content"] == "Hi there"

    def test_list_content_not_affected(self):
        """List-type content (tool results, etc.) should not be touched."""
        sanitize = _get_sanitize_fn()
        tool_results = [{"type": "tool_result", "tool_use_id": "123", "content": ""}]
        messages = [
            {"role": "user", "content": tool_results},
        ]
        result = sanitize(messages)
        # List content should pass through unchanged
        assert result[0]["content"] is tool_results


class TestCombinedSanitization:
    """Test that both sanitization rules work together correctly."""

    def test_empty_user_and_trailing_whitespace_assistant(self):
        sanitize = _get_sanitize_fn()
        messages = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": "Response "},
        ]
        result = sanitize(messages)
        assert result[0]["content"] == "."
        assert result[1]["content"] == "Response"

    def test_multiple_messages_mixed(self):
        sanitize = _get_sanitize_fn()
        messages = [
            {"role": "user", "content": "  "},
            {"role": "assistant", "content": "Step 1"},
            {"role": "user", "content": "Good"},
            {"role": "assistant", "content": "Final answer\n"},
        ]
        result = sanitize(messages)
        assert result[0]["content"] == "."
        assert result[1]["content"] == "Step 1"
        assert result[2]["content"] == "Good"
        assert result[3]["content"] == "Final answer"

    def test_empty_messages_list(self):
        sanitize = _get_sanitize_fn()
        result = sanitize([])
        assert result == []
