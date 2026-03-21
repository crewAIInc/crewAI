"""Tests for empty message content handling in AnthropicCompletion.

The Anthropic API rejects messages with empty string content ("") with a
400 error. This can happen in pipelines where one agent's empty output
becomes the next agent's input. These tests verify that empty content is
replaced with a single space before sending to the API.

See: https://github.com/crewAIInc/crewAI/issues/4427
"""

import os
from unittest.mock import patch

import pytest

from crewai.llms.providers.anthropic.completion import AnthropicCompletion


@pytest.fixture(autouse=True)
def mock_anthropic_api_key():
    """Automatically mock ANTHROPIC_API_KEY for all tests."""
    if "ANTHROPIC_API_KEY" not in os.environ:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            yield
    else:
        yield


class TestEmptyContentHandling:
    """Verify that empty message content is replaced before hitting the API."""

    def _get_completion(self) -> AnthropicCompletion:
        return AnthropicCompletion(model="claude-3-5-sonnet-20241022")

    def test_empty_user_message_content_replaced(self):
        """Empty string user content should become a single space."""
        completion = self._get_completion()
        messages = [{"role": "user", "content": ""}]
        formatted, _ = completion._format_messages_for_anthropic(messages)

        user_msgs = [m for m in formatted if m.get("role") == "user"]
        assert len(user_msgs) >= 1
        # The last user message (which came from our input) must not be empty
        assert user_msgs[-1]["content"] != ""
        assert user_msgs[-1]["content"] == " "

    def test_none_user_message_content_replaced(self):
        """None user content should also become a single space."""
        completion = self._get_completion()
        messages = [{"role": "user", "content": None}]
        formatted, _ = completion._format_messages_for_anthropic(messages)

        user_msgs = [m for m in formatted if m.get("role") == "user"]
        assert len(user_msgs) >= 1
        assert user_msgs[-1]["content"] == " "

    def test_nonempty_user_message_unchanged(self):
        """Non-empty user content should pass through unchanged."""
        completion = self._get_completion()
        messages = [{"role": "user", "content": "Hello, world!"}]
        formatted, _ = completion._format_messages_for_anthropic(messages)

        user_msgs = [m for m in formatted if m.get("role") == "user"]
        assert user_msgs[-1]["content"] == "Hello, world!"

    def test_empty_assistant_message_content_replaced(self):
        """Empty string assistant content should become a single space."""
        completion = self._get_completion()
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": "Continue"},
        ]
        formatted, _ = completion._format_messages_for_anthropic(messages)

        assistant_msgs = [m for m in formatted if m.get("role") == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0]["content"] == " "

    def test_pipeline_empty_output_scenario(self):
        """Simulate a pipeline where agent A returns empty output to agent B.

        This is the exact scenario from issue #4427: one agent produces
        empty output that becomes the next agent's user message.
        """
        completion = self._get_completion()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": ""},  # Empty output from previous agent
        ]
        formatted, system_msg = completion._format_messages_for_anthropic(messages)

        assert system_msg == "You are a helpful assistant."
        user_msgs = [m for m in formatted if m.get("role") == "user"]
        assert len(user_msgs) >= 1
        # Must not be empty — Anthropic API would reject ""
        for msg in user_msgs:
            content = msg.get("content")
            if isinstance(content, str):
                assert content != "", (
                    "Empty string content would cause Anthropic API 400 error"
                )

    def test_multiple_empty_messages_all_replaced(self):
        """All empty messages in a conversation should be sanitized."""
        completion = self._get_completion()
        messages = [
            {"role": "user", "content": ""},
            {"role": "assistant", "content": ""},
            {"role": "user", "content": ""},
        ]
        formatted, _ = completion._format_messages_for_anthropic(messages)

        for msg in formatted:
            content = msg.get("content")
            if isinstance(content, str):
                assert content != "", (
                    f"Message with role={msg.get('role')} has empty content"
                )
