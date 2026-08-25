"""Truncation detection shared across providers.

A response cut off by the token cap is otherwise indistinguishable from a complete
one, so callers cannot tell "the model was interrupted" from "the model answered
badly". These tests pin the behaviour rather than the wording.
"""

import logging

import pytest

from crewai.llms._finish_reason_utils import is_truncated, warn_if_truncated


@pytest.mark.parametrize(
    "finish_reason",
    [
        "length",  # OpenAI, Azure
        "max_tokens",  # Anthropic, Bedrock
        "MAX_TOKENS",  # Gemini
        "maxTokens",
        "max-tokens",
        "  Length  ",
    ],
)
def test_recognises_every_provider_spelling(finish_reason):
    assert is_truncated(finish_reason) is True


@pytest.mark.parametrize(
    "finish_reason",
    ["stop", "STOP", "tool_calls", "content_filter", "end_turn", "", None, 42, object()],
)
def test_ignores_non_truncation_reasons(finish_reason):
    assert is_truncated(finish_reason) is False


def test_warns_once_and_names_the_budget(caplog):
    with caplog.at_level(logging.WARNING):
        emitted = warn_if_truncated("length", max_tokens=16, model="gpt-4o-mini")

    assert emitted is True
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    # The reader needs to know which knob to turn, so the cap has to appear.
    assert "16" in message
    assert "gpt-4o-mini" in message


def test_silent_on_a_complete_response(caplog):
    with caplog.at_level(logging.WARNING):
        emitted = warn_if_truncated("stop", max_tokens=16, model="gpt-4o-mini")

    assert emitted is False
    assert caplog.records == []


def test_warns_without_optional_context(caplog):
    """max_tokens is often unset, and a warning is still better than silence."""
    with caplog.at_level(logging.WARNING):
        emitted = warn_if_truncated("max_tokens")

    assert emitted is True
    assert len(caplog.records) == 1
