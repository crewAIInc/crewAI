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
    """Every vendor spells truncation differently and all of them must be caught.

    A provider whose spelling is missed looks identical to a complete response,
    which is the failure this helper exists to prevent.
    """
    assert is_truncated(finish_reason) is True


@pytest.mark.parametrize(
    "finish_reason",
    ["stop", "STOP", "tool_calls", "content_filter", "end_turn", "", None, 42, object()],
)
def test_ignores_non_truncation_reasons(finish_reason):
    """Normal completions and non-string values must not raise a false alarm.

    A warning on every healthy turn is worse than no warning at all, because
    readers learn to ignore it.
    """
    assert is_truncated(finish_reason) is False


def test_warns_once_and_names_the_budget(caplog):
    """One warning per truncated response, naming the cap that caused it.

    The cap has to appear or the reader cannot tell which setting to raise.
    """
    with caplog.at_level(logging.WARNING):
        emitted = warn_if_truncated("length", max_tokens=16, model="gpt-4o-mini")

    assert emitted is True
    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    # The reader needs to know which knob to turn, so the cap has to appear.
    assert "16" in message
    assert "gpt-4o-mini" in message


def test_silent_on_a_complete_response(caplog):
    """A complete response must log nothing at all."""
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


def test_warning_names_whichever_cap_the_provider_actually_sent():
    """Providers send different cap parameters, so the message must not assume one.

    Gemini sends ``max_output_tokens`` and OpenAI/Azure reasoning models send
    ``max_completion_tokens``. Naming the wrong setting sends the reader to a
    knob that will not change anything.
    """
    import logging

    from crewai.llms._finish_reason_utils import warn_if_truncated

    for cap, model in [(2048, "gemini-2.5-flash"), (4096, "o3-mini"), (16, "claude-haiku-4-5")]:
        records = []
        handler = logging.Handler()
        handler.emit = records.append  # type: ignore[method-assign]
        logging.getLogger().addHandler(handler)
        try:
            assert warn_if_truncated("length", cap, model) is True
        finally:
            logging.getLogger().removeHandler(handler)
        assert str(cap) in records[0].getMessage()
        assert model in records[0].getMessage()
