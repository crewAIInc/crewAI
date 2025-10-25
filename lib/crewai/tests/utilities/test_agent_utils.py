"""Test agent utility functions."""

import pytest
from unittest.mock import MagicMock, patch

from crewai.agent import Agent
from crewai.utilities.agent_utils import handle_context_length
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededError,
)
from crewai.utilities.i18n import I18N
from crewai.utilities.printer import Printer


def test_handle_context_length_raises_exception_when_respect_context_window_false():
    """Test that handle_context_length raises LLMContextLengthExceededError when respect_context_window is False."""
    # Create mocks for dependencies
    printer = Printer()
    i18n = I18N()

    # Create an agent just for its LLM
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        respect_context_window=False,
    )

    llm = agent.llm

    # Create test messages
    messages = [
        {
            "role": "user",
            "content": "This is a test message that would exceed context length",
        }
    ]

    # Set up test parameters
    respect_context_window = False
    callbacks = []

    with pytest.raises(LLMContextLengthExceededError) as excinfo:
        handle_context_length(
            respect_context_window=respect_context_window,
            printer=printer,
            messages=messages,
            llm=llm,
            callbacks=callbacks,
            i18n=i18n,
        )

    assert "Context length exceeded" in str(excinfo.value)
    assert "user opted not to summarize" in str(excinfo.value)


def test_handle_context_length_summarizes_when_respect_context_window_true():
    """Test that handle_context_length calls summarize_messages when respect_context_window is True."""
    # Create mocks for dependencies
    printer = Printer()
    i18n = I18N()

    # Create an agent just for its LLM
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        respect_context_window=True,
    )

    llm = agent.llm

    # Create test messages
    messages = [
        {
            "role": "user",
            "content": "This is a test message that would exceed context length",
        }
    ]

    # Set up test parameters
    respect_context_window = True
    callbacks = []

    with patch("crewai.utilities.agent_utils.summarize_messages") as mock_summarize:
        handle_context_length(
            respect_context_window=respect_context_window,
            printer=printer,
            messages=messages,
            llm=llm,
            callbacks=callbacks,
            i18n=i18n,
        )

        mock_summarize.assert_called_once_with(
            messages=messages, llm=llm, callbacks=callbacks, i18n=i18n
        )


def test_handle_context_length_does_not_raise_system_exit():
    """Test that handle_context_length does NOT raise SystemExit (regression test for issue #3774)."""
    # Create mocks for dependencies
    printer = Printer()
    i18n = I18N()

    # Create an agent just for its LLM
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        respect_context_window=False,
    )

    llm = agent.llm

    # Create test messages
    messages = [
        {
            "role": "user",
            "content": "This is a test message that would exceed context length",
        }
    ]

    # Set up test parameters
    respect_context_window = False
    callbacks = []

    with pytest.raises(Exception) as excinfo:
        handle_context_length(
            respect_context_window=respect_context_window,
            printer=printer,
            messages=messages,
            llm=llm,
            callbacks=callbacks,
            i18n=i18n,
        )

    assert not isinstance(excinfo.value, SystemExit), (
        "handle_context_length should not raise SystemExit. "
        "It should raise LLMContextLengthExceededError instead."
    )

    assert isinstance(excinfo.value, LLMContextLengthExceededError), (
        f"Expected LLMContextLengthExceededError but got {type(excinfo.value).__name__}"
    )
