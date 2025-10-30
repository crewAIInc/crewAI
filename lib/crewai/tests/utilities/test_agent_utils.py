import pytest
from unittest.mock import MagicMock
from src.crewai.utilities.agent_utils import is_null_response_because_context_length_exceeded

def test_is_null_response_because_context_length_exceeded_true():
    """
    Test that the function returns True when the exception is a ValueError
    with 'None or empty' and there are messages.
    """
    # Arrange
    mock_llm = MagicMock()
    mock_llm.get_context_window_size.return_value = 10
    exception = ValueError("Invalid response from LLM call - None or empty.")
    messages = [{"content": "This is a test message."}]

    # Act
    result = is_null_response_because_context_length_exceeded(exception, messages, mock_llm)

    # Assert
    assert result is True


def test_is_null_response_because_context_length_exceeded_false_wrong_exception():
    """
    Test that the function returns False when the exception is not a ValueError.
    """
    # Arrange
    mock_llm = MagicMock()
    mock_llm.get_context_window_size.return_value = 10
    exception = TypeError("Some other error.")
    messages = [{"content": "This is a test message."}]

    # Act
    result = is_null_response_because_context_length_exceeded(exception, messages, mock_llm)

    # Assert
    assert result is False


def test_is_null_response_because_context_length_exceeded_false_wrong_message():
    """
    Test that the function returns False when the exception message does not
    contain 'None or empty'.
    """
    # Arrange
    mock_llm = MagicMock()
    mock_llm.get_context_window_size.return_value = 10
    exception = ValueError("Another value error.")
    messages = [{"content": "This is a test message."}]

    # Act
    result = is_null_response_because_context_length_exceeded(exception, messages, mock_llm)

    # Assert
    assert result is False


def test_is_null_response_because_context_length_exceeded_false_empty_messages():
    """
    Test that the function returns False when the messages list is empty.
    """
    # Arrange
    mock_llm = MagicMock()
    mock_llm.get_context_window_size.return_value = 10
    exception = ValueError("Invalid response from LLM call - None or empty.")
    messages = []

    # Act
    result = is_null_response_because_context_length_exceeded(exception, messages, mock_llm)

    # Assert
    assert result is False


def test_is_null_response_because_context_length_exceeded_false():
    """
    Test that the function returns True when the exception is a ValueError
    with 'None or empty' and there are messages.
    """
    # Arrange
    mock_llm = MagicMock()
    mock_llm.get_context_window_size.return_value = 50
    exception = ValueError("Invalid response from LLM call - None or empty.")
    messages = [{"content": "This is a test message."}]

    # Act
    result = is_null_response_because_context_length_exceeded(exception, messages, mock_llm)

    # Assert
    assert result is False


