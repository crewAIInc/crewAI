from unittest.mock import MagicMock, patch

import pytest

from crewai.llm import LLM


def test_llm_call_with_litellm_1_66_3():
    """Test that the LLM class works with litellm v1.66.3+"""
    llm = LLM(
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=50,
        stop=["STOP"],
        presence_penalty=0.1,
        frequency_penalty=0.1,
    )
    messages = [{"role": "user", "content": "Say 'Hello, World!' and then say STOP"}]

    with patch("litellm.completion") as mocked_completion:
        mock_message = MagicMock()
        mock_message.content = "Hello, World! I won't say the stop word."
        mock_choice = MagicMock()
        mock_choice.message = mock_message
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = {
            "prompt_tokens": 10,
            "completion_tokens": 10,
            "total_tokens": 20,
        }

        mocked_completion.return_value = mock_response

        response = llm.call(messages)

        mocked_completion.assert_called_once()

        assert "Hello, World!" in response
        assert "STOP" not in response

        _, kwargs = mocked_completion.call_args
        assert kwargs["model"] == "gpt-3.5-turbo"
        assert kwargs["temperature"] == 0.7
        assert kwargs["max_tokens"] == 50
        assert kwargs["stop"] == ["STOP"]
        assert kwargs["presence_penalty"] == 0.1
        assert kwargs["frequency_penalty"] == 0.1
