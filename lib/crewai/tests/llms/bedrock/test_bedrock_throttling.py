import asyncio
import os
import sys
from unittest.mock import MagicMock, patch
from botocore.exceptions import ClientError
import pytest

from crewai.llms.providers.bedrock.completion import BedrockCompletion
from crewai.llms.providers.bedrock.throttling import is_bedrock_throttling_error
from crewai.utilities.agent_utils import is_context_length_exceeded
from crewai.utilities.exceptions.context_window_exceeding_exception import (
    LLMContextLengthExceededError,
)


@pytest.fixture(autouse=True)
def mock_aws_credentials():
    """Mock AWS credentials and boto3 Session for tests."""
    try:
        from pytest_recording.network import unblock_socket

        unblock_socket()
    except (ImportError, AttributeError):
        pass

    with patch.dict(
        os.environ,
        {
            "AWS_ACCESS_KEY_ID": "test-access-key",
            "AWS_SECRET_ACCESS_KEY": "test-secret-key",
            "AWS_DEFAULT_REGION": "us-east-1",
        },
    ):
        with patch("crewai.llms.providers.bedrock.completion.Session") as mock_session_class:
            mock_session_instance = MagicMock()
            mock_client = MagicMock()
            mock_session_instance.client.return_value = mock_client
            mock_session_class.return_value = mock_session_instance
            yield mock_session_class, mock_client


def _make_client_error(code: str, message: str) -> ClientError:
    return ClientError(
        {"Error": {"Code": code, "Message": message}},
        "Converse",
    )


class TestBedrockThrottlingClassification:
    """Unit tests for Bedrock throttling vs context-length error classification."""

    def test_client_error_throttling_code(self):
        err = _make_client_error("ThrottlingException", "Rate exceeded")
        assert is_bedrock_throttling_error(err) is True
        assert is_context_length_exceeded(err) is False

    def test_client_error_too_many_tokens_throttling_message(self):
        err = _make_client_error(
            "ThrottlingException",
            "Too many tokens, please wait before trying again",
        )
        assert is_bedrock_throttling_error(err) is True
        assert is_context_length_exceeded(err) is False

    def test_client_error_too_many_requests_code(self):
        err = _make_client_error("TooManyRequestsException", "Request rate limit exceeded")
        assert is_bedrock_throttling_error(err) is True
        assert is_context_length_exceeded(err) is False

    def test_generic_exception_too_many_tokens_retryable_message(self):
        err = Exception("Too many tokens, please wait before trying again")
        assert is_bedrock_throttling_error(err) is True
        assert is_context_length_exceeded(err) is False

    def test_generic_exception_throttling_string(self):
        err = Exception(
            "An error occurred (ThrottlingException) when calling the Converse operation: Too many tokens, please wait before trying again"
        )
        assert is_bedrock_throttling_error(err) is True
        assert is_context_length_exceeded(err) is False

    def test_genuine_context_length_errors_are_not_throttling(self):
        errors = [
            Exception(
                "ValidationException: This model's maximum context length is 200000 tokens. "
                "However, your request resulted in 250000 tokens."
            ),
            Exception("Input is too long for requested model"),
            Exception("maximum context length exceeded"),
            Exception("context window full"),
            Exception("exceeds token limit"),
        ]
        for err in errors:
            assert is_bedrock_throttling_error(err) is False, f"Expected False for {err}"
            assert is_context_length_exceeded(err) is True, f"Expected True for {err}"

    def test_unrelated_errors_are_neither(self):
        err = ValueError("Invalid message format")
        assert is_bedrock_throttling_error(err) is False
        assert is_context_length_exceeded(err) is False


class TestBedrockThrottlingRetry:
    """Unit tests for Bedrock retry with exponential backoff on throttling."""

    @patch("time.sleep")
    def test_sync_call_retries_on_throttling_and_succeeds(self, mock_sleep):
        llm = BedrockCompletion(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            max_retries=2,
            retry_delay=0.1,
        )
        mock_client = MagicMock()
        throttle_err = _make_client_error(
            "ThrottlingException",
            "Too many tokens, please wait before trying again",
        )
        success_response = {
            "output": {"message": {"role": "assistant", "content": [{"text": "Hello"}]}},
            "usage": {"inputTokens": 10, "outputTokens": 5, "totalTokens": 15},
        }
        mock_client.converse.side_effect = [throttle_err, success_response]
        llm._client = mock_client

        result = llm.call(messages=[{"role": "user", "content": "Hi"}])
        assert result == "Hello"
        assert mock_client.converse.call_count == 2
        mock_sleep.assert_called_once_with(0.1)

    @patch("time.sleep")
    def test_sync_call_exhausts_retries_and_does_not_trigger_context_error(
        self, mock_sleep
    ):
        llm = BedrockCompletion(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            max_retries=2,
            retry_delay=0.1,
        )
        mock_client = MagicMock()
        throttle_err = _make_client_error(
            "ThrottlingException",
            "Too many tokens, please wait before trying again",
        )
        mock_client.converse.side_effect = throttle_err
        llm._client = mock_client

        with pytest.raises(Exception) as exc_info:
            llm.call(messages=[{"role": "user", "content": "Hi"}])

        assert not issubclass(exc_info.type, LLMContextLengthExceededError)
        assert is_bedrock_throttling_error(exc_info.value) is True
        # 1 initial + 2 retries = 3 calls
        assert mock_client.converse.call_count == 3
        assert mock_sleep.call_count == 2

    @patch("time.sleep")
    def test_sync_call_does_not_retry_genuine_context_length_error(
        self, mock_sleep
    ):
        llm = BedrockCompletion(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            max_retries=2,
            retry_delay=0.1,
        )
        mock_client = MagicMock()
        context_err = Exception("maximum context length exceeded")
        mock_client.converse.side_effect = context_err
        llm._client = mock_client

        with pytest.raises(LLMContextLengthExceededError):
            llm.call(messages=[{"role": "user", "content": "Hi"}])

        assert mock_client.converse.call_count == 1
        mock_sleep.assert_not_called()

    @patch("asyncio.sleep")
    def test_async_call_retry_logic(self, mock_async_sleep):
        llm = BedrockCompletion(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            max_retries=2,
            retry_delay=0.1,
        )
        throttle_err = _make_client_error(
            "ThrottlingException",
            "Too many tokens, please wait before trying again",
        )
        calls = 0

        async def _test_coro():
            nonlocal calls
            calls += 1
            if calls == 1:
                raise throttle_err
            return "async success"

        async def _run():
            return await llm._acall_with_retry(_test_coro)

        try:
            from pytest_recording.network import unblock_socket

            unblock_socket()
        except (ImportError, AttributeError):
            pass

        result = asyncio.run(_run())
        assert result == "async success"
        assert calls == 2
        mock_async_sleep.assert_called_once_with(0.1)
