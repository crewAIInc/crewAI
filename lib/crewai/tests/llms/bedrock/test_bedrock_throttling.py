import asyncio
import os
import sys
from unittest.mock import MagicMock, patch
from botocore.exceptions import ClientError, EventStreamError
from pydantic import ValidationError
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
    """Helper to construct a botocore ClientError with specific code and message."""
    return ClientError(
        {"Error": {"Code": code, "Message": message}},
        "Converse",
    )


class TestBedrockThrottlingClassification:
    """Unit tests for Bedrock throttling vs context-length error classification."""

    def test_client_error_throttling_code(self):
        """Test detection of ClientError with ThrottlingException code."""
        err = _make_client_error("ThrottlingException", "Rate exceeded")
        assert is_bedrock_throttling_error(err) is True
        assert is_context_length_exceeded(err) is False

    def test_client_error_too_many_tokens_throttling_message(self):
        """Test detection of ClientError with 'Too many tokens, please wait' message."""
        err = _make_client_error(
            "ThrottlingException",
            "Too many tokens, please wait before trying again",
        )
        assert is_bedrock_throttling_error(err) is True
        assert is_context_length_exceeded(err) is False

    def test_client_error_too_many_requests_code(self):
        """Test detection of ClientError with TooManyRequestsException code."""
        err = _make_client_error("TooManyRequestsException", "Request rate limit exceeded")
        assert is_bedrock_throttling_error(err) is True
        assert is_context_length_exceeded(err) is False

    def test_generic_exception_too_many_tokens_retryable_message(self):
        """Test detection of generic Exception containing rate limit message."""
        err = Exception("Too many tokens, please wait before trying again")
        assert is_bedrock_throttling_error(err) is True
        assert is_context_length_exceeded(err) is False

    def test_generic_exception_throttling_string(self):
        """Test detection of generic Exception with ThrottlingException substring."""
        err = Exception(
            "An error occurred (ThrottlingException) when calling the Converse operation: Too many tokens, please wait before trying again"
        )
        assert is_bedrock_throttling_error(err) is True
        assert is_context_length_exceeded(err) is False

    def test_wrapped_exception_cause_inspection(self):
        """Test inspection of __cause__ chain for structured ClientError codes with neutral messages."""
        cause = _make_client_error("TooManyRequestsException", "Neutral error message")
        wrapper = RuntimeError("Bedrock completion failed: operation error")
        wrapper.__cause__ = cause
        assert is_bedrock_throttling_error(wrapper) is True
        assert is_context_length_exceeded(wrapper) is False

    def test_wrapped_model_not_ready_exception_cause_inspection(self):
        """Test inspection of __cause__ chain for ModelNotReadyException with neutral messages."""
        cause = _make_client_error("ModelNotReadyException", "Internal condition")
        wrapper = RuntimeError("Bedrock completion failed")
        wrapper.__cause__ = cause
        assert is_bedrock_throttling_error(wrapper) is True
        assert is_context_length_exceeded(wrapper) is False

    def test_wrapped_exception_context_inspection(self):
        """Test inspection of __context__ chain for structured ClientError codes."""
        context_err = _make_client_error("RequestLimitExceeded", "Rate limit")
        wrapper = RuntimeError("Wrapped error")
        wrapper.__context__ = context_err
        assert is_bedrock_throttling_error(wrapper) is True
        assert is_context_length_exceeded(wrapper) is False

    def test_event_stream_error_throttling_detection(self):
        """Test detection of botocore EventStreamError with throttlingException code."""
        err = EventStreamError(
            {"Error": {"Code": "throttlingException", "Message": "Rate limit exceeded"}},
            "converse_stream",
        )
        assert is_bedrock_throttling_error(err) is True
        assert is_context_length_exceeded(err) is False

    def test_genuine_context_length_errors_are_not_throttling(self):
        """Test genuine context length errors are not classified as throttling."""
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
        """Test unrelated errors are neither throttling nor context length errors."""
        err = ValueError("Invalid message format")
        assert is_bedrock_throttling_error(err) is False
        assert is_context_length_exceeded(err) is False


class TestBedrockRetryValidation:
    """Unit tests for BedrockCompletion retry configuration validation."""

    def test_negative_max_retries_rejected(self):
        """Test BedrockCompletion rejects negative max_retries."""
        with pytest.raises(ValidationError):
            BedrockCompletion(
                model="anthropic.claude-3-5-sonnet-20241022-v2:0",
                max_retries=-1,
            )

    def test_negative_retry_delay_rejected(self):
        """Test BedrockCompletion rejects negative retry_delay."""
        with pytest.raises(ValidationError):
            BedrockCompletion(
                model="anthropic.claude-3-5-sonnet-20241022-v2:0",
                retry_delay=-0.5,
            )

    def test_negative_max_retry_delay_rejected(self):
        """Test BedrockCompletion rejects negative max_retry_delay."""
        with pytest.raises(ValidationError):
            BedrockCompletion(
                model="anthropic.claude-3-5-sonnet-20241022-v2:0",
                max_retry_delay=-1.0,
            )


class TestBedrockBackoffJitter:
    """Unit tests for Bedrock backoff delay and jitter calculation."""

    def test_calculate_backoff_delay_applies_bounded_jitter(self):
        """Test backoff delay incorporates jitter bounded between 0.5 * base and base."""
        llm = BedrockCompletion(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            retry_delay=1.0,
            max_retry_delay=30.0,
        )
        delays = [llm._calculate_backoff_delay(attempt=1) for _ in range(20)]
        for delay in delays:
            assert 1.0 <= delay <= 2.0

    def test_calculate_backoff_delay_caps_at_max_retry_delay(self):
        """Test backoff delay never exceeds max_retry_delay even on high attempts."""
        llm = BedrockCompletion(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            retry_delay=2.0,
            max_retry_delay=5.0,
        )
        for attempt in range(10):
            delay = llm._calculate_backoff_delay(attempt)
            assert delay <= 5.0

    def test_calculate_backoff_delay_zero_delay(self):
        """Test backoff delay is 0.0 when retry_delay is 0.0."""
        llm = BedrockCompletion(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            retry_delay=0.0,
        )
        assert llm._calculate_backoff_delay(0) == 0.0


class TestBedrockThrottlingRetry:
    """Unit tests for Bedrock retry with exponential backoff on throttling."""

    @patch("time.sleep")
    def test_sync_call_retries_on_throttling_and_succeeds(self, mock_sleep):
        """Test sync call retries on throttling and succeeds on subsequent attempt."""
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
        assert mock_sleep.call_count == 1
        assert 0.05 <= mock_sleep.call_args[0][0] <= 0.1

    @patch("time.sleep")
    def test_sync_call_exhausts_retries_and_does_not_trigger_context_error(
        self, mock_sleep
    ):
        """Test sync call exhausts retries on throttling without raising context length error."""
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
        """Test sync call does not retry genuine context length errors."""
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
        """Test async call retry logic on throttling."""
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
        assert mock_async_sleep.call_count == 1
        assert 0.05 <= mock_async_sleep.call_args[0][0] <= 0.1

    @patch("time.sleep")
    def test_streaming_retries_before_chunk_emitted(self, mock_sleep):
        """Test streaming retries when throttling occurs before any chunk is emitted."""
        llm = BedrockCompletion(
            model="anthropic.claude-3-5-sonnet-20241022-v2:0",
            max_retries=2,
            retry_delay=0.1,
            stream=True,
        )
        mock_client = MagicMock()
        throttle_err = _make_client_error(
            "ThrottlingException",
            "Too many tokens, please wait before trying again",
        )
        success_stream = [
            {"contentBlockDelta": {"delta": {"text": "Hello world"}}},
            {"messageStop": {"stopReason": "end_turn"}},
        ]
        success_response = {
            "stream": success_stream,
            "stopReason": "end_turn",
        }
        mock_client.converse_stream.side_effect = [throttle_err, success_response]
        llm._client = mock_client

        result = llm.call(messages=[{"role": "user", "content": "Hi"}])
        assert result == "Hello world"
        assert mock_client.converse_stream.call_count == 2
        assert mock_sleep.call_count == 1
