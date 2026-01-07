"""Tests for LiteLLM token usage tracking functionality.

These tests verify that token usage metrics are properly tracked for:
- Non-streaming responses
- Async non-streaming responses
- Async streaming responses

This addresses GitHub issue #4170 where token usage metrics were not being
updated when using litellm with streaming responses and async calls.
"""

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai.llm import LLM


class MockUsage:
    """Mock usage object that mimics litellm's usage response."""

    def __init__(
        self,
        prompt_tokens: int = 10,
        completion_tokens: int = 20,
        total_tokens: int = 30,
    ):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = total_tokens


class MockMessage:
    """Mock message object that mimics litellm's message response."""

    def __init__(self, content: str = "Test response"):
        self.content = content
        self.tool_calls = None


class MockChoice:
    """Mock choice object that mimics litellm's choice response."""

    def __init__(self, content: str = "Test response"):
        self.message = MockMessage(content)


class MockResponse:
    """Mock response object that mimics litellm's completion response."""

    def __init__(
        self,
        content: str = "Test response",
        prompt_tokens: int = 10,
        completion_tokens: int = 20,
    ):
        self.choices = [MockChoice(content)]
        self.usage = MockUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        )


class MockStreamDelta:
    """Mock delta object for streaming responses."""

    def __init__(self, content: str | None = None):
        self.content = content
        self.tool_calls = None


class MockStreamChoice:
    """Mock choice object for streaming responses."""

    def __init__(self, content: str | None = None):
        self.delta = MockStreamDelta(content)


class MockStreamChunk:
    """Mock chunk object for streaming responses."""

    def __init__(
        self,
        content: str | None = None,
        usage: MockUsage | None = None,
    ):
        self.choices = [MockStreamChoice(content)]
        self.usage = usage


def test_non_streaming_response_tracks_token_usage():
    """Test that non-streaming responses properly track token usage."""
    llm = LLM(model="gpt-4o-mini", is_litellm=True, stream=False)

    mock_response = MockResponse(
        content="Hello, world!",
        prompt_tokens=15,
        completion_tokens=25,
    )

    with patch("litellm.completion", return_value=mock_response):
        result = llm.call("Say hello")

        assert result == "Hello, world!"

        # Verify token usage was tracked
        usage_summary = llm.get_token_usage_summary()
        assert usage_summary.prompt_tokens == 15
        assert usage_summary.completion_tokens == 25
        assert usage_summary.total_tokens == 40
        assert usage_summary.successful_requests == 1


def test_non_streaming_response_accumulates_token_usage():
    """Test that multiple non-streaming calls accumulate token usage."""
    llm = LLM(model="gpt-4o-mini", is_litellm=True, stream=False)

    mock_response1 = MockResponse(
        content="First response",
        prompt_tokens=10,
        completion_tokens=20,
    )
    mock_response2 = MockResponse(
        content="Second response",
        prompt_tokens=15,
        completion_tokens=25,
    )

    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = mock_response1
        llm.call("First call")

        mock_completion.return_value = mock_response2
        llm.call("Second call")

        # Verify accumulated token usage
        usage_summary = llm.get_token_usage_summary()
        assert usage_summary.prompt_tokens == 25  # 10 + 15
        assert usage_summary.completion_tokens == 45  # 20 + 25
        assert usage_summary.total_tokens == 70  # 30 + 40
        assert usage_summary.successful_requests == 2


@pytest.mark.asyncio
async def test_async_non_streaming_response_tracks_token_usage():
    """Test that async non-streaming responses properly track token usage."""
    llm = LLM(model="gpt-4o-mini", is_litellm=True, stream=False)

    mock_response = MockResponse(
        content="Async hello!",
        prompt_tokens=12,
        completion_tokens=18,
    )

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_response
        result = await llm.acall("Say hello async")

        assert result == "Async hello!"

        # Verify token usage was tracked
        usage_summary = llm.get_token_usage_summary()
        assert usage_summary.prompt_tokens == 12
        assert usage_summary.completion_tokens == 18
        assert usage_summary.total_tokens == 30
        assert usage_summary.successful_requests == 1


@pytest.mark.asyncio
async def test_async_non_streaming_response_accumulates_token_usage():
    """Test that multiple async non-streaming calls accumulate token usage."""
    llm = LLM(model="gpt-4o-mini", is_litellm=True, stream=False)

    mock_response1 = MockResponse(
        content="First async response",
        prompt_tokens=8,
        completion_tokens=12,
    )
    mock_response2 = MockResponse(
        content="Second async response",
        prompt_tokens=10,
        completion_tokens=15,
    )

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_response1
        await llm.acall("First async call")

        mock_acompletion.return_value = mock_response2
        await llm.acall("Second async call")

        # Verify accumulated token usage
        usage_summary = llm.get_token_usage_summary()
        assert usage_summary.prompt_tokens == 18  # 8 + 10
        assert usage_summary.completion_tokens == 27  # 12 + 15
        assert usage_summary.total_tokens == 45  # 20 + 25
        assert usage_summary.successful_requests == 2


@pytest.mark.asyncio
async def test_async_streaming_response_tracks_token_usage():
    """Test that async streaming responses properly track token usage."""
    llm = LLM(model="gpt-4o-mini", is_litellm=True, stream=True)

    # Create mock streaming chunks
    chunks = [
        MockStreamChunk(content="Hello"),
        MockStreamChunk(content=", "),
        MockStreamChunk(content="world"),
        MockStreamChunk(content="!"),
        # Final chunk with usage info (this is how litellm typically sends usage)
        MockStreamChunk(
            content=None,
            usage=MockUsage(prompt_tokens=20, completion_tokens=30, total_tokens=50),
        ),
    ]

    async def mock_async_generator() -> AsyncIterator[MockStreamChunk]:
        for chunk in chunks:
            yield chunk

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_async_generator()
        result = await llm.acall("Say hello streaming")

        assert result == "Hello, world!"

        # Verify token usage was tracked
        usage_summary = llm.get_token_usage_summary()
        assert usage_summary.prompt_tokens == 20
        assert usage_summary.completion_tokens == 30
        assert usage_summary.total_tokens == 50
        assert usage_summary.successful_requests == 1


@pytest.mark.asyncio
async def test_async_streaming_response_with_dict_usage():
    """Test that async streaming handles dict-based usage info."""
    llm = LLM(model="gpt-4o-mini", is_litellm=True, stream=True)

    # Create mock streaming chunks using dict format
    class DictStreamChunk:
        def __init__(
            self,
            content: str | None = None,
            usage: dict | None = None,
        ):
            self.choices = [MockStreamChoice(content)]
            # Simulate dict-based usage (some providers return this)
            self._usage = usage

        @property
        def usage(self) -> MockUsage | None:
            if self._usage:
                return MockUsage(**self._usage)
            return None

    chunks = [
        DictStreamChunk(content="Test"),
        DictStreamChunk(content=" response"),
        DictStreamChunk(
            content=None,
            usage={
                "prompt_tokens": 25,
                "completion_tokens": 35,
                "total_tokens": 60,
            },
        ),
    ]

    async def mock_async_generator() -> AsyncIterator[DictStreamChunk]:
        for chunk in chunks:
            yield chunk

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_async_generator()
        result = await llm.acall("Test streaming with dict usage")

        assert result == "Test response"

        # Verify token usage was tracked
        usage_summary = llm.get_token_usage_summary()
        assert usage_summary.prompt_tokens == 25
        assert usage_summary.completion_tokens == 35
        assert usage_summary.total_tokens == 60
        assert usage_summary.successful_requests == 1


def test_streaming_response_tracks_token_usage():
    """Test that sync streaming responses properly track token usage."""
    llm = LLM(model="gpt-4o-mini", is_litellm=True, stream=True)

    # Create mock streaming chunks
    chunks = [
        MockStreamChunk(content="Sync"),
        MockStreamChunk(content=" streaming"),
        MockStreamChunk(content=" test"),
        # Final chunk with usage info
        MockStreamChunk(
            content=None,
            usage=MockUsage(prompt_tokens=18, completion_tokens=22, total_tokens=40),
        ),
    ]

    with patch("litellm.completion", return_value=iter(chunks)):
        result = llm.call("Test sync streaming")

        assert result == "Sync streaming test"

        # Verify token usage was tracked
        usage_summary = llm.get_token_usage_summary()
        assert usage_summary.prompt_tokens == 18
        assert usage_summary.completion_tokens == 22
        assert usage_summary.total_tokens == 40
        assert usage_summary.successful_requests == 1


def test_token_usage_with_no_usage_info():
    """Test that token usage tracking handles missing usage info gracefully."""
    llm = LLM(model="gpt-4o-mini", is_litellm=True, stream=False)

    # Create mock response without usage info
    mock_response = MagicMock()
    mock_response.choices = [MockChoice("Response without usage")]
    mock_response.usage = None

    with patch("litellm.completion", return_value=mock_response):
        result = llm.call("Test without usage")

        assert result == "Response without usage"

        # Verify token usage remains at zero
        usage_summary = llm.get_token_usage_summary()
        assert usage_summary.prompt_tokens == 0
        assert usage_summary.completion_tokens == 0
        assert usage_summary.total_tokens == 0
        assert usage_summary.successful_requests == 0


@pytest.mark.asyncio
async def test_async_streaming_with_no_usage_info():
    """Test that async streaming handles missing usage info gracefully."""
    llm = LLM(model="gpt-4o-mini", is_litellm=True, stream=True)

    # Create mock streaming chunks without usage info
    chunks = [
        MockStreamChunk(content="No"),
        MockStreamChunk(content=" usage"),
        MockStreamChunk(content=" info"),
    ]

    async def mock_async_generator() -> AsyncIterator[MockStreamChunk]:
        for chunk in chunks:
            yield chunk

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_async_generator()
        result = await llm.acall("Test without usage info")

        assert result == "No usage info"

        # Verify token usage remains at zero
        usage_summary = llm.get_token_usage_summary()
        assert usage_summary.prompt_tokens == 0
        assert usage_summary.completion_tokens == 0
        assert usage_summary.total_tokens == 0
        assert usage_summary.successful_requests == 0
