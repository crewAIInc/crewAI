"""Tests for Bedrock async completion functionality.

Note: These tests are skipped in CI because VCR.py does not support
aiobotocore's HTTP session. The cassettes were recorded locally but
cannot be played back properly in CI.
"""

import pytest
import tiktoken

from crewai.llm import LLM

SKIP_REASON = "VCR does not support aiobotocore async HTTP client"


@pytest.mark.vcr()
@pytest.mark.asyncio
@pytest.mark.skip(reason=SKIP_REASON)
async def test_bedrock_async_basic_call():
    """Test basic async call with Bedrock."""
    llm = LLM(model="bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0")

    result = await llm.acall("Say hello")

    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.vcr()
@pytest.mark.asyncio
@pytest.mark.skip(reason=SKIP_REASON)
async def test_bedrock_async_with_temperature():
    """Test async call with temperature parameter."""
    llm = LLM(model="bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0", temperature=0.1)

    result = await llm.acall("Say the word 'test' once")

    assert result is not None
    assert isinstance(result, str)


@pytest.mark.vcr()
@pytest.mark.asyncio
@pytest.mark.skip(reason=SKIP_REASON)
async def test_bedrock_async_with_max_tokens():
    """Test async call with max_tokens parameter."""
    llm = LLM(model="bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0", max_tokens=10)

    result = await llm.acall("Write a very long story about a dragon.")

    assert result is not None
    assert isinstance(result, str)
    encoder = tiktoken.get_encoding("cl100k_base")
    token_count = len(encoder.encode(result))
    assert token_count <= 10


@pytest.mark.vcr()
@pytest.mark.asyncio
@pytest.mark.skip(reason=SKIP_REASON)
async def test_bedrock_async_with_system_message():
    """Test async call with system message."""
    llm = LLM(model="bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ]

    result = await llm.acall(messages)

    assert result is not None
    assert isinstance(result, str)


@pytest.mark.vcr()
@pytest.mark.asyncio
@pytest.mark.skip(reason=SKIP_REASON)
async def test_bedrock_async_conversation():
    """Test async call with conversation history."""
    llm = LLM(model="bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0")

    messages = [
        {"role": "user", "content": "My name is Alice."},
        {"role": "assistant", "content": "Hello Alice! Nice to meet you."},
        {"role": "user", "content": "What is my name?"}
    ]

    result = await llm.acall(messages)

    assert result is not None
    assert isinstance(result, str)


@pytest.mark.vcr()
@pytest.mark.asyncio
@pytest.mark.skip(reason=SKIP_REASON)
async def test_bedrock_async_multiple_calls():
    """Test making multiple async calls in sequence."""
    llm = LLM(model="bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0")

    result1 = await llm.acall("What is 1+1?")
    result2 = await llm.acall("What is 2+2?")

    assert result1 is not None
    assert result2 is not None
    assert isinstance(result1, str)
    assert isinstance(result2, str)


@pytest.mark.vcr()
@pytest.mark.asyncio
@pytest.mark.skip(reason=SKIP_REASON)
async def test_bedrock_async_with_parameters():
    """Test async call with multiple parameters."""
    llm = LLM(
        model="bedrock/us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        temperature=0.7,
        max_tokens=100,
        top_p=0.9
    )

    result = await llm.acall("Tell me a short fact")

    assert result is not None
    assert isinstance(result, str)
