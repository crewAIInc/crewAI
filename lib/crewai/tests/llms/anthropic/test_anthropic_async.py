"""Tests for Anthropic async completion functionality."""

import pytest

from crewai.llm import LLM


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_anthropic_async_basic_call():
    """Test basic async call with Anthropic."""
    llm = LLM(model="anthropic/claude-sonnet-4-0")

    result = await llm.acall("Say hello")

    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_anthropic_async_with_temperature():
    """Test async call with temperature parameter."""
    llm = LLM(model="anthropic/claude-sonnet-4-0", temperature=0.1)

    result = await llm.acall("Say the word 'test' once")

    assert result is not None
    assert isinstance(result, str)


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_anthropic_async_with_max_tokens():
    """Test async call with max_tokens parameter."""
    llm = LLM(model="anthropic/claude-sonnet-4-0", max_tokens=10)

    result = await llm.acall("Write a very long story about a dragon.")

    assert result is not None
    assert isinstance(result, str)


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_anthropic_async_with_system_message():
    """Test async call with system message."""
    llm = LLM(model="anthropic/claude-sonnet-4-0")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ]

    result = await llm.acall(messages)

    assert result is not None
    assert isinstance(result, str)


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_anthropic_async_conversation():
    """Test async call with conversation history."""
    llm = LLM(model="anthropic/claude-sonnet-4-0")

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
async def test_anthropic_async_stop_sequences():
    """Test async call with stop sequences."""
    llm = LLM(
        model="anthropic/claude-sonnet-4-0",
        stop_sequences=["END", "STOP"]
    )

    result = await llm.acall("Count from 1 to 10")

    assert result is not None
    assert isinstance(result, str)


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_anthropic_async_multiple_calls():
    """Test making multiple async calls in sequence."""
    llm = LLM(model="anthropic/claude-sonnet-4-0")

    result1 = await llm.acall("What is 1+1?")
    result2 = await llm.acall("What is 2+2?")

    assert result1 is not None
    assert result2 is not None
    assert isinstance(result1, str)
    assert isinstance(result2, str)
