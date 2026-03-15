"""Tests for Azure async completion functionality."""

import pytest
import tiktoken

from crewai import Agent, Task, Crew
from crewai.llm import LLM


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_azure_async_non_streaming():
    """Test basic async non-streaming call."""
    llm = LLM(model="azure/gpt-4o-mini", stream=False)

    result = await llm.acall("Say hello")

    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_azure_async_multiple_calls():
    """Test making multiple async calls in sequence."""
    llm = LLM(model="azure/gpt-4o-mini", stream=False)

    result1 = await llm.acall("What is 1+1?")
    result2 = await llm.acall("What is 2+2?")

    assert result1 is not None
    assert result2 is not None
    assert isinstance(result1, str)
    assert isinstance(result2, str)


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_azure_async_with_temperature():
    """Test async call with temperature parameter."""
    llm = LLM(model="azure/gpt-4o-mini", temperature=0.1, stream=False)

    result = await llm.acall("Say the word 'test' once")

    assert result is not None
    assert isinstance(result, str)


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_azure_async_with_max_tokens():
    """Test async call with max_tokens parameter."""
    llm = LLM(model="azure/gpt-4o-mini", max_tokens=10, stream=False)

    result = await llm.acall("Write a very long story about a dragon.")

    assert result is not None
    assert isinstance(result, str)
    encoder = tiktoken.get_encoding("cl100k_base")
    token_count = len(encoder.encode(result))
    assert token_count <= 10


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_azure_async_with_system_message():
    """Test async call with system message."""
    llm = LLM(model="azure/gpt-4o-mini", stream=False)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ]

    result = await llm.acall(messages)

    assert result is not None
    assert isinstance(result, str)


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_azure_async_with_parameters():
    """Test async call with multiple parameters."""
    llm = LLM(
        model="azure/gpt-4o-mini",
        temperature=0.7,
        max_tokens=100,
        top_p=0.9,
        frequency_penalty=0.5,
        presence_penalty=0.3,
        stream=False
    )

    result = await llm.acall("Tell me a short fact")

    assert result is not None
    assert isinstance(result, str)


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_azure_async_conversation():
    """Test async call with conversation history."""
    llm = LLM(model="azure/gpt-4o-mini", stream=False)

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
async def test_azure_async_streaming_returns_usage_metrics():
    """
    Test that Azure async streaming calls return proper token usage metrics.
    """
    agent = Agent(
        role="Research Assistant",
        goal="Find information about the capital of Germany",
        backstory="You are a helpful research assistant.",
        llm=LLM(model="azure/gpt-4o-mini", stream=True),
        verbose=True,
    )

    task = Task(
        description="What is the capital of Germany?",
        expected_output="The capital of Germany",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task])
    result = await crew.kickoff_async()

    assert result.token_usage is not None
    assert result.token_usage.total_tokens > 0
    assert result.token_usage.prompt_tokens > 0
    assert result.token_usage.completion_tokens > 0
    assert result.token_usage.successful_requests >= 1
