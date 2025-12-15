"""Tests for Google (Gemini) async completion functionality."""

import pytest
import tiktoken

from crewai import Agent, Task, Crew
from crewai.llm import LLM
from crewai.llms.providers.gemini.completion import GeminiCompletion


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_gemini_async_basic_call():
    """Test basic async call with Gemini."""
    llm = LLM(model="gemini/gemini-3-pro-preview")

    result = await llm.acall("Say hello")

    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_gemini_async_with_temperature():
    """Test async call with temperature parameter."""
    llm = LLM(model="gemini/gemini-3-pro-preview", temperature=0.1)

    result = await llm.acall("Say the word 'test' once")

    assert result is not None
    assert isinstance(result, str)


@pytest.mark.asyncio
@pytest.mark.vcr
async def test_gemini_async_with_max_tokens():
    """Test async call with max_tokens parameter."""
    llm = GeminiCompletion(model="gemini-3-pro-preview", max_output_tokens=1000)

    result = await llm.acall("Write a very short story about a dragon.")

    assert result is not None
    assert isinstance(result, str)
    encoder = tiktoken.get_encoding("cl100k_base")
    token_count = len(encoder.encode(result))
    assert token_count <= 1000


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_gemini_async_with_system_message():
    """Test async call with system message."""
    llm = LLM(model="gemini/gemini-3-pro-preview")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is 2+2?"}
    ]

    result = await llm.acall(messages)

    assert result is not None
    assert isinstance(result, str)


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_gemini_async_conversation():
    """Test async call with conversation history."""
    llm = LLM(model="gemini/gemini-3-pro-preview")

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
async def test_gemini_async_multiple_calls():
    """Test making multiple async calls in sequence."""
    llm = LLM(model="gemini/gemini-3-pro-preview")

    result1 = await llm.acall("What is 1+1?")
    result2 = await llm.acall("What is 2+2?")

    assert result1 is not None
    assert result2 is not None
    assert isinstance(result1, str)
    assert isinstance(result2, str)


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_gemini_async_with_parameters():
    """Test async call with multiple parameters."""
    llm = LLM(
        model="gemini/gemini-3-pro-preview",
        temperature=0.7,
        max_output_tokens=1000,
        top_p=0.9
    )

    result = await llm.acall("Tell me a short fact")

    assert result is not None
    assert isinstance(result, str)


@pytest.mark.vcr()
@pytest.mark.asyncio
@pytest.mark.skip(reason="VCR cannot replay SSE streaming responses")
async def test_google_async_streaming_returns_usage_metrics():
    """
    Test that Google Gemini async streaming calls return proper token usage metrics.
    """
    agent = Agent(
        role="Research Assistant",
        goal="Find information about the capital of Canada",
        backstory="You are a helpful research assistant.",
        llm=LLM(model="gemini/gemini-2.0-flash-exp", stream=True),
        verbose=True,
    )

    task = Task(
        description="What is the capital of Canada?",
        expected_output="The capital of Canada",
        agent=agent,
    )

    crew = Crew(agents=[agent], tasks=[task])
    result = await crew.kickoff_async()

    assert result.token_usage is not None
    assert result.token_usage.total_tokens > 0
    assert result.token_usage.prompt_tokens > 0
    assert result.token_usage.completion_tokens > 0
    assert result.token_usage.successful_requests >= 1
