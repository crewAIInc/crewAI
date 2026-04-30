"""Async tests for Azure OpenAI Responses API support."""

import pytest


@pytest.mark.vcr()
@pytest.mark.asyncio
async def test_acall_delegates_to_responses():
    from crewai.llm import LLM

    llm = LLM(model="azure/gpt-5.2-chat", api="responses")
    result = await llm.acall("Say hello in one sentence.")

    assert isinstance(result, str)
    assert len(result) > 0
