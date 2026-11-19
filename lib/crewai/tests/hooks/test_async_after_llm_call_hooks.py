"""Async LLM calls must run the after_llm_call hooks like sync calls do."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

from crewai import LLM
from crewai.hooks.llm_hooks import (
    LLMCallHookContext,
    clear_all_llm_call_hooks,
    register_after_llm_call_hook,
)
import pytest


@pytest.fixture(autouse=True)
def _clean_llm_hooks():
    from crewai.hooks import llm_hooks

    original_before = llm_hooks._before_llm_call_hooks.copy()
    original_after = llm_hooks._after_llm_call_hooks.copy()
    clear_all_llm_call_hooks()

    yield

    clear_all_llm_call_hooks()
    llm_hooks._before_llm_call_hooks.extend(original_before)
    llm_hooks._after_llm_call_hooks.extend(original_after)


@pytest.mark.asyncio
async def test_acall_runs_after_llm_call_hooks() -> None:
    llm = LLM(model="gpt-4o-mini", is_litellm=True)

    def redact(context: LLMCallHookContext) -> str:
        return (context.response or "").replace("secret", "[redacted]")

    register_after_llm_call_hook(redact)

    with patch.object(
        llm,
        "_ahandle_non_streaming_response",
        new=AsyncMock(return_value="a secret"),
    ):
        result = await llm.acall([{"role": "user", "content": "say the secret"}])

    assert result == "a [redacted]"
