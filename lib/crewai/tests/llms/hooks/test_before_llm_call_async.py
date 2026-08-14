"""Tests that before_llm_call hooks run on async acall() path for all native providers.

Regression tests for issue #6739: before_llm_call hooks were never invoked on the
async acall() path of the five native providers, meaning a blocking hook (PII gate,
spend cap, policy check) silently failed to block the request whenever the caller
used acall() / kickoff_async.
"""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from crewai.llm import LLM

# Async tests are skipped on Windows due to pytest-recording network guard
# conflicting with asyncio's socket.socketpair() used for event loop creation.
# CI runs on Linux and covers these paths fully.
skip_async_on_windows = pytest.mark.skipif(
    sys.platform == "win32",
    reason="Async tests skipped on Windows (pytest-recording + asyncio compatibility)",
)


@skip_async_on_windows
class TestBeforeLLMCallHookRunsOnAsyncOpenAI:
    """Verify before_llm_call hook is invoked on OpenAI native async path."""

    @pytest.mark.asyncio
    async def test_acall_invokes_before_llm_call_hooks(self):
        """acall() must invoke _invoke_before_llm_call_hooks before the API call."""
        llm = LLM(model="gpt-4o-mini")
        provider = llm._provider

        with patch.object(
            provider, "_invoke_before_llm_call_hooks", return_value=True
        ), patch.object(
            provider, "_acall_completions", return_value="mocked response"
        ), patch.object(
            provider, "_acall_responses", return_value="mocked response"
        ):
            result = await llm.acall("Hello")

            provider._invoke_before_llm_call_hooks.assert_called_once()
            assert result == "mocked response"

    @pytest.mark.asyncio
    async def test_acall_blocks_when_hook_returns_false(self):
        """acall() must raise ValueError and NOT call API when hook blocks."""
        llm = LLM(model="gpt-4o-mini")
        provider = llm._provider

        with patch.object(
            provider, "_invoke_before_llm_call_hooks", return_value=False
        ), patch.object(
            provider, "_acall_completions"
        ) as mock_acall_completions, patch.object(
            provider, "_acall_responses"
        ) as mock_acall_responses:
            with pytest.raises(ValueError, match="blocked by before_llm_call hook"):
                await llm.acall("Hello")

            mock_acall_completions.assert_not_called()
            mock_acall_responses.assert_not_called()


@skip_async_on_windows
class TestBeforeLLMCallHookRunsOnAsyncAnthropic:
    """Verify before_llm_call hook is invoked on Anthropic native async path."""

    @pytest.mark.asyncio
    async def test_acall_invokes_before_llm_call_hooks(self):
        """acall() must invoke _invoke_before_llm_call_hooks before the API call."""
        llm = LLM(model="anthropic/claude-3-5-haiku-latest")
        provider = llm._provider

        with patch.object(
            provider, "_invoke_before_llm_call_hooks", return_value=True
        ), patch.object(
            provider, "_ahandle_completion", return_value="mocked response"
        ), patch.object(
            provider, "_ahandle_streaming_completion", return_value="mocked response"
        ):
            result = await llm.acall("Hello")

            provider._invoke_before_llm_call_hooks.assert_called_once()
            assert result == "mocked response"

    @pytest.mark.asyncio
    async def test_acall_blocks_when_hook_returns_false(self):
        """acall() must raise ValueError and NOT call API when hook blocks."""
        llm = LLM(model="anthropic/claude-3-5-haiku-latest")
        provider = llm._provider

        with patch.object(
            provider, "_invoke_before_llm_call_hooks", return_value=False
        ), patch.object(
            provider, "_ahandle_completion"
        ) as mock_ahandle, patch.object(
            provider, "_ahandle_streaming_completion"
        ) as mock_ahandle_streaming:
            with pytest.raises(ValueError, match="blocked by before_llm_call hook"):
                await llm.acall("Hello")

            mock_ahandle.assert_not_called()
            mock_ahandle_streaming.assert_not_called()


@skip_async_on_windows
class TestBeforeLLMCallHookRunsOnAsyncBedrock:
    """Verify before_llm_call hook is invoked on Bedrock native async path."""

    @pytest.mark.asyncio
    async def test_acall_invokes_before_llm_call_hooks(self):
        """acall() must invoke _invoke_before_llm_call_hooks before the API call."""
        llm = LLM(
            model="bedrock/anthropic.claude-3-5-haiku-20241022-v1:0",
        )
        provider = llm._provider

        with patch.object(
            provider, "_invoke_before_llm_call_hooks", return_value=True
        ), patch.object(
            provider, "_ahandle_converse_response", return_value="mocked response"
        ), patch.object(
            provider, "_ahandle_streaming_converse", return_value="mocked response"
        ):
            result = await llm.acall("Hello")

            provider._invoke_before_llm_call_hooks.assert_called_once()
            assert result == "mocked response"

    @pytest.mark.asyncio
    async def test_acall_blocks_when_hook_returns_false(self):
        """acall() must raise ValueError and NOT call API when hook blocks."""
        llm = LLM(
            model="bedrock/anthropic.claude-3-5-haiku-20241022-v1:0",
        )
        provider = llm._provider

        with patch.object(
            provider, "_invoke_before_llm_call_hooks", return_value=False
        ), patch.object(
            provider, "_ahandle_converse_response"
        ) as mock_ahandle, patch.object(
            provider, "_ahandle_streaming_converse"
        ) as mock_ahandle_streaming:
            with pytest.raises(ValueError, match="blocked by before_llm_call hook"):
                await llm.acall("Hello")

            mock_ahandle.assert_not_called()
            mock_ahandle_streaming.assert_not_called()


@skip_async_on_windows
class TestBeforeLLMCallHookRunsOnAsyncGemini:
    """Verify before_llm_call hook is invoked on Gemini native async path."""

    @pytest.mark.asyncio
    async def test_acall_invokes_before_llm_call_hooks(self):
        """acall() must invoke _invoke_before_llm_call_hooks before the API call."""
        llm = LLM(model="gemini/gemini-2.0-flash")
        provider = llm._provider

        with patch.object(
            provider, "_invoke_before_llm_call_hooks", return_value=True
        ), patch.object(
            provider, "_ahandle_completion", return_value="mocked response"
        ), patch.object(
            provider, "_ahandle_streaming_completion", return_value="mocked response"
        ):
            result = await llm.acall("Hello")

            provider._invoke_before_llm_call_hooks.assert_called_once()
            assert result == "mocked response"

    @pytest.mark.asyncio
    async def test_acall_blocks_when_hook_returns_false(self):
        """acall() must raise ValueError and NOT call API when hook blocks."""
        llm = LLM(model="gemini/gemini-2.0-flash")
        provider = llm._provider

        with patch.object(
            provider, "_invoke_before_llm_call_hooks", return_value=False
        ), patch.object(
            provider, "_ahandle_completion"
        ) as mock_ahandle, patch.object(
            provider, "_ahandle_streaming_completion"
        ) as mock_ahandle_streaming:
            with pytest.raises(ValueError, match="blocked by before_llm_call hook"):
                await llm.acall("Hello")

            mock_ahandle.assert_not_called()
            mock_ahandle_streaming.assert_not_called()


@skip_async_on_windows
class TestBeforeLLMCallHookRunsOnAsyncAzure:
    """Verify before_llm_call hook is invoked on Azure AI native async path."""

    @pytest.mark.asyncio
    async def test_acall_invokes_before_llm_call_hooks(self):
        """acall() must invoke _invoke_before_llm_call_hooks before the API call."""
        llm = LLM(
            model="azure/gpt-4o-mini",
            base_url="https://example.openai.azure.com",
            api_key="test-key",
            api_version="2024-06-01",
        )
        provider = llm._provider

        with patch.object(
            provider, "_invoke_before_llm_call_hooks", return_value=True
        ), patch.object(
            provider, "_ahandle_completion", return_value="mocked response"
        ), patch.object(
            provider, "_ahandle_streaming_completion", return_value="mocked response"
        ):
            result = await llm.acall("Hello")

            provider._invoke_before_llm_call_hooks.assert_called_once()
            assert result == "mocked response"

    @pytest.mark.asyncio
    async def test_acall_blocks_when_hook_returns_false(self):
        """acall() must raise ValueError and NOT call API when hook blocks."""
        llm = LLM(
            model="azure/gpt-4o-mini",
            base_url="https://example.openai.azure.com",
            api_key="test-key",
            api_version="2024-06-01",
        )
        provider = llm._provider

        with patch.object(
            provider, "_invoke_before_llm_call_hooks", return_value=False
        ), patch.object(
            provider, "_ahandle_completion"
        ) as mock_ahandle, patch.object(
            provider, "_ahandle_streaming_completion"
        ) as mock_ahandle_streaming:
            with pytest.raises(ValueError, match="blocked by before_llm_call hook"):
                await llm.acall("Hello")

            mock_ahandle.assert_not_called()
            mock_ahandle_streaming.assert_not_called()
