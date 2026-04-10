"""Tests for Anthropic stop_reason extraction and truncation warning.

Validates that:
- stop_reason is extracted from Anthropic API responses and forwarded
  to LLMCallCompletedEvent in all code paths (sync/async, streaming/non-streaming,
  tool-use conversation).
- A warning is logged when stop_reason == "max_tokens".
- The agent role is included in the warning when available.
"""

from __future__ import annotations

import logging
import os
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai.events.types.llm_events import LLMCallCompletedEvent, LLMCallType


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_anthropic_api_key():
    """Ensure ANTHROPIC_API_KEY is set for all tests."""
    if "ANTHROPIC_API_KEY" not in os.environ:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
            yield
    else:
        yield


def _make_text_block(text: str = "Hello") -> SimpleNamespace:
    """Create a minimal text block mimicking anthropic.types.TextBlock."""
    return SimpleNamespace(type="text", text=text)


def _make_usage(input_tokens: int = 10, output_tokens: int = 20) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_input_tokens=0,
    )


def _make_response(
    stop_reason: str = "end_turn",
    text: str = "Hello",
    content: list | None = None,
) -> SimpleNamespace:
    """Build a fake Anthropic Message response."""
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=content or [_make_text_block(text)],
        usage=_make_usage(),
        id="msg_test123",
    )


def _make_agent(role: str = "Researcher") -> SimpleNamespace:
    return SimpleNamespace(role=role, id="agent-1")


# ---------------------------------------------------------------------------
# LLMCallCompletedEvent field
# ---------------------------------------------------------------------------

class TestLLMCallCompletedEventStopReason:
    """Ensure the event model accepts and defaults stop_reason correctly."""

    def test_stop_reason_defaults_to_none(self):
        event = LLMCallCompletedEvent(
            response="hi",
            call_type=LLMCallType.LLM_CALL,
            call_id="test-call",
        )
        assert event.stop_reason is None

    def test_stop_reason_can_be_set(self):
        event = LLMCallCompletedEvent(
            response="hi",
            call_type=LLMCallType.LLM_CALL,
            call_id="test-call",
            stop_reason="max_tokens",
        )
        assert event.stop_reason == "max_tokens"

    def test_stop_reason_end_turn(self):
        event = LLMCallCompletedEvent(
            response="done",
            call_type=LLMCallType.LLM_CALL,
            call_id="test-call",
            stop_reason="end_turn",
        )
        assert event.stop_reason == "end_turn"

    def test_stop_reason_tool_use(self):
        event = LLMCallCompletedEvent(
            response="tool",
            call_type=LLMCallType.TOOL_CALL,
            call_id="test-call",
            stop_reason="tool_use",
        )
        assert event.stop_reason == "tool_use"


# ---------------------------------------------------------------------------
# _check_and_get_stop_reason helper
# ---------------------------------------------------------------------------

class TestCheckAndGetStopReason:
    """Unit tests for the _check_and_get_stop_reason helper method."""

    def _make_completion(self) -> Any:
        from crewai.llms.providers.anthropic.completion import AnthropicCompletion

        return AnthropicCompletion(
            model="claude-3-5-sonnet-20241022",
            api_key="test-key",
        )

    def test_returns_end_turn(self):
        comp = self._make_completion()
        resp = _make_response(stop_reason="end_turn")
        assert comp._check_and_get_stop_reason(resp) == "end_turn"

    def test_returns_max_tokens(self):
        comp = self._make_completion()
        resp = _make_response(stop_reason="max_tokens")
        assert comp._check_and_get_stop_reason(resp) == "max_tokens"

    def test_returns_none_when_missing(self):
        comp = self._make_completion()
        resp = SimpleNamespace(content=[], usage=_make_usage())
        assert comp._check_and_get_stop_reason(resp) is None

    def test_warns_on_max_tokens(self, caplog):
        comp = self._make_completion()
        resp = _make_response(stop_reason="max_tokens")
        with caplog.at_level(logging.WARNING):
            comp._check_and_get_stop_reason(resp)
        assert "stop_reason='max_tokens'" in caplog.text
        assert "Consider increasing max_tokens" in caplog.text

    def test_no_warning_on_end_turn(self, caplog):
        comp = self._make_completion()
        resp = _make_response(stop_reason="end_turn")
        with caplog.at_level(logging.WARNING):
            comp._check_and_get_stop_reason(resp)
        assert "max_tokens" not in caplog.text

    def test_warning_includes_agent_role(self, caplog):
        comp = self._make_completion()
        resp = _make_response(stop_reason="max_tokens")
        agent = _make_agent("DataAnalyst")
        with caplog.at_level(logging.WARNING):
            comp._check_and_get_stop_reason(resp, from_agent=agent)
        assert "[DataAnalyst]" in caplog.text

    def test_warning_without_agent(self, caplog):
        comp = self._make_completion()
        resp = _make_response(stop_reason="max_tokens")
        with caplog.at_level(logging.WARNING):
            comp._check_and_get_stop_reason(resp, from_agent=None)
        assert "Truncated response:" in caplog.text


# ---------------------------------------------------------------------------
# Integration: stop_reason propagated through _handle_completion
# ---------------------------------------------------------------------------

class TestHandleCompletionStopReason:
    """Verify stop_reason flows through _handle_completion to the event."""

    def _make_completion(self) -> Any:
        from crewai.llms.providers.anthropic.completion import AnthropicCompletion

        return AnthropicCompletion(
            model="claude-3-5-sonnet-20241022",
            api_key="test-key",
        )

    def test_stop_reason_end_turn_emitted(self):
        comp = self._make_completion()
        resp = _make_response(stop_reason="end_turn", text="Hello world")
        comp.client = MagicMock()
        comp.client.messages.create.return_value = resp

        emitted_events: list[LLMCallCompletedEvent] = []
        original_emit = comp._emit_call_completed_event

        def capture_emit(**kwargs):
            # Build event manually so we can inspect it
            emitted_events.append(kwargs)
            return original_emit(**kwargs)

        params = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4096,
            "stream": False,
        }

        with patch.object(comp, "_emit_call_completed_event", side_effect=capture_emit):
            result = comp._handle_completion(params)

        assert result == "Hello world"
        assert len(emitted_events) == 1
        assert emitted_events[0]["stop_reason"] == "end_turn"

    def test_stop_reason_max_tokens_emitted_and_warns(self, caplog):
        comp = self._make_completion()
        resp = _make_response(stop_reason="max_tokens", text="Truncated...")
        comp.client = MagicMock()
        comp.client.messages.create.return_value = resp

        emitted_events: list[dict] = []

        def capture_emit(**kwargs):
            emitted_events.append(kwargs)

        params = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4096,
            "stream": False,
        }

        with (
            patch.object(comp, "_emit_call_completed_event", side_effect=capture_emit),
            caplog.at_level(logging.WARNING),
        ):
            comp._handle_completion(params)

        assert len(emitted_events) == 1
        assert emitted_events[0]["stop_reason"] == "max_tokens"
        assert "stop_reason='max_tokens'" in caplog.text

    def test_stop_reason_none_when_attribute_missing(self):
        comp = self._make_completion()
        # Response without stop_reason attribute
        resp = SimpleNamespace(
            content=[_make_text_block("ok")],
            usage=_make_usage(),
            id="msg_no_stop",
        )
        comp.client = MagicMock()
        comp.client.messages.create.return_value = resp

        emitted_events: list[dict] = []

        def capture_emit(**kwargs):
            emitted_events.append(kwargs)

        params = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4096,
            "stream": False,
        }

        with patch.object(comp, "_emit_call_completed_event", side_effect=capture_emit):
            comp._handle_completion(params)

        assert len(emitted_events) == 1
        assert emitted_events[0]["stop_reason"] is None


# ---------------------------------------------------------------------------
# Integration: stop_reason propagated through _ahandle_completion
# ---------------------------------------------------------------------------

class TestAsyncHandleCompletionStopReason:
    """Verify stop_reason flows through _ahandle_completion."""

    def _make_completion(self) -> Any:
        from crewai.llms.providers.anthropic.completion import AnthropicCompletion

        return AnthropicCompletion(
            model="claude-3-5-sonnet-20241022",
            api_key="test-key",
        )

    @pytest.mark.asyncio
    async def test_async_stop_reason_emitted(self):
        comp = self._make_completion()
        resp = _make_response(stop_reason="end_turn", text="async hello")
        comp.async_client = MagicMock()
        comp.async_client.messages.create = AsyncMock(return_value=resp)

        emitted_events: list[dict] = []

        def capture_emit(**kwargs):
            emitted_events.append(kwargs)

        params = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4096,
            "stream": False,
        }

        with patch.object(comp, "_emit_call_completed_event", side_effect=capture_emit):
            result = await comp._ahandle_completion(params)

        assert result == "async hello"
        assert len(emitted_events) == 1
        assert emitted_events[0]["stop_reason"] == "end_turn"

    @pytest.mark.asyncio
    async def test_async_max_tokens_warns(self, caplog):
        comp = self._make_completion()
        resp = _make_response(stop_reason="max_tokens", text="cut off")
        comp.async_client = MagicMock()
        comp.async_client.messages.create = AsyncMock(return_value=resp)

        params = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4096,
            "stream": False,
        }

        with (
            patch.object(comp, "_emit_call_completed_event"),
            caplog.at_level(logging.WARNING),
        ):
            await comp._ahandle_completion(params)

        assert "stop_reason='max_tokens'" in caplog.text


# ---------------------------------------------------------------------------
# Integration: stop_reason in _handle_tool_use_conversation
# ---------------------------------------------------------------------------

class TestToolUseConversationStopReason:
    """Verify stop_reason in sync tool-use conversation path."""

    def _make_completion(self) -> Any:
        from crewai.llms.providers.anthropic.completion import AnthropicCompletion

        return AnthropicCompletion(
            model="claude-3-5-sonnet-20241022",
            api_key="test-key",
        )

    def test_tool_conversation_stop_reason(self):
        comp = self._make_completion()

        # Initial response with tool use
        tool_block = SimpleNamespace(
            type="tool_use", id="tool-1", name="search", input={"q": "test"}
        )
        initial_resp = SimpleNamespace(
            stop_reason="tool_use",
            content=[tool_block],
            usage=_make_usage(),
        )

        # Final response after tool execution
        final_resp = _make_response(stop_reason="end_turn", text="Final answer")
        comp.client = MagicMock()
        comp.client.messages.create.return_value = final_resp

        emitted_events: list[dict] = []

        def capture_emit(**kwargs):
            emitted_events.append(kwargs)

        def mock_tool_exec(**kwargs):
            return "tool result"

        params = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4096,
            "stream": False,
        }

        with (
            patch.object(comp, "_emit_call_completed_event", side_effect=capture_emit),
            patch.object(comp, "_handle_tool_execution", return_value="tool result"),
        ):
            result = comp._handle_tool_use_conversation(
                initial_response=initial_resp,
                tool_uses=[tool_block],
                params=params,
                available_functions={"search": lambda **kw: "ok"},
            )

        assert result == "Final answer"
        assert len(emitted_events) == 1
        assert emitted_events[0]["stop_reason"] == "end_turn"

    def test_tool_conversation_max_tokens_warns(self, caplog):
        comp = self._make_completion()

        tool_block = SimpleNamespace(
            type="tool_use", id="tool-1", name="search", input={"q": "test"}
        )
        initial_resp = SimpleNamespace(
            stop_reason="tool_use",
            content=[tool_block],
            usage=_make_usage(),
        )

        final_resp = _make_response(stop_reason="max_tokens", text="Truncated")
        comp.client = MagicMock()
        comp.client.messages.create.return_value = final_resp

        params = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4096,
            "stream": False,
        }

        with (
            patch.object(comp, "_emit_call_completed_event"),
            patch.object(comp, "_handle_tool_execution", return_value="tool result"),
            caplog.at_level(logging.WARNING),
        ):
            comp._handle_tool_use_conversation(
                initial_response=initial_resp,
                tool_uses=[tool_block],
                params=params,
                available_functions={"search": lambda **kw: "ok"},
            )

        assert "stop_reason='max_tokens'" in caplog.text


# ---------------------------------------------------------------------------
# Integration: stop_reason in async _ahandle_tool_use_conversation
# ---------------------------------------------------------------------------

class TestAsyncToolUseConversationStopReason:
    """Verify stop_reason in async tool-use conversation path."""

    def _make_completion(self) -> Any:
        from crewai.llms.providers.anthropic.completion import AnthropicCompletion

        return AnthropicCompletion(
            model="claude-3-5-sonnet-20241022",
            api_key="test-key",
        )

    @pytest.mark.asyncio
    async def test_async_tool_conversation_stop_reason(self):
        comp = self._make_completion()

        tool_block = SimpleNamespace(
            type="tool_use", id="tool-1", name="search", input={"q": "test"}
        )
        initial_resp = SimpleNamespace(
            stop_reason="tool_use",
            content=[tool_block],
            usage=_make_usage(),
        )

        final_resp = _make_response(stop_reason="end_turn", text="Async final")
        comp.async_client = MagicMock()
        comp.async_client.messages.create = AsyncMock(return_value=final_resp)

        emitted_events: list[dict] = []

        def capture_emit(**kwargs):
            emitted_events.append(kwargs)

        params = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4096,
            "stream": False,
        }

        with (
            patch.object(comp, "_emit_call_completed_event", side_effect=capture_emit),
            patch.object(comp, "_handle_tool_execution", return_value="tool result"),
        ):
            result = await comp._ahandle_tool_use_conversation(
                initial_response=initial_resp,
                tool_uses=[tool_block],
                params=params,
                available_functions={"search": lambda **kw: "ok"},
            )

        assert result == "Async final"
        assert len(emitted_events) == 1
        assert emitted_events[0]["stop_reason"] == "end_turn"
