"""Shared behavioral contract for the native LLM providers.

Every native provider has its own suite for SDK-specific behaviour (tool calls,
streaming, structured output, provider quirks). This module pins only the part
they must all agree on, so a change in one provider cannot silently drift away
from the others:

* a plain-text request returns the provider's text, not a wrapper object,
* the sync and async entry points return the same thing,
* the request reaches the SDK with the model the caller asked for.

The SDK responses here are real ``ChatCompletion`` / ``Message`` instances
rather than loose mocks, so a provider that starts reading a field the SDK does
not send fails here instead of in production. No network call is made and no
credentials are read, apart from the placeholder keys the constructors require.
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from anthropic.types import Message, TextBlock, ToolUseBlock, Usage
from openai.types.chat import ChatCompletion
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage
from openai.types.chat.chat_completion_message_tool_call import (
    ChatCompletionMessageFunctionToolCall,
    Function,
)
from openai.types.completion_usage import CompletionUsage

from crewai.llms.providers.anthropic.completion import AnthropicCompletion
from crewai.llms.providers.openai.completion import OpenAICompletion


OPENAI_MODEL = "gpt-4o"
ANTHROPIC_MODEL = "claude-sonnet-4-6"


def _openai_chat_completion(text: str) -> ChatCompletion:
    """A real ChatCompletion, so the provider's field reads are exercised."""
    return ChatCompletion(
        id="chatcmpl-contract",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content=text),
            )
        ],
        created=0,
        model=OPENAI_MODEL,
        object="chat.completion",
        usage=CompletionUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    )


def _openai_raw_response(text: str) -> Any:
    """The ``with_raw_response`` wrapper the provider parses."""
    raw = MagicMock()
    raw.text = ""
    raw.http_response = httpx.Response(
        200, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    )
    raw.parse.return_value = _openai_chat_completion(text)
    return raw


def _anthropic_message(text: str) -> Message:
    """A real Message, so the provider's content-block reads are exercised."""
    return Message(
        id="msg_contract",
        content=[TextBlock(type="text", text=text)],
        model=ANTHROPIC_MODEL,
        role="assistant",
        stop_reason="end_turn",
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=3, output_tokens=2),
    )


class _ProviderCase:
    """How to build one provider with a fake SDK behind it."""

    def __init__(
        self,
        name: str,
        completion_class: type,
        model: str,
        sync_builder: Callable[[str], tuple[Any, Any]],
        async_builder: Callable[[str], tuple[Any, Any]],
        failure_builder: Callable[[], tuple[Any, Any]],
        tool_builder: Callable[[str], tuple[Any, Any]],
    ) -> None:
        self.name = name
        self.completion_class = completion_class
        self.model = model
        self._sync_builder = sync_builder
        self._async_builder = async_builder
        self._failure_builder = failure_builder
        self._tool_builder = tool_builder

    def completion(self) -> Any:
        return self.completion_class(model=self.model, api_key="placeholder-key")

    def with_sync_response(self, text: str) -> tuple[Any, Any]:
        return self._sync_builder(text)

    def with_async_response(self, text: str) -> tuple[Any, Any]:
        return self._async_builder(text)

    def with_failing_call(self) -> tuple[Any, Any]:
        return self._failure_builder()

    def with_tool_call_response(self, call_id: str) -> tuple[Any, Any]:
        return self._tool_builder(call_id)


def _openai_sync(text: str) -> tuple[Any, Any]:
    fake_client = MagicMock()
    fake_client.chat.completions.with_raw_response.create.return_value = (
        _openai_raw_response(text)
    )
    return fake_client, patch.object(
        OpenAICompletion, "_get_sync_client", return_value=fake_client
    )


def _openai_failure() -> tuple[Any, Any]:
    fake_client = MagicMock()
    fake_client.chat.completions.with_raw_response.create.side_effect = RuntimeError(
        "boom"
    )
    return fake_client, patch.object(
        OpenAICompletion, "_get_sync_client", return_value=fake_client
    )


def _openai_async(text: str) -> tuple[Any, Any]:
    fake_client = MagicMock()
    fake_client.chat.completions.with_raw_response.create = AsyncMock(
        return_value=_openai_raw_response(text)
    )
    return fake_client, patch.object(
        OpenAICompletion, "_get_async_client", return_value=fake_client
    )


def _anthropic_sync(text: str) -> tuple[Any, Any]:
    fake_client = MagicMock()
    fake_client.messages.create.return_value = _anthropic_message(text)
    return fake_client, patch.object(
        AnthropicCompletion, "_get_sync_client", return_value=fake_client
    )


def _anthropic_failure() -> tuple[Any, Any]:
    fake_client = MagicMock()
    fake_client.messages.create.side_effect = RuntimeError("boom")
    return fake_client, patch.object(
        AnthropicCompletion, "_get_sync_client", return_value=fake_client
    )


def _openai_tool_sync(call_id: str) -> tuple[Any, Any]:
    completion = ChatCompletion(
        id="chatcmpl-tool",
        choices=[
            Choice(
                index=0,
                finish_reason="tool_calls",
                message=ChatCompletionMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[
                        ChatCompletionMessageFunctionToolCall(
                            id=call_id,
                            type="function",
                            function=Function(name="lookup", arguments='{"q": "x"}'),
                        )
                    ],
                ),
            )
        ],
        created=0,
        model=OPENAI_MODEL,
        object="chat.completion",
        usage=CompletionUsage(prompt_tokens=3, completion_tokens=2, total_tokens=5),
    )
    wrapper = MagicMock()
    wrapper.text = ""
    wrapper.http_response = httpx.Response(
        200, request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    )
    wrapper.parse.return_value = completion
    fake_client = MagicMock()
    fake_client.chat.completions.with_raw_response.create.return_value = wrapper
    return fake_client, patch.object(
        OpenAICompletion, "_get_sync_client", return_value=fake_client
    )


def _anthropic_tool_sync(call_id: str) -> tuple[Any, Any]:
    message = Message(
        id="msg_tool",
        content=[ToolUseBlock(id=call_id, name="lookup", input={"q": "x"}, type="tool_use")],
        model=ANTHROPIC_MODEL,
        role="assistant",
        stop_reason="tool_use",
        stop_sequence=None,
        type="message",
        usage=Usage(input_tokens=3, output_tokens=2),
    )
    fake_client = MagicMock()
    fake_client.messages.create.return_value = message
    return fake_client, patch.object(
        AnthropicCompletion, "_get_sync_client", return_value=fake_client
    )


def _anthropic_async(text: str) -> tuple[Any, Any]:
    fake_client = MagicMock()
    fake_client.messages.create = AsyncMock(return_value=_anthropic_message(text))
    return fake_client, patch.object(
        AnthropicCompletion, "_get_async_client", return_value=fake_client
    )


PROVIDERS = [
    pytest.param(
        _ProviderCase(
            "openai",
            OpenAICompletion,
            OPENAI_MODEL,
            _openai_sync,
            _openai_async,
            _openai_failure,
            _openai_tool_sync,
        ),
        id="openai",
    ),
    pytest.param(
        _ProviderCase(
            "anthropic",
            AnthropicCompletion,
            ANTHROPIC_MODEL,
            _anthropic_sync,
            _anthropic_async,
            _anthropic_failure,
            _anthropic_tool_sync,
        ),
        id="anthropic",
    ),
]


@pytest.mark.parametrize("case", PROVIDERS)
def test_plain_text_request_returns_the_provider_text(case: _ProviderCase) -> None:
    """``call`` returns the assistant text itself, not the SDK response object."""
    llm = case.completion()
    _, context = case.with_sync_response("hello from the contract")

    with context:
        result = llm.call("hi")

    assert isinstance(result, str)
    assert result == "hello from the contract"


@pytest.mark.parametrize("case", PROVIDERS)
def test_async_call_matches_the_sync_call(case: _ProviderCase) -> None:
    """``acall`` returns the same text as ``call`` for the same request."""
    llm = case.completion()
    _, context = case.with_async_response("async answer")

    with context:
        result = asyncio.run(llm.acall("hi"))

    assert isinstance(result, str)
    assert result == "async answer"


@pytest.mark.parametrize("case", PROVIDERS)
def test_request_carries_the_requested_model(case: _ProviderCase) -> None:
    """The SDK sees the model the caller configured, once, unchanged."""
    llm = case.completion()
    fake_client, context = case.with_sync_response("ok")

    with context:
        llm.call("hi")

    if case.name == "openai":
        sent = fake_client.chat.completions.with_raw_response.create.call_args.kwargs
    else:
        sent = fake_client.messages.create.call_args.kwargs

    assert sent["model"] == case.model
    assert sent["messages"]


@pytest.mark.parametrize("case", PROVIDERS)
def test_a_plain_string_prompt_reaches_the_sdk_as_a_user_message(
    case: _ProviderCase,
) -> None:
    """A bare string prompt is normalised to a user message before the call."""
    llm = case.completion()
    fake_client, context = case.with_sync_response("ok")

    with context:
        llm.call("what is the contract?")

    if case.name == "openai":
        sent = fake_client.chat.completions.with_raw_response.create.call_args.kwargs
    else:
        sent = fake_client.messages.create.call_args.kwargs

    roles = [message["role"] for message in sent["messages"]]
    assert roles[-1] == "user"
    assert sent["messages"][-1]["content"] != ""


@pytest.mark.parametrize("case", PROVIDERS)
def test_one_call_records_usage_with_the_same_keys(case: _ProviderCase) -> None:
    """Both providers surface identical usage keys for the same exchange."""
    llm = case.completion()
    _, context = case.with_sync_response("hello")

    with context:
        llm.call("hi")

    usage = llm._token_usage
    assert usage["successful_requests"] == 1
    assert usage["prompt_tokens"] == 3
    assert usage["completion_tokens"] == 2
    assert usage["total_tokens"] == 5


@pytest.mark.parametrize("case", PROVIDERS)
def test_a_failed_call_emits_exactly_one_failure_event(case: _ProviderCase) -> None:
    """One logical failure is one ``llm_call_failed`` event, either provider."""
    from crewai.events.event_bus import crewai_event_bus
    from crewai.events.types.llm_events import LLMCallFailedEvent

    with crewai_event_bus.scoped_handlers():
        received: list[LLMCallFailedEvent] = []
        crewai_event_bus.register_handler(
            LLMCallFailedEvent,
            lambda source, event: received.append(event),
        )

        llm = case.completion()
        _, context = case.with_failing_call()

        with context:
            with pytest.raises(RuntimeError, match="boom"):
                llm.call("hi")

        # Sync handlers run on the bus's executor; flush before asserting.
        assert crewai_event_bus.flush(timeout=5)
        assert len(received) == 1
        assert received[0].model == case.model


@pytest.mark.parametrize("case", PROVIDERS)
def test_a_tool_call_response_reaches_the_caller_with_its_id(
    case: _ProviderCase,
) -> None:
    """Without available_functions the caller owns execution, so it must get
    the provider's tool calls back — id, name and arguments readable."""
    llm = case.completion()
    _, context = case.with_tool_call_response("tc_42")

    with context:
        result = llm.call("hi")

    assert isinstance(result, list) and len(result) == 1
    tool_call = result[0]
    call_id = tool_call.get("id") if isinstance(tool_call, dict) else tool_call.id
    name = (
        tool_call.get("name")
        if isinstance(tool_call, dict)
        else getattr(tool_call, "name", None)
        or tool_call.function.name
    )
    assert call_id == "tc_42"
    assert name == "lookup"
