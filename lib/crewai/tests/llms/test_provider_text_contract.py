"""Shared behavioral contract for native providers on a plain-text request.

Issue #8006 asks for a small set of shared tests that say what every native
provider must do identically, so provider drift shows up as a failing
assertion instead of as a surprise in production. This module covers the first
slice only: a plain-text request, the text that comes back, and sync/async
parity -- for OpenAI and Anthropic, driven by fake SDK responses so the tests
need no API key and make no paid call. Multimodal content, tool calls,
streaming, token usage and further providers are separate follow-ups.

The contract asserted here, for a non-streaming, tool-free call:

1. ``call(str)`` reaches the provider SDK exactly once, carrying the model and
   the prompt as a single ``user`` message whose text is the prompt verbatim.
2. The return value is the assistant text as a ``str`` -- never a container or
   its ``repr``.
3. Assistant text the provider omits collapses to ``""``, never ``None``.
4. ``acall`` returns what ``call`` returns, and goes through the async client.
5. Both paths emit exactly one ``LLMCallStartedEvent`` and one
   ``LLMCallCompletedEvent``, the latter carrying the returned text and
   ``LLMCallType.LLM_CALL``.

Legitimate differences are recorded on the provider adapters rather than
hidden behind a shared assertion -- see ``_SYSTEM_IN_MESSAGES`` for system
prompt placement, and the per-provider fake responses for the differing shapes
of "no assistant text".
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import httpx
import pytest

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallStartedEvent,
    LLMCallType,
)
from crewai.llm import LLM
from crewai.llms.base_llm import BaseLLM


try:
    from crewai.llms.providers.anthropic.completion import (  # noqa: F401
        AnthropicCompletion,
    )

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


@pytest.fixture(autouse=True)
def fake_api_keys():
    """Give every provider a syntactically valid key so no real one is needed."""
    with patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "test-key", "ANTHROPIC_API_KEY": "test-key"},
    ):
        yield


class _Recorder:
    """Stands in for a provider SDK call, recording the request it received."""

    def __init__(self, response: Any) -> None:
        self.response = response
        self.requests: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        return self.response

    @property
    def request(self) -> dict[str, Any]:
        assert len(self.requests) == 1, (
            f"expected exactly one provider request, got {len(self.requests)}"
        )
        return self.requests[0]


class _AsyncRecorder(_Recorder):
    """``_Recorder`` for the async client, which is awaited rather than called."""

    async def __call__(self, **kwargs: Any) -> Any:  # type: ignore[override]
        self.requests.append(kwargs)
        return self.response


class _ProviderContract:
    """Adapter from one native provider's SDK surface to the shared contract.

    Subclasses supply the fake SDK response and the client attribute each
    provider actually calls; everything the tests assert is provider-agnostic.
    """

    id: str
    model: str

    def build(self, **kwargs: Any) -> BaseLLM:
        return LLM(model=self.model, **kwargs)

    def response(self, text: str | None) -> Any:
        """A fake SDK response carrying ``text``, or no assistant text at all."""
        raise NotImplementedError

    def stub_sync(
        self, llm: BaseLLM, monkeypatch: pytest.MonkeyPatch, text: str | None
    ) -> _Recorder:
        raise NotImplementedError

    def stub_async(
        self, llm: BaseLLM, monkeypatch: pytest.MonkeyPatch, text: str | None
    ) -> _AsyncRecorder:
        raise NotImplementedError

    @staticmethod
    def user_texts(request: dict[str, Any]) -> list[str]:
        """Text of every ``user`` message in a recorded request."""
        return [
            msg["content"]
            for msg in request["messages"]
            if msg.get("role") == "user"
        ]

    @staticmethod
    def system_text(request: dict[str, Any]) -> str:
        """System instructions wherever this provider puts them."""
        raise NotImplementedError


class _OpenAIRawResponse:
    """Stands in for the wrapper returned by ``with_raw_response.create``."""

    def __init__(self, completion: Any) -> None:
        self._completion = completion
        self.text = completion.model_dump_json()
        self.http_response = httpx.Response(
            200, request=httpx.Request("POST", "https://api.openai.com/v1/")
        )

    def parse(self) -> Any:
        return self._completion


class _OpenAIContract(_ProviderContract):
    id = "openai"
    model = "gpt-4o-mini"

    def response(self, text: str | None) -> Any:
        from openai.types.chat import ChatCompletion, ChatCompletionMessage
        from openai.types.chat.chat_completion import Choice
        from openai.types.completion_usage import CompletionUsage

        return _OpenAIRawResponse(
            ChatCompletion(
                id="chatcmpl-contract",
                created=0,
                model=self.model,
                object="chat.completion",
                choices=[
                    Choice(
                        index=0,
                        finish_reason="stop",
                        # OpenAI reports "no assistant text" as a null content.
                        message=ChatCompletionMessage(role="assistant", content=text),
                    )
                ],
                usage=CompletionUsage(
                    prompt_tokens=11, completion_tokens=5, total_tokens=16
                ),
            )
        )

    def stub_sync(
        self, llm: BaseLLM, monkeypatch: pytest.MonkeyPatch, text: str | None
    ) -> _Recorder:
        recorder = _Recorder(self.response(text))
        monkeypatch.setattr(
            llm._client.chat.completions.with_raw_response, "create", recorder
        )
        return recorder

    def stub_async(
        self, llm: BaseLLM, monkeypatch: pytest.MonkeyPatch, text: str | None
    ) -> _AsyncRecorder:
        recorder = _AsyncRecorder(self.response(text))
        monkeypatch.setattr(
            llm._async_client.chat.completions.with_raw_response, "create", recorder
        )
        return recorder

    @staticmethod
    def system_text(request: dict[str, Any]) -> str:
        return "".join(
            msg["content"]
            for msg in request["messages"]
            if msg.get("role") == "system"
        )


class _AnthropicContract(_ProviderContract):
    id = "anthropic"
    model = "anthropic/claude-sonnet-4-0"

    def response(self, text: str | None) -> Any:
        from anthropic.types import Message, TextBlock, Usage

        return Message(
            id="msg_contract",
            type="message",
            role="assistant",
            model="claude-sonnet-4-0",
            # Anthropic reports "no assistant text" as an empty block list,
            # rather than as OpenAI's null content.
            content=[] if text is None else [TextBlock(type="text", text=text)],
            stop_reason="end_turn",
            stop_sequence=None,
            usage=Usage(input_tokens=11, output_tokens=5),
        )

    def stub_sync(
        self, llm: BaseLLM, monkeypatch: pytest.MonkeyPatch, text: str | None
    ) -> _Recorder:
        recorder = _Recorder(self.response(text))
        monkeypatch.setattr(llm._client.messages, "create", recorder)
        return recorder

    def stub_async(
        self, llm: BaseLLM, monkeypatch: pytest.MonkeyPatch, text: str | None
    ) -> _AsyncRecorder:
        recorder = _AsyncRecorder(self.response(text))
        monkeypatch.setattr(llm._async_client.messages, "create", recorder)
        return recorder

    @staticmethod
    def system_text(request: dict[str, Any]) -> str:
        system = request.get("system")
        if isinstance(system, list):
            return "".join(
                block.get("text", "") for block in system if isinstance(block, dict)
            )
        return system or ""


# Anthropic's native provider is an optional extra; skip rather than fail the
# whole module when it is not installed, as tests/llms/test_multimodal.py does.
PROVIDERS = [
    pytest.param(_OpenAIContract(), id="openai"),
    pytest.param(
        _AnthropicContract(),
        id="anthropic",
        marks=pytest.mark.skipif(
            not HAS_ANTHROPIC, reason="Anthropic SDK not installed"
        ),
    ),
]

# Recorded difference: OpenAI leaves the system prompt in the message list;
# Anthropic hoists it into a top-level `system` parameter, so its message list
# carries only the conversation turns.
_SYSTEM_IN_MESSAGES = {"openai": True, "anthropic": False}

PROMPT = "Name the capital of Italy."
ANSWER = "Rome."


@pytest.mark.parametrize("provider", PROVIDERS)
def test_plain_text_prompt_is_sent_as_one_user_message(
    provider: _ProviderContract, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A bare string prompt reaches the provider once, verbatim, as one user turn."""
    llm = provider.build()
    recorder = provider.stub_sync(llm, monkeypatch, ANSWER)

    llm.call(PROMPT)

    assert provider.user_texts(recorder.request) == [PROMPT]
    assert recorder.request["model"] == llm.model


@pytest.mark.parametrize("provider", PROVIDERS)
def test_assistant_text_is_returned_verbatim(
    provider: _ProviderContract, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The return value is the assistant text itself, not a container or repr."""
    llm = provider.build()
    provider.stub_sync(llm, monkeypatch, ANSWER)

    result = llm.call(PROMPT)

    assert result == ANSWER
    assert isinstance(result, str)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_missing_assistant_text_becomes_an_empty_string(
    provider: _ProviderContract, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Callers always get a string back, even when the model returned no text.

    The two providers express "no text" differently -- a null ``content`` for
    OpenAI, an empty block list for Anthropic -- and both must collapse to ``""``.
    """
    llm = provider.build()
    provider.stub_sync(llm, monkeypatch, None)

    assert llm.call(PROMPT) == ""


@pytest.mark.parametrize("provider", PROVIDERS)
def test_system_prompt_is_delivered_where_the_provider_expects_it(
    provider: _ProviderContract, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A system message is forwarded; where it lands is a provider difference."""
    llm = provider.build()
    recorder = provider.stub_sync(llm, monkeypatch, ANSWER)

    llm.call(
        [
            {"role": "system", "content": "You are terse."},
            {"role": "user", "content": PROMPT},
        ]
    )

    request = recorder.request
    assert provider.system_text(request) == "You are terse."
    assert provider.user_texts(request) == [PROMPT]
    assert (
        any(msg.get("role") == "system" for msg in request["messages"])
        is _SYSTEM_IN_MESSAGES[provider.id]
    )


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.asyncio
async def test_acall_returns_what_call_returns(
    provider: _ProviderContract, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``acall`` is sync parity over the async client, not a different contract."""
    sync_llm = provider.build()
    async_llm = provider.build()
    sync_recorder = provider.stub_sync(sync_llm, monkeypatch, ANSWER)
    async_recorder = provider.stub_async(async_llm, monkeypatch, ANSWER)

    sync_result = sync_llm.call(PROMPT)
    async_result = await async_llm.acall(PROMPT)

    assert async_result == sync_result == ANSWER
    assert provider.user_texts(async_recorder.request) == provider.user_texts(
        sync_recorder.request
    )


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.asyncio
async def test_acall_does_not_use_the_sync_client(
    provider: _ProviderContract, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Blocking the event loop on the sync client is a bug, so prove it is unused."""
    llm = provider.build()
    sync_recorder = provider.stub_sync(llm, monkeypatch, ANSWER)
    async_recorder = provider.stub_async(llm, monkeypatch, ANSWER)

    await llm.acall(PROMPT)

    assert len(async_recorder.requests) == 1
    assert sync_recorder.requests == []


def _capture_call_events() -> tuple[list[LLMCallStartedEvent], list[Any]]:
    started: list[LLMCallStartedEvent] = []
    completed: list[Any] = []

    @crewai_event_bus.on(LLMCallStartedEvent)
    def _on_started(_source: Any, event: LLMCallStartedEvent) -> None:
        started.append(event)

    @crewai_event_bus.on(LLMCallCompletedEvent)
    def _on_completed(_source: Any, event: LLMCallCompletedEvent) -> None:
        completed.append(event)

    return started, completed


def _assert_one_completed_llm_call(
    started: list[LLMCallStartedEvent],
    completed: list[Any],
    model: str,
) -> None:
    assert len(started) == 1
    assert len(completed) == 1
    assert started[0].model == model
    assert completed[0].model == model
    assert completed[0].response == ANSWER
    assert completed[0].call_type == LLMCallType.LLM_CALL


@pytest.mark.parametrize("provider", PROVIDERS)
def test_call_emits_started_and_completed_events(
    provider: _ProviderContract, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The public event stream reports the call once, with the returned text."""
    llm = provider.build()
    provider.stub_sync(llm, monkeypatch, ANSWER)

    with crewai_event_bus.scoped_handlers():
        started, completed = _capture_call_events()
        llm.call(PROMPT)
        # Sync handlers run in a thread pool, so the completed event can still
        # be in flight when the call returns.
        crewai_event_bus.flush()

    _assert_one_completed_llm_call(started, completed, llm.model)


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.asyncio
async def test_acall_emits_the_same_events_as_call(
    provider: _ProviderContract, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Async callers see the same event contract as sync callers."""
    llm = provider.build()
    provider.stub_async(llm, monkeypatch, ANSWER)

    with crewai_event_bus.scoped_handlers():
        started, completed = _capture_call_events()
        await llm.acall(PROMPT)
        crewai_event_bus.flush()

    _assert_one_completed_llm_call(started, completed, llm.model)
