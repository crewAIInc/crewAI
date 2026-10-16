"""Shared behavioral contract for native providers: tools, streaming, usage, failures.

Issue #8006, milestone 3. The plain-text slice (milestone 1) and the
``LLMMessage.content`` shapes (milestone 2) are covered elsewhere; this module
covers what happens around them, for OpenAI (Chat Completions) and Anthropic,
driven by fake SDK responses so the tests need no API key and make no paid call.

The contract asserted here:

1. Tool calls. Without ``available_functions`` a tool-calling response is
   returned to the caller as one entry per call, in order, each carrying the
   provider's call id, the tool name and the arguments. With
   ``available_functions`` the named function runs once with the parsed
   arguments and its result is returned.
2. Tool results. A ``tool`` message is sent back paired with the id of the
   call it answers.
3. Streaming. A streamed answer is returned whole, and every text delta is
   emitted as one ``LLMStreamChunkEvent`` in order. A streamed tool call is
   returned like a non-streamed one, and its chunk events share one id while
   the arguments accumulate.
4. Token usage. Each successful call adds its prompt and completion tokens to
   ``get_token_usage_summary()`` and reports them on ``LLMCallCompletedEvent``.
5. Failures. A failed logical call raises, emits exactly one
   ``LLMCallFailedEvent`` and no ``LLMCallCompletedEvent``, and counts no usage.

Two cases fail on current ``main`` and are marked as strict known failures
against the issue that reports the defect (#7783), so they turn into errors as
soon as it is fixed and the mark is left behind. A third defect these tests
exposed, #8060, is already fixed and its case now passes.

Legitimate differences are recorded on the provider adapters rather than hidden
behind a shared assertion: the shape of a returned tool call (``tool_calls``),
and where a tool result lives in the request (``tool_result_ids``).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
import json
import os
from typing import Any
from unittest.mock import patch

import pytest

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallStartedEvent,
    LLMCallType,
    LLMStreamChunkEvent,
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


PROMPT = "What is the temperature in Rome and in Oslo?"
ANSWER_CHUNKS = ["Rome is ", "warmer ", "than Oslo."]
ANSWER = "".join(ANSWER_CHUNKS)
PROMPT_TOKENS = 11
COMPLETION_TOKENS = 5

TOOL_NAME = "get_current_temperature"
TOOL_SCHEMA: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": TOOL_NAME,
        "description": "Get the current temperature in a city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    },
}
# (call id, tool name, arguments) for each call the fake model makes.
ToolCallSpec = tuple[str, str, dict[str, Any]]
ONE_CALL: list[ToolCallSpec] = [("call_rome", TOOL_NAME, {"city": "Rome"})]
TWO_CALLS: list[ToolCallSpec] = [
    ("call_rome", TOOL_NAME, {"city": "Rome"}),
    ("call_oslo", TOOL_NAME, {"city": "Oslo"}),
]


@pytest.fixture(autouse=True)
def fake_api_keys() -> Iterator[None]:
    """Give every provider a syntactically valid key so no real one is needed."""
    with patch.dict(
        os.environ,
        {"OPENAI_API_KEY": "test-key", "ANTHROPIC_API_KEY": "test-key"},
    ):
        yield


class _Recorder:
    """Stands in for a provider SDK call, recording the request it received.

    ``outcome`` is returned, or raised when it is an exception.
    """

    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.requests: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.requests.append(kwargs)
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        return self.outcome

    @property
    def request(self) -> dict[str, Any]:
        assert len(self.requests) == 1, (
            f"expected exactly one provider request, got {len(self.requests)}"
        )
        return self.requests[0]


class _AsyncRecorder(_Recorder):
    """``_Recorder`` for the async client, which is awaited rather than called."""

    async def __call__(self, **kwargs: Any) -> Any:  # type: ignore[override]
        return super().__call__(**kwargs)


class _AsyncChunks:
    """An async iterator over prepared stream chunks."""

    def __init__(self, chunks: list[Any]) -> None:
        self._chunks = iter(chunks)

    def __aiter__(self) -> AsyncIterator[Any]:
        return self

    async def __anext__(self) -> Any:
        try:
            return next(self._chunks)
        except StopIteration:
            raise StopAsyncIteration from None


class _ProviderContract:
    """Adapter from one native provider's SDK surface to the shared contract.

    Subclasses supply the fake SDK responses and the client attribute each
    provider actually calls; everything the tests assert is provider-agnostic.
    """

    id: str
    model: str

    def build(self, **kwargs: Any) -> BaseLLM:
        return LLM(model=self.model, **kwargs)

    def text_response(self) -> Any:
        """A fake non-streaming response answering with ``ANSWER``."""
        raise NotImplementedError

    def tool_response(self, calls: list[ToolCallSpec]) -> Any:
        """A fake non-streaming response requesting ``calls``."""
        raise NotImplementedError

    def stub(
        self, llm: BaseLLM, monkeypatch: pytest.MonkeyPatch, outcome: Any
    ) -> _Recorder:
        """Replace the sync non-streaming SDK call."""
        raise NotImplementedError

    def stub_async(
        self, llm: BaseLLM, monkeypatch: pytest.MonkeyPatch, outcome: Any
    ) -> _AsyncRecorder:
        """Replace the async non-streaming SDK call."""
        raise NotImplementedError

    def stub_text_stream(
        self, llm: BaseLLM, monkeypatch: pytest.MonkeyPatch
    ) -> _Recorder:
        """Replace the sync streaming SDK call with one yielding ``ANSWER_CHUNKS``."""
        raise NotImplementedError

    def stub_text_stream_async(
        self, llm: BaseLLM, monkeypatch: pytest.MonkeyPatch
    ) -> _Recorder:
        """Async counterpart of ``stub_text_stream``."""
        raise NotImplementedError

    def stub_tool_stream(
        self, llm: BaseLLM, monkeypatch: pytest.MonkeyPatch, call: ToolCallSpec
    ) -> _Recorder:
        """Replace the sync streaming SDK call with one streaming a tool call."""
        raise NotImplementedError

    def stub_failing_stream(
        self, llm: BaseLLM, monkeypatch: pytest.MonkeyPatch, error: Exception
    ) -> _Recorder:
        """Replace the sync streaming SDK call with one that raises ``error``."""
        raise NotImplementedError

    @staticmethod
    def tool_calls(result: Any) -> list[ToolCallSpec]:
        """Normalize the tool calls a provider returned to the caller."""
        raise NotImplementedError

    @staticmethod
    def request_tool_names(request: dict[str, Any]) -> list[str]:
        """Names of the tools offered to the model in a recorded request."""
        raise NotImplementedError

    @staticmethod
    def tool_call_ids(request: dict[str, Any]) -> list[str]:
        """Ids of the assistant tool calls replayed in a recorded request."""
        raise NotImplementedError

    @staticmethod
    def tool_result_ids(request: dict[str, Any]) -> list[str]:
        """Ids the tool results in a recorded request are paired with."""
        raise NotImplementedError


class _OpenAIRawResponse:
    """Stands in for the wrapper returned by ``with_raw_response.create``."""

    def __init__(self, completion: Any) -> None:
        import httpx

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

    def _completion(self, message: Any, finish_reason: str) -> _OpenAIRawResponse:
        from openai.types.chat import ChatCompletion
        from openai.types.chat.chat_completion import Choice
        from openai.types.completion_usage import CompletionUsage

        return _OpenAIRawResponse(
            ChatCompletion(
                id="chatcmpl-contract",
                created=0,
                model=self.model,
                object="chat.completion",
                choices=[Choice(index=0, finish_reason=finish_reason, message=message)],
                usage=CompletionUsage(
                    prompt_tokens=PROMPT_TOKENS,
                    completion_tokens=COMPLETION_TOKENS,
                    total_tokens=PROMPT_TOKENS + COMPLETION_TOKENS,
                ),
            )
        )

    def text_response(self) -> Any:
        from openai.types.chat import ChatCompletionMessage

        return self._completion(
            ChatCompletionMessage(role="assistant", content=ANSWER), "stop"
        )

    def tool_response(self, calls: list[ToolCallSpec]) -> Any:
        from openai.types.chat import ChatCompletionMessage

        return self._completion(
            ChatCompletionMessage.model_validate(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": name, "arguments": json.dumps(args)},
                        }
                        for call_id, name, args in calls
                    ],
                }
            ),
            "tool_calls",
        )

    def stub(
        self, llm: BaseLLM, monkeypatch: pytest.MonkeyPatch, outcome: Any
    ) -> _Recorder:
        recorder = _Recorder(outcome)
        monkeypatch.setattr(
            llm._get_sync_client().chat.completions.with_raw_response,
            "create",
            recorder,
        )
        return recorder

    def stub_async(
        self, llm: BaseLLM, monkeypatch: pytest.MonkeyPatch, outcome: Any
    ) -> _AsyncRecorder:
        recorder = _AsyncRecorder(outcome)
        monkeypatch.setattr(
            llm._get_async_client().chat.completions.with_raw_response,
            "create",
            recorder,
        )
        return recorder

    def _chunk(self, delta: dict[str, Any] | None, **extra: Any) -> Any:
        from openai.types.chat import ChatCompletionChunk

        return ChatCompletionChunk.model_validate(
            {
                "id": "chatcmpl-contract",
                "created": 0,
                "model": self.model,
                "object": "chat.completion.chunk",
                "choices": []
                if delta is None
                else [
                    {
                        "index": 0,
                        "delta": delta,
                        "finish_reason": extra.pop("finish_reason", None),
                    }
                ],
                **extra,
            }
        )

    def _usage_chunk(self) -> Any:
        # OpenAI reports streaming usage on a final chunk with no choices.
        return self._chunk(
            None,
            usage={
                "prompt_tokens": PROMPT_TOKENS,
                "completion_tokens": COMPLETION_TOKENS,
                "total_tokens": PROMPT_TOKENS + COMPLETION_TOKENS,
            },
        )

    def _text_chunks(self) -> list[Any]:
        return [
            *(self._chunk({"content": text}) for text in ANSWER_CHUNKS),
            self._chunk({}, finish_reason="stop"),
            self._usage_chunk(),
        ]

    def stub_text_stream(
        self, llm: BaseLLM, monkeypatch: pytest.MonkeyPatch
    ) -> _Recorder:
        recorder = _Recorder(iter(self._text_chunks()))
        monkeypatch.setattr(llm._get_sync_client().chat.completions, "create", recorder)
        return recorder

    def stub_text_stream_async(
        self, llm: BaseLLM, monkeypatch: pytest.MonkeyPatch
    ) -> _Recorder:
        recorder = _AsyncRecorder(_AsyncChunks(self._text_chunks()))
        monkeypatch.setattr(
            llm._get_async_client().chat.completions, "create", recorder
        )
        return recorder

    def stub_tool_stream(
        self, llm: BaseLLM, monkeypatch: pytest.MonkeyPatch, call: ToolCallSpec
    ) -> _Recorder:
        call_id, name, args = call
        arguments = json.dumps(args)
        half = len(arguments) // 2

        def tool_delta(**function: Any) -> dict[str, Any]:
            return {"tool_calls": [{"index": 0, "function": function}]}

        first = tool_delta(name=name, arguments="")
        first["tool_calls"][0].update({"id": call_id, "type": "function"})
        recorder = _Recorder(
            iter(
                [
                    self._chunk(first),
                    self._chunk(tool_delta(arguments=arguments[:half])),
                    self._chunk(tool_delta(arguments=arguments[half:])),
                    self._chunk({}, finish_reason="tool_calls"),
                    self._usage_chunk(),
                ]
            )
        )
        monkeypatch.setattr(llm._get_sync_client().chat.completions, "create", recorder)
        return recorder

    def stub_failing_stream(
        self, llm: BaseLLM, monkeypatch: pytest.MonkeyPatch, error: Exception
    ) -> _Recorder:
        recorder = _Recorder(error)
        monkeypatch.setattr(llm._get_sync_client().chat.completions, "create", recorder)
        return recorder

    @staticmethod
    def tool_calls(result: Any) -> list[ToolCallSpec]:
        # Recorded difference: the non-streaming path returns the SDK's tool
        # call objects, the streaming path returns plain dicts.
        calls: list[ToolCallSpec] = []
        for call in result:
            if isinstance(call, dict):
                function = call["function"]
                calls.append(
                    (call["id"], function["name"], json.loads(function["arguments"]))
                )
            else:
                calls.append(
                    (
                        call.id,
                        call.function.name,
                        json.loads(call.function.arguments),
                    )
                )
        return calls

    @staticmethod
    def request_tool_names(request: dict[str, Any]) -> list[str]:
        return [tool["function"]["name"] for tool in request["tools"]]

    @staticmethod
    def tool_call_ids(request: dict[str, Any]) -> list[str]:
        return [
            call["id"]
            for msg in request["messages"]
            if msg.get("role") == "assistant"
            for call in msg.get("tool_calls") or []
        ]

    @staticmethod
    def tool_result_ids(request: dict[str, Any]) -> list[str]:
        # OpenAI keeps a tool result as its own ``tool`` message.
        return [
            msg["tool_call_id"]
            for msg in request["messages"]
            if msg.get("role") == "tool"
        ]


class _AnthropicStream:
    """Stands in for the manager returned by ``messages.stream``.

    Works as both the sync and the async context manager, since the provider
    uses the same event and final-message shapes on both clients.
    """

    def __init__(self, events: list[Any], final_message: Any) -> None:
        self._events = events
        self._final_message = final_message

    def __enter__(self) -> _AnthropicStream:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def __iter__(self) -> Iterator[Any]:
        return iter(self._events)

    def get_final_message(self) -> Any:
        return self._final_message


class _AnthropicAsyncStream(_AnthropicStream):
    async def __aenter__(self) -> _AnthropicAsyncStream:
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None

    def __aiter__(self) -> AsyncIterator[Any]:
        return _AsyncChunks(self._events)

    async def get_final_message(self) -> Any:  # type: ignore[override]
        return self._final_message


class _AnthropicContract(_ProviderContract):
    id = "anthropic"
    model = "anthropic/claude-sonnet-4-0"

    def _message(self, content: list[Any], stop_reason: str) -> Any:
        from anthropic.types import Message, Usage

        return Message(
            id="msg_contract",
            type="message",
            role="assistant",
            model="claude-sonnet-4-0",
            content=content,
            stop_reason=stop_reason,
            stop_sequence=None,
            usage=Usage(input_tokens=PROMPT_TOKENS, output_tokens=COMPLETION_TOKENS),
        )

    def text_response(self) -> Any:
        from anthropic.types import TextBlock

        return self._message([TextBlock(type="text", text=ANSWER)], "end_turn")

    def tool_response(self, calls: list[ToolCallSpec]) -> Any:
        from anthropic.types import ToolUseBlock

        return self._message(
            [
                ToolUseBlock(type="tool_use", id=call_id, name=name, input=args)
                for call_id, name, args in calls
            ],
            "tool_use",
        )

    def stub(
        self, llm: BaseLLM, monkeypatch: pytest.MonkeyPatch, outcome: Any
    ) -> _Recorder:
        recorder = _Recorder(outcome)
        monkeypatch.setattr(llm._get_sync_client().messages, "create", recorder)
        return recorder

    def stub_async(
        self, llm: BaseLLM, monkeypatch: pytest.MonkeyPatch, outcome: Any
    ) -> _AsyncRecorder:
        recorder = _AsyncRecorder(outcome)
        monkeypatch.setattr(llm._get_async_client().messages, "create", recorder)
        return recorder

    def _text_events(self) -> list[Any]:
        from anthropic.types import (
            RawContentBlockDeltaEvent,
            RawMessageStartEvent,
            TextDelta,
        )

        return [
            RawMessageStartEvent(
                type="message_start", message=self._message([], "end_turn")
            ),
            *(
                RawContentBlockDeltaEvent(
                    type="content_block_delta",
                    index=0,
                    delta=TextDelta(type="text_delta", text=text),
                )
                for text in ANSWER_CHUNKS
            ),
        ]

    def stub_text_stream(
        self, llm: BaseLLM, monkeypatch: pytest.MonkeyPatch
    ) -> _Recorder:
        recorder = _Recorder(
            _AnthropicStream(self._text_events(), self.text_response())
        )
        monkeypatch.setattr(llm._get_sync_client().messages, "stream", recorder)
        return recorder

    def stub_text_stream_async(
        self, llm: BaseLLM, monkeypatch: pytest.MonkeyPatch
    ) -> _Recorder:
        # ``messages.stream`` is called, not awaited, on the async client too.
        recorder = _Recorder(
            _AnthropicAsyncStream(self._text_events(), self.text_response())
        )
        monkeypatch.setattr(llm._get_async_client().messages, "stream", recorder)
        return recorder

    def stub_tool_stream(
        self, llm: BaseLLM, monkeypatch: pytest.MonkeyPatch, call: ToolCallSpec
    ) -> _Recorder:
        from anthropic.types import (
            InputJSONDelta,
            RawContentBlockDeltaEvent,
            RawContentBlockStartEvent,
            RawMessageStartEvent,
            ToolUseBlock,
        )

        call_id, name, args = call
        arguments = json.dumps(args)
        half = len(arguments) // 2
        events = [
            RawMessageStartEvent(
                type="message_start", message=self._message([], "tool_use")
            ),
            RawContentBlockStartEvent(
                type="content_block_start",
                index=0,
                content_block=ToolUseBlock(
                    type="tool_use", id=call_id, name=name, input={}
                ),
            ),
            *(
                RawContentBlockDeltaEvent(
                    type="content_block_delta",
                    index=0,
                    delta=InputJSONDelta(type="input_json_delta", partial_json=part),
                )
                for part in (arguments[:half], arguments[half:])
            ),
        ]
        recorder = _Recorder(_AnthropicStream(events, self.tool_response([call])))
        monkeypatch.setattr(llm._get_sync_client().messages, "stream", recorder)
        return recorder

    def stub_failing_stream(
        self, llm: BaseLLM, monkeypatch: pytest.MonkeyPatch, error: Exception
    ) -> _Recorder:
        recorder = _Recorder(error)
        monkeypatch.setattr(llm._get_sync_client().messages, "stream", recorder)
        return recorder

    @staticmethod
    def tool_calls(result: Any) -> list[ToolCallSpec]:
        # Anthropic returns its tool-use blocks, whose arguments are already a
        # dict rather than OpenAI's JSON string.
        return [(block.id, block.name, block.input) for block in result]

    @staticmethod
    def request_tool_names(request: dict[str, Any]) -> list[str]:
        return [tool["name"] for tool in request["tools"]]

    @staticmethod
    def _blocks(
        request: dict[str, Any], role: str, block_type: str
    ) -> list[dict[str, Any]]:
        return [
            block
            for msg in request["messages"]
            if msg.get("role") == role and isinstance(msg.get("content"), list)
            for block in msg["content"]
            if isinstance(block, dict) and block.get("type") == block_type
        ]

    @classmethod
    def tool_call_ids(cls, request: dict[str, Any]) -> list[str]:  # type: ignore[override]
        return [b["id"] for b in cls._blocks(request, "assistant", "tool_use")]

    @classmethod
    def tool_result_ids(cls, request: dict[str, Any]) -> list[str]:  # type: ignore[override]
        # Anthropic has no ``tool`` role: a tool result is a ``tool_result``
        # block inside a user turn.
        return [b["tool_use_id"] for b in cls._blocks(request, "user", "tool_result")]


# Anthropic's native provider is an optional extra; skip rather than fail the
# whole module when it is not installed, as tests/llms/test_multimodal.py does.
_ANTHROPIC_MARKS = [
    pytest.mark.skipif(not HAS_ANTHROPIC, reason="Anthropic SDK not installed")
]


def _providers(**known_failures: str) -> list[Any]:
    """Both providers, with ``provider_id="reason"`` marking a known failure.

    Known failures are strict, so the mark has to be removed in the same change
    that fixes the defect it points at.
    """
    return [
        pytest.param(
            contract,
            id=contract.id,
            marks=[
                *marks,
                *(
                    [pytest.mark.xfail(reason=known_failures[contract.id], strict=True)]
                    if contract.id in known_failures
                    else []
                ),
            ],
        )
        for contract, marks in (
            (_OpenAIContract(), []),
            (_AnthropicContract(), _ANTHROPIC_MARKS),
        )
    ]


PROVIDERS = _providers()

# A defect these tests exposed that is still open.
_OPENAI_FAILS_TWICE = (
    "#7783: the non-streaming OpenAI handlers emit LLMCallFailedEvent and "
    "call()/acall() emit it again"
)


class _Events:
    """The LLM events one call emitted, collected through the public event bus."""

    def __init__(self) -> None:
        self.started: list[LLMCallStartedEvent] = []
        self.completed: list[LLMCallCompletedEvent] = []
        self.failed: list[LLMCallFailedEvent] = []
        self.chunks: list[LLMStreamChunkEvent] = []

    def listen(self) -> None:
        for event_type, bucket in (
            (LLMCallStartedEvent, self.started),
            (LLMCallCompletedEvent, self.completed),
            (LLMCallFailedEvent, self.failed),
            (LLMStreamChunkEvent, self.chunks),
        ):
            crewai_event_bus.on(event_type)(self._collector(bucket))

    @staticmethod
    def _collector(bucket: list[Any]) -> Any:
        def collect(_source: Any, event: Any) -> None:
            bucket.append(event)

        return collect

    @property
    def text_chunks(self) -> list[str]:
        return [c.chunk for c in self.chunks if c.call_type != LLMCallType.TOOL_CALL]

    @property
    def tool_chunks(self) -> list[LLMStreamChunkEvent]:
        return [c for c in self.chunks if c.call_type == LLMCallType.TOOL_CALL]


@pytest.fixture
def events() -> Iterator[_Events]:
    """Collect LLM events for the duration of one test."""
    collected = _Events()
    with crewai_event_bus.scoped_handlers():
        collected.listen()
        yield collected


def _flush() -> None:
    # Sync handlers run in a thread pool, so an event can still be in flight
    # when the call returns.
    crewai_event_bus.flush()


# --- Tool calls ---------------------------------------------------------------


@pytest.mark.parametrize("provider", PROVIDERS)
def test_tool_calls_are_returned_with_id_name_and_arguments(
    provider: _ProviderContract, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without ``available_functions`` the caller gets every call, in order."""
    llm = provider.build()
    recorder = provider.stub(llm, monkeypatch, provider.tool_response(TWO_CALLS))

    result = llm.call(PROMPT, tools=[TOOL_SCHEMA])

    assert provider.tool_calls(result) == TWO_CALLS
    assert provider.request_tool_names(recorder.request) == [TOOL_NAME]


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.asyncio
async def test_acall_returns_the_same_tool_calls_as_call(
    provider: _ProviderContract, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Async callers get the same tool calls, through the async client."""
    llm = provider.build()
    sync_recorder = provider.stub(llm, monkeypatch, provider.tool_response(TWO_CALLS))
    async_recorder = provider.stub_async(
        llm, monkeypatch, provider.tool_response(TWO_CALLS)
    )

    result = await llm.acall(PROMPT, tools=[TOOL_SCHEMA])

    assert provider.tool_calls(result) == TWO_CALLS
    assert len(async_recorder.requests) == 1
    assert sync_recorder.requests == []


@pytest.mark.parametrize("provider", PROVIDERS)
def test_returned_tool_calls_are_reported_as_one_tool_call_event(
    provider: _ProviderContract, monkeypatch: pytest.MonkeyPatch, events: _Events
) -> None:
    """A response that asks for tools completes the call once, as a tool call."""
    llm = provider.build()
    provider.stub(llm, monkeypatch, provider.tool_response(ONE_CALL))

    llm.call(PROMPT, tools=[TOOL_SCHEMA])
    _flush()

    assert len(events.started) == 1
    assert [e.call_type for e in events.completed] == [LLMCallType.TOOL_CALL]
    assert events.failed == []


@pytest.mark.parametrize("provider", PROVIDERS)
def test_available_function_runs_once_with_the_parsed_arguments(
    provider: _ProviderContract, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With ``available_functions`` the tool runs and its result is returned."""
    llm = provider.build()
    provider.stub(llm, monkeypatch, provider.tool_response(ONE_CALL))
    received: list[dict[str, Any]] = []

    def get_current_temperature(**kwargs: Any) -> str:
        received.append(kwargs)
        return "21C"

    result = llm.call(
        PROMPT,
        tools=[TOOL_SCHEMA],
        available_functions={TOOL_NAME: get_current_temperature},
    )

    assert result == "21C"
    assert received == [{"city": "Rome"}]


@pytest.mark.parametrize("provider", PROVIDERS)
def test_tool_result_is_sent_back_paired_with_its_call_id(
    provider: _ProviderContract, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each tool result names the call it answers; where it lives is recorded
    on the adapter (a ``tool`` message for OpenAI, a user-turn block for
    Anthropic)."""
    llm = provider.build()
    recorder = provider.stub(llm, monkeypatch, provider.text_response())

    llm.call(
        [
            {"role": "user", "content": PROMPT},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }
                    for call_id, name, args in TWO_CALLS
                ],
            },
            {"role": "tool", "tool_call_id": "call_rome", "content": "21C"},
            {"role": "tool", "tool_call_id": "call_oslo", "content": "4C"},
        ],
        tools=[TOOL_SCHEMA],
    )

    request = recorder.request
    assert provider.tool_call_ids(request) == ["call_rome", "call_oslo"]
    assert provider.tool_result_ids(request) == ["call_rome", "call_oslo"]


# --- Streaming ----------------------------------------------------------------


@pytest.mark.parametrize("provider", PROVIDERS)
def test_streamed_text_is_returned_whole_and_emitted_in_order(
    provider: _ProviderContract, monkeypatch: pytest.MonkeyPatch, events: _Events
) -> None:
    """Every text delta is one chunk event, and the chunks add up to the answer."""
    llm = provider.build(stream=True)
    provider.stub_text_stream(llm, monkeypatch)

    result = llm.call(PROMPT)
    _flush()

    assert result == ANSWER
    assert events.text_chunks == ANSWER_CHUNKS
    assert events.tool_chunks == []


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.asyncio
async def test_async_streamed_text_matches_the_sync_stream(
    provider: _ProviderContract, monkeypatch: pytest.MonkeyPatch, events: _Events
) -> None:
    """``acall`` streams the same chunks and returns the same answer."""
    llm = provider.build(stream=True)
    provider.stub_text_stream_async(llm, monkeypatch)

    result = await llm.acall(PROMPT)
    _flush()

    assert result == ANSWER
    assert events.text_chunks == ANSWER_CHUNKS


@pytest.mark.parametrize("provider", PROVIDERS)
def test_streamed_text_completes_the_call_once_with_the_full_answer(
    provider: _ProviderContract, monkeypatch: pytest.MonkeyPatch, events: _Events
) -> None:
    """Streaming does not change the started/completed contract."""
    llm = provider.build(stream=True)
    provider.stub_text_stream(llm, monkeypatch)

    llm.call(PROMPT)
    _flush()

    assert len(events.started) == 1
    assert [(e.call_type, e.response) for e in events.completed] == [
        (LLMCallType.LLM_CALL, ANSWER)
    ]


@pytest.mark.parametrize("provider", PROVIDERS)
def test_streamed_tool_call_is_returned_like_a_non_streamed_one(
    provider: _ProviderContract, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Arguments streamed in pieces come back as one complete call."""
    llm = provider.build(stream=True)
    provider.stub_tool_stream(llm, monkeypatch, ONE_CALL[0])

    result = llm.call(PROMPT, tools=[TOOL_SCHEMA])

    assert provider.tool_calls(result) == ONE_CALL


@pytest.mark.parametrize("provider", PROVIDERS)
def test_streamed_tool_call_chunks_share_one_id_and_accumulate_arguments(
    provider: _ProviderContract, monkeypatch: pytest.MonkeyPatch, events: _Events
) -> None:
    """Consumers can follow one tool call across its chunk events."""
    call_id, name, args = ONE_CALL[0]
    llm = provider.build(stream=True)
    provider.stub_tool_stream(llm, monkeypatch, ONE_CALL[0])

    llm.call(PROMPT, tools=[TOOL_SCHEMA])
    _flush()

    tool_calls = [chunk.tool_call for chunk in events.tool_chunks]
    assert tool_calls
    assert all(tc is not None for tc in tool_calls)
    assert {tc.id for tc in tool_calls if tc} == {call_id}
    assert {tc.function.name for tc in tool_calls if tc} == {name}
    accumulated = [tc.function.arguments for tc in tool_calls if tc]
    assert accumulated == sorted(accumulated, key=len)
    assert json.loads(accumulated[-1]) == args
    assert events.text_chunks == []


@pytest.mark.parametrize("provider", PROVIDERS)
def test_streamed_tool_call_is_reported_as_one_tool_call_event(
    provider: _ProviderContract, monkeypatch: pytest.MonkeyPatch, events: _Events
) -> None:
    """A streamed tool call completes the call once, as the non-streamed one does."""
    llm = provider.build(stream=True)
    provider.stub_tool_stream(llm, monkeypatch, ONE_CALL[0])

    llm.call(PROMPT, tools=[TOOL_SCHEMA])
    _flush()

    assert len(events.started) == 1
    assert [e.call_type for e in events.completed] == [LLMCallType.TOOL_CALL]


# --- Token usage --------------------------------------------------------------


def _assert_usage(llm: BaseLLM, calls: int) -> None:
    usage = llm.get_token_usage_summary()
    assert usage.prompt_tokens == PROMPT_TOKENS * calls
    assert usage.completion_tokens == COMPLETION_TOKENS * calls
    assert usage.total_tokens == (PROMPT_TOKENS + COMPLETION_TOKENS) * calls
    assert usage.successful_requests == calls


@pytest.mark.parametrize("provider", PROVIDERS)
def test_each_call_adds_its_tokens_to_the_usage_summary(
    provider: _ProviderContract, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Usage is cumulative per instance: one call's worth, then two."""
    llm = provider.build()
    provider.stub(llm, monkeypatch, provider.text_response())

    llm.call(PROMPT)
    _assert_usage(llm, calls=1)

    llm.call(PROMPT)
    _assert_usage(llm, calls=2)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_tool_call_response_counts_its_tokens(
    provider: _ProviderContract, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A response that only asks for tools is still a billed request."""
    llm = provider.build()
    provider.stub(llm, monkeypatch, provider.tool_response(ONE_CALL))

    llm.call(PROMPT, tools=[TOOL_SCHEMA])

    _assert_usage(llm, calls=1)


@pytest.mark.parametrize("provider", PROVIDERS)
def test_streamed_call_counts_its_tokens(
    provider: _ProviderContract, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Streaming reports usage at the end of the stream; it must still count."""
    llm = provider.build(stream=True)
    provider.stub_text_stream(llm, monkeypatch)

    llm.call(PROMPT)

    _assert_usage(llm, calls=1)


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("stream", [False, True], ids=["call", "stream"])
def test_completed_event_reports_the_tokens_of_its_call(
    provider: _ProviderContract,
    stream: bool,
    monkeypatch: pytest.MonkeyPatch,
    events: _Events,
) -> None:
    """Listeners see the same numbers the usage summary records.

    Only the totals are compared: providers name the prompt and completion
    counters differently on the raw event payload.
    """
    llm = provider.build(stream=stream)
    if stream:
        provider.stub_text_stream(llm, monkeypatch)
    else:
        provider.stub(llm, monkeypatch, provider.text_response())

    llm.call(PROMPT)
    _flush()

    assert len(events.completed) == 1
    usage = events.completed[0].usage
    assert usage is not None
    assert usage["total_tokens"] == PROMPT_TOKENS + COMPLETION_TOKENS


# --- Failures -----------------------------------------------------------------


def _assert_failed(llm: BaseLLM, events: _Events) -> None:
    assert len(events.started) == 1
    assert events.failed
    assert "provider exploded" in events.failed[0].error
    assert events.completed == []
    assert llm.get_token_usage_summary().successful_requests == 0


def _assert_one_failure(llm: BaseLLM, events: _Events) -> None:
    _assert_failed(llm, events)
    assert len(events.failed) == 1


@pytest.mark.parametrize("provider", PROVIDERS)
def test_failed_call_raises_without_completing(
    provider: _ProviderContract, monkeypatch: pytest.MonkeyPatch, events: _Events
) -> None:
    """A failed call raises, reports the error and records no completion or usage."""
    llm = provider.build()
    provider.stub(llm, monkeypatch, RuntimeError("provider exploded"))

    with pytest.raises(RuntimeError, match="provider exploded"):
        llm.call(PROMPT)
    _flush()

    _assert_failed(llm, events)


@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.asyncio
async def test_failed_acall_raises_without_completing(
    provider: _ProviderContract, monkeypatch: pytest.MonkeyPatch, events: _Events
) -> None:
    """The async path reports a failure the same way."""
    llm = provider.build()
    provider.stub_async(llm, monkeypatch, RuntimeError("provider exploded"))

    with pytest.raises(RuntimeError, match="provider exploded"):
        await llm.acall(PROMPT)
    _flush()

    _assert_failed(llm, events)


@pytest.mark.parametrize("provider", _providers(openai=_OPENAI_FAILS_TWICE))
def test_failed_call_emits_one_failed_event(
    provider: _ProviderContract, monkeypatch: pytest.MonkeyPatch, events: _Events
) -> None:
    """One failed logical call is one failure event, however deep it was raised."""
    llm = provider.build()
    provider.stub(llm, monkeypatch, RuntimeError("provider exploded"))

    with pytest.raises(RuntimeError, match="provider exploded"):
        llm.call(PROMPT)
    _flush()

    assert len(events.failed) == 1


@pytest.mark.parametrize("provider", _providers(openai=_OPENAI_FAILS_TWICE))
@pytest.mark.asyncio
async def test_failed_acall_emits_one_failed_event(
    provider: _ProviderContract, monkeypatch: pytest.MonkeyPatch, events: _Events
) -> None:
    """The async path emits one failure event too."""
    llm = provider.build()
    provider.stub_async(llm, monkeypatch, RuntimeError("provider exploded"))

    with pytest.raises(RuntimeError, match="provider exploded"):
        await llm.acall(PROMPT)
    _flush()

    assert len(events.failed) == 1


@pytest.mark.parametrize("provider", PROVIDERS)
def test_failed_stream_raises_and_emits_one_failed_event(
    provider: _ProviderContract, monkeypatch: pytest.MonkeyPatch, events: _Events
) -> None:
    """A stream that cannot be opened fails the call once, with no chunks."""
    llm = provider.build(stream=True)
    provider.stub_failing_stream(llm, monkeypatch, RuntimeError("provider exploded"))

    with pytest.raises(RuntimeError, match="provider exploded"):
        llm.call(PROMPT)
    _flush()

    _assert_one_failure(llm, events)
    assert events.chunks == []
