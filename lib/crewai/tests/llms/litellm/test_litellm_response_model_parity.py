"""response_model must be honoured on all four LiteLLM completion paths.

The sync/async and streaming/non-streaming handlers each receive ``response_model``
from ``LLM.call``/``LLM.acall``. These tests pin that every one of them actually
converts the model output, rather than returning the raw text.

No cassettes: ``litellm.completion``/``acompletion`` and ``InternalInstructor`` are
both replaced, so nothing here touches the network.
"""

import json
from typing import Any

from litellm.types.utils import (
    ChatCompletionDeltaToolCall,
    Delta as LiteLLMDelta,
    Function,
    ModelResponseStream,
    StreamingChoices as LiteLLMStreamingChoices,
)
import pytest
from pydantic import BaseModel

from crewai import llm as llm_module
from crewai.llm import LLM


class Landmark(BaseModel):
    """Target schema for the structured-output conversion."""

    city: str
    population: int


# What a model streams when it answers in prose rather than JSON. Deliberately not
# valid JSON: if a handler returns this verbatim, response_model was ignored.
PROSE = "The Eiffel Tower is in Paris, which has about 2,100,000 residents."
STRUCTURED = '{"city": "Paris", "population": 2100000}'


class _FakeInstructor:
    """Stand-in for InternalInstructor that converts without a second LLM call."""

    def __init__(self, content: Any = None, model: Any = None, llm: Any = None, **_: Any):
        self.model = model

    def to_pydantic(self) -> BaseModel:
        return self.model.model_validate_json(STRUCTURED)


def _stream_chunk(content: str, finish: str | None = None) -> ModelResponseStream:
    """A real litellm streaming chunk; the handlers type-check against these."""
    return ModelResponseStream(
        id="chunk-1",
        choices=[
            LiteLLMStreamingChoices(
                index=0,
                delta=LiteLLMDelta(content=content),
                finish_reason=finish,
            )
        ],
    )


def _tool_call_chunk(name: str, arguments: str, finish: str | None = None) -> ModelResponseStream:
    """A streaming chunk carrying a tool call rather than content.

    The name is repeated on every chunk because the sync handler reads tool_calls
    off the last chunk's delta while the async one accumulates across chunks.
    """
    return ModelResponseStream(
        id="chunk-tc",
        choices=[
            LiteLLMStreamingChoices(
                index=0,
                delta=LiteLLMDelta(
                    content=None,
                    tool_calls=[
                        ChatCompletionDeltaToolCall(
                            index=0,
                            id="call-1",
                            type="function",
                            function=Function(name=name, arguments=arguments),
                        )
                    ],
                ),
                finish_reason=finish,
            )
        ],
    )


class _Message:
    def __init__(self, content: str) -> None:
        self.content = content
        self.tool_calls = None


class _Choice:
    def __init__(self, content: str) -> None:
        self.message = _Message(content)
        self.finish_reason = "stop"


class _Response:
    """Minimal non-streaming response shape, accessed both ways by the handlers."""

    def __init__(self, content: str) -> None:
        self.choices = [_Choice(content)]
        self.id = "resp-1"
        self.usage = None

    def __getitem__(self, key: str) -> Any:
        return {"choices": [{"message": {"content": PROSE}}]}[key]


@pytest.fixture
def fake_litellm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve PROSE from every litellm entry point, sync and async, stream and not."""

    def _chunks():
        yield _stream_chunk(PROSE[:20])
        yield _stream_chunk(PROSE[20:], finish="stop")

    async def _achunks():
        yield _stream_chunk(PROSE[:20])
        yield _stream_chunk(PROSE[20:], finish="stop")

    import litellm

    monkeypatch.setattr(
        litellm,
        "completion",
        lambda **kw: _chunks() if kw.get("stream") else _Response(PROSE),
    )

    async def _acompletion(**kw: Any) -> Any:
        return _achunks() if kw.get("stream") else _Response(PROSE)

    monkeypatch.setattr(litellm, "acompletion", _acompletion)
    monkeypatch.setattr(llm_module, "InternalInstructor", _FakeInstructor)
    monkeypatch.setattr(
        "crewai.utilities.internal_instructor.InternalInstructor", _FakeInstructor
    )


def _assert_structured(result: Any) -> None:
    """The handler must return the converted schema, not the raw prose."""
    assert isinstance(result, str)
    assert result != PROSE, "response_model was ignored: raw model text returned"
    assert json.loads(result) == {"city": "Paris", "population": 2100000}


MESSAGES = [{"role": "user", "content": "Where is the Eiffel Tower?"}]


@pytest.mark.parametrize("stream", [False, True], ids=["non_streaming", "streaming"])
def test_sync_call_honours_response_model(fake_litellm: None, stream: bool) -> None:
    """Control: both sync paths already converted the output."""
    llm = LLM(model="gpt-4o", is_litellm=True, api_key="test", stream=stream)

    _assert_structured(llm.call(MESSAGES, response_model=Landmark))


@pytest.mark.asyncio
async def test_async_non_streaming_honours_response_model(fake_litellm: None) -> None:
    """Control: the async non-streaming path already converted the output."""
    llm = LLM(model="gpt-4o", is_litellm=True, api_key="test", stream=False)

    _assert_structured(await llm.acall(MESSAGES, response_model=Landmark))


@pytest.mark.asyncio
async def test_async_streaming_honours_response_model(fake_litellm: None) -> None:
    """The regression: _ahandle_streaming_response accepted response_model unused."""
    llm = LLM(model="gpt-4o", is_litellm=True, api_key="test", stream=True)

    _assert_structured(await llm.acall(MESSAGES, response_model=Landmark))


@pytest.mark.asyncio
async def test_async_streaming_emits_the_structured_response(
    fake_litellm: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What is returned must also be what is reported to listeners.

    Converting only the return value would leave observability — traces, token
    accounting, anything subscribed to the completed call — showing the raw prose.
    """
    emitted: list[Any] = []
    original = LLM._handle_emit_call_events

    def _spy(self: LLM, response: Any, call_type: Any, **kwargs: Any) -> Any:
        emitted.append(response)
        return original(self, response, call_type, **kwargs)

    monkeypatch.setattr(LLM, "_handle_emit_call_events", _spy)
    llm = LLM(model="gpt-4o", is_litellm=True, api_key="test", stream=True)

    result = await llm.acall(MESSAGES, response_model=Landmark)

    _assert_structured(result)
    assert emitted, "the completed call event was never emitted"
    assert PROSE not in emitted, "listeners were handed the raw prose"
    assert [json.loads(payload) for payload in emitted] == [
        {"city": "Paris", "population": 2100000}
    ]


@pytest.mark.asyncio
async def test_async_streaming_without_response_model_returns_text(
    fake_litellm: None,
) -> None:
    """Control: with no response_model the raw streamed text is still returned."""
    llm = LLM(model="gpt-4o", is_litellm=True, api_key="test", stream=True)

    assert await llm.acall(MESSAGES) == PROSE


@pytest.mark.asyncio
async def test_async_streaming_declined_tool_call_is_not_converted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A tool call the helper declines must not be turned into structured output.

    `_handle_streaming_tool_calls` returns None for an unknown tool, leaving the
    accumulated content empty. Converting that would fabricate an answer the model
    never produced — so the async handler must fall through the same way the sync
    one does, and both must agree on what comes back.
    """

    def _chunks() -> Any:
        yield _tool_call_chunk("unknown_tool", '{"a":')
        yield _tool_call_chunk("unknown_tool", "1}", finish="tool_calls")

    async def _achunks() -> Any:
        for chunk in _chunks():
            yield chunk

    import litellm

    monkeypatch.setattr(litellm, "completion", lambda **kw: _chunks())

    async def _acompletion(**kw: Any) -> Any:
        return _achunks()

    monkeypatch.setattr(litellm, "acompletion", _acompletion)
    monkeypatch.setattr(llm_module, "InternalInstructor", _FakeInstructor)
    monkeypatch.setattr(
        "crewai.utilities.internal_instructor.InternalInstructor", _FakeInstructor
    )

    available_functions = {"a_different_tool": lambda **_: "unused"}
    sync_llm = LLM(model="gpt-4o", is_litellm=True, api_key="test", stream=True)
    async_llm = LLM(model="gpt-4o", is_litellm=True, api_key="test", stream=True)

    sync_result = sync_llm.call(
        MESSAGES, response_model=Landmark, available_functions=available_functions
    )
    async_result = await async_llm.acall(
        MESSAGES, response_model=Landmark, available_functions=available_functions
    )

    assert async_result != STRUCTURED, "a structured answer was invented from no content"
    assert async_result == sync_result
