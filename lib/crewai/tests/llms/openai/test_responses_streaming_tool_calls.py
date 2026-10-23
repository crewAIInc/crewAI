from __future__ import annotations

from types import SimpleNamespace

import pytest

from crewai.events.types.llm_events import LLMCallType
from crewai.llms.providers.openai.completion import OpenAICompletion

MESSAGES = [{"role": "user", "content": "multiply"}]
FUNCTION_CALL = SimpleNamespace(
    type="function_call",
    call_id="call_abc",
    name="multiply",
    arguments='{"a": 17, "b": 23}',
)
FINAL_RESPONSE = SimpleNamespace(
    id="resp_1",
    status="completed",
    output=[FUNCTION_CALL],
    usage=None,
)
FAILED_RESPONSE = SimpleNamespace(
    id="resp_2",
    status="failed",
    output=[FUNCTION_CALL],
    error=SimpleNamespace(message="tool stream failed"),
    usage=None,
)
INCOMPLETE_RESPONSE = SimpleNamespace(
    id="resp_3",
    status="incomplete",
    output=[],
    incomplete_details=SimpleNamespace(reason="max_output_tokens"),
    usage=None,
)


def _events(response=FINAL_RESPONSE, terminal_type="response.completed"):
    yield SimpleNamespace(type="response.created", response=response)
    yield SimpleNamespace(type="response.output_item.done", item=FUNCTION_CALL)
    yield SimpleNamespace(type=terminal_type, response=response)


async def _async_events(response=FINAL_RESPONSE, terminal_type="response.completed"):
    for event in _events(response=response, terminal_type=terminal_type):
        yield event


def _events_without_terminal(response=FINAL_RESPONSE):
    yield SimpleNamespace(type="response.created", response=response)
    yield SimpleNamespace(type="response.output_item.done", item=FUNCTION_CALL)


async def _async_events_without_terminal(response=FINAL_RESPONSE):
    for event in _events_without_terminal(response=response):
        yield event


def _incomplete_text_events():
    yield SimpleNamespace(type="response.created", response=INCOMPLETE_RESPONSE)
    yield SimpleNamespace(type="response.output_text.delta", delta="partial")
    yield SimpleNamespace(type="response.incomplete", response=INCOMPLETE_RESPONSE)


async def _async_incomplete_text_events():
    for event in _incomplete_text_events():
        yield event


def _build_llm() -> OpenAICompletion:
    return OpenAICompletion(model="gpt-5.5", api_key="sk-test", api="responses", stream=True)


def test_streaming_responses_returns_tool_calls_to_executor_when_no_functions(monkeypatch):
    llm = _build_llm()
    completed: list[dict] = []

    monkeypatch.setattr(
        llm,
        "_get_sync_client",
        lambda: SimpleNamespace(
            responses=SimpleNamespace(create=lambda **_kwargs: _events())
        ),
    )
    monkeypatch.setattr(
        llm,
        "_emit_call_completed_event",
        lambda **kwargs: completed.append(kwargs),
    )

    result = llm._handle_streaming_responses(
        params={"input": MESSAGES}, available_functions=None
    )

    assert result == [
        {"id": "call_abc", "name": "multiply", "arguments": '{"a": 17, "b": 23}'}
    ]
    assert completed == [
        {
            "response": result,
            "call_type": LLMCallType.TOOL_CALL,
            "from_task": None,
            "from_agent": None,
            "messages": MESSAGES,
            "usage": None,
            "finish_reason": "completed",
            "response_id": "resp_1",
        }
    ]


def test_streaming_responses_does_not_return_tool_calls_without_completed_event(
    monkeypatch,
):
    llm = _build_llm()
    completed: list[dict] = []

    monkeypatch.setattr(
        llm,
        "_get_sync_client",
        lambda: SimpleNamespace(
            responses=SimpleNamespace(create=lambda **_kwargs: _events_without_terminal())
        ),
    )
    monkeypatch.setattr(
        llm,
        "_emit_call_completed_event",
        lambda **kwargs: completed.append(kwargs),
    )

    with pytest.raises(RuntimeError, match="before response.completed"):
        llm._handle_streaming_responses(
            params={"input": MESSAGES}, available_functions=None
        )

    assert completed == []


def test_streaming_responses_does_not_return_tool_calls_after_failed_terminal_event(
    monkeypatch,
):
    llm = _build_llm()
    completed: list[dict] = []

    monkeypatch.setattr(
        llm,
        "_get_sync_client",
        lambda: SimpleNamespace(
            responses=SimpleNamespace(
                create=lambda **_kwargs: _events(
                    response=FAILED_RESPONSE, terminal_type="response.failed"
                )
            )
        ),
    )
    monkeypatch.setattr(
        llm,
        "_emit_call_completed_event",
        lambda **kwargs: completed.append(kwargs),
    )

    with pytest.raises(RuntimeError, match="tool stream failed"):
        llm._handle_streaming_responses(
            params={"input": MESSAGES}, available_functions=None
        )

    assert completed == []


def test_streaming_responses_returns_partial_text_after_incomplete_terminal_event(
    monkeypatch,
):
    llm = _build_llm()
    completed: list[dict] = []

    monkeypatch.setattr(
        llm,
        "_get_sync_client",
        lambda: SimpleNamespace(
            responses=SimpleNamespace(create=lambda **_kwargs: _incomplete_text_events())
        ),
    )
    monkeypatch.setattr(
        llm,
        "_emit_call_completed_event",
        lambda **kwargs: completed.append(kwargs),
    )

    assert (
        llm._handle_streaming_responses(
            params={"input": MESSAGES}, available_functions=None
        )
        == "partial"
    )
    assert completed[0]["response"] == "partial"
    assert completed[0]["call_type"] is LLMCallType.LLM_CALL


@pytest.mark.asyncio
async def test_async_streaming_responses_returns_tool_calls_to_executor_when_no_functions(
    monkeypatch,
):
    llm = _build_llm()
    completed: list[dict] = []

    async def create(**_kwargs):
        return _async_events()

    monkeypatch.setattr(
        llm,
        "_get_async_client",
        lambda: SimpleNamespace(responses=SimpleNamespace(create=create)),
    )
    monkeypatch.setattr(
        llm,
        "_emit_call_completed_event",
        lambda **kwargs: completed.append(kwargs),
    )

    result = await llm._ahandle_streaming_responses(
        params={"input": MESSAGES}, available_functions=None
    )

    assert result == [
        {"id": "call_abc", "name": "multiply", "arguments": '{"a": 17, "b": 23}'}
    ]
    assert completed[0]["call_type"] is LLMCallType.TOOL_CALL
    assert completed[0]["response"] == result
    assert completed[0]["messages"] == MESSAGES
    assert completed[0]["finish_reason"] == "completed"
    assert completed[0]["response_id"] == "resp_1"


@pytest.mark.asyncio
async def test_async_streaming_responses_does_not_return_tool_calls_without_completed_event(
    monkeypatch,
):
    llm = _build_llm()
    completed: list[dict] = []

    async def create(**_kwargs):
        return _async_events_without_terminal()

    monkeypatch.setattr(
        llm,
        "_get_async_client",
        lambda: SimpleNamespace(responses=SimpleNamespace(create=create)),
    )
    monkeypatch.setattr(
        llm,
        "_emit_call_completed_event",
        lambda **kwargs: completed.append(kwargs),
    )

    with pytest.raises(RuntimeError, match="before response.completed"):
        await llm._ahandle_streaming_responses(
            params={"input": MESSAGES}, available_functions=None
        )

    assert completed == []


@pytest.mark.asyncio
async def test_async_streaming_responses_returns_partial_text_after_incomplete_terminal_event(
    monkeypatch,
):
    llm = _build_llm()
    completed: list[dict] = []

    async def create(**_kwargs):
        return _async_incomplete_text_events()

    monkeypatch.setattr(
        llm,
        "_get_async_client",
        lambda: SimpleNamespace(responses=SimpleNamespace(create=create)),
    )
    monkeypatch.setattr(
        llm,
        "_emit_call_completed_event",
        lambda **kwargs: completed.append(kwargs),
    )

    assert (
        await llm._ahandle_streaming_responses(
            params={"input": MESSAGES}, available_functions=None
        )
        == "partial"
    )
    assert completed[0]["response"] == "partial"
    assert completed[0]["call_type"] is LLMCallType.LLM_CALL


@pytest.mark.asyncio
async def test_async_streaming_responses_does_not_return_tool_calls_after_failed_terminal_event(
    monkeypatch,
):
    llm = _build_llm()
    completed: list[dict] = []

    async def create(**_kwargs):
        return _async_events(response=FAILED_RESPONSE, terminal_type="response.failed")

    monkeypatch.setattr(
        llm,
        "_get_async_client",
        lambda: SimpleNamespace(responses=SimpleNamespace(create=create)),
    )
    monkeypatch.setattr(
        llm,
        "_emit_call_completed_event",
        lambda **kwargs: completed.append(kwargs),
    )

    with pytest.raises(RuntimeError, match="tool stream failed"):
        await llm._ahandle_streaming_responses(
            params={"input": MESSAGES}, available_functions=None
        )

    assert completed == []
