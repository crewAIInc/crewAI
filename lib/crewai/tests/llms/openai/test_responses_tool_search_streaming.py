"""Async and streamed Responses parity for hosted tool discovery.

Streams are built from OpenAI SDK event models and served by fake clients,
so no network access or API key is needed.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from types import SimpleNamespace
from typing import Any

from crewai import Agent, Crew, Task
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.llm_events import (
    LLMCallCompletedEvent,
    LLMCallFailedEvent,
    LLMCallType,
)
from crewai.events.types.tool_usage_events import (
    ToolUsageErrorEvent,
    ToolUsageFinishedEvent,
    ToolUsageStartedEvent,
)
from crewai.llms.providers.openai.completion import OpenAICompletion
from crewai.tools import BaseTool
from openai.types.responses import Response, ResponseStreamEvent
from pydantic import Field, TypeAdapter
import pytest

from tests.utils import wait_for_event_handlers


MESSAGES = [{"role": "user", "content": "Look things up"}]
MODES = ["sync", "async"]

TOOL_SEARCH_ITEMS: list[dict[str, Any]] = [
    {
        "id": "ts_1",
        "type": "tool_search_call",
        "execution": "server",
        "call_id": None,
        "status": "completed",
        "arguments": {"paths": ["lookup", "weather"]},
    },
    {
        "id": "tso_1",
        "type": "tool_search_output",
        "execution": "server",
        "call_id": None,
        "status": "completed",
        "tools": [
            {
                "type": "function",
                "name": name,
                "description": "Discovered tool",
                "parameters": {"type": "object", "properties": {}},
                "strict": False,
                "defer_loading": True,
            }
            for name in ("lookup", "weather")
        ],
    },
]

DISCOVERED_CALLS: list[tuple[str, str, str, str]] = [
    ("fc_1", "call_lookup", "lookup", '{"query": "hello"}'),
    ("fc_2", "call_weather", "weather", '{"city": "Paris"}'),
]

_EVENT_ADAPTER: TypeAdapter[Any] = TypeAdapter(ResponseStreamEvent)


def _function_call(
    item_id: str, call_id: str, name: str, arguments: str, status: str
) -> dict[str, Any]:
    return {
        "id": item_id,
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "arguments": arguments,
        "status": status,
    }


def _message(text: str) -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _response(
    output: list[dict[str, Any]],
    *,
    response_id: str = "resp_1",
    status: str = "completed",
    **extra: Any,
) -> Response:
    return Response.model_validate(
        {
            "id": response_id,
            "created_at": 0,
            "object": "response",
            "model": "gpt-5.5",
            "output": output,
            "status": status,
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            **extra,
        }
    )


class _EventBuilder:
    """Collect SDK stream events with increasing sequence numbers."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    def add(self, event_type: str, **fields: Any) -> None:
        self.events.append(
            _EVENT_ADAPTER.validate_python(
                {"type": event_type, "sequence_number": len(self.events), **fields}
            )
        )


def _discovery_stream_prefix(chunk_size: int = 4) -> tuple[_EventBuilder, Response]:
    """Events up to, but excluding, the terminal event.

    Hosted search items come first, then two function calls whose argument
    fragments are interleaved, as the API may do with parallel tool calls.
    Returns the builder and the matching completed response.
    """
    builder = _EventBuilder()
    final_output = [
        *TOOL_SEARCH_ITEMS,
        *(_function_call(*call, "completed") for call in DISCOVERED_CALLS),
    ]
    builder.add(
        "response.created",
        response=_response([], status="in_progress").model_dump(),
    )
    for index, item in enumerate(TOOL_SEARCH_ITEMS):
        builder.add("response.output_item.added", output_index=index, item=item)
        builder.add("response.output_item.done", output_index=index, item=item)

    first_call_index = len(TOOL_SEARCH_ITEMS)
    fragments: list[list[str]] = []
    for offset, (item_id, call_id, name, arguments) in enumerate(DISCOVERED_CALLS):
        builder.add(
            "response.output_item.added",
            output_index=first_call_index + offset,
            item=_function_call(item_id, call_id, name, "", "in_progress"),
        )
        fragments.append(
            [
                arguments[i : i + chunk_size]
                for i in range(0, len(arguments), chunk_size)
            ]
        )

    for step in range(max(len(parts) for parts in fragments)):
        for offset, parts in enumerate(fragments):
            if step < len(parts):
                builder.add(
                    "response.function_call_arguments.delta",
                    item_id=DISCOVERED_CALLS[offset][0],
                    output_index=first_call_index + offset,
                    delta=parts[step],
                )

    for offset, call in enumerate(DISCOVERED_CALLS):
        item_id, _, name, arguments = call
        builder.add(
            "response.function_call_arguments.done",
            item_id=item_id,
            output_index=first_call_index + offset,
            name=name,
            arguments=arguments,
        )
        builder.add(
            "response.output_item.done",
            output_index=first_call_index + offset,
            item=_function_call(*call, "completed"),
        )
    return builder, _response(final_output)


def _completed_discovery_stream() -> tuple[list[Any], Response]:
    builder, final_response = _discovery_stream_prefix()
    builder.add("response.completed", response=final_response.model_dump())
    return builder.events, final_response


def _text_stream(text: str, response_id: str) -> list[Any]:
    builder = _EventBuilder()
    builder.add(
        "response.output_text.delta",
        item_id="msg_1",
        output_index=0,
        content_index=0,
        delta=text,
        logprobs=[],
    )
    builder.add(
        "response.completed",
        response=_response([_message(text)], response_id=response_id).model_dump(),
    )
    return builder.events


def _sync_client(*streams: list[Any]) -> SimpleNamespace:
    pending = [iter(stream) for stream in streams]
    requests: list[dict[str, Any]] = []

    def create(**kwargs: Any) -> Any:
        requests.append(kwargs)
        return pending.pop(0)

    return SimpleNamespace(responses=SimpleNamespace(create=create), requests=requests)


def _async_client(*streams: list[Any]) -> SimpleNamespace:
    pending = list(streams)
    requests: list[dict[str, Any]] = []

    async def iterate(events: list[Any]) -> AsyncIterator[Any]:
        for event in events:
            yield event

    async def create(**kwargs: Any) -> AsyncIterator[Any]:
        requests.append(kwargs)
        return iterate(pending.pop(0))

    return SimpleNamespace(responses=SimpleNamespace(create=create), requests=requests)


def _build_llm(*, stream: bool = True) -> OpenAICompletion:
    return OpenAICompletion(
        model="gpt-5.5",
        api="responses",
        api_key="sk-test",
        stream=stream,
        tool_search=True,
    )


async def _stream_once(
    llm: OpenAICompletion,
    mode: str,
    events: list[Any],
    available_functions: dict[str, Any] | None = None,
) -> Any:
    """Run one streamed Responses call through the public sync or async API."""
    if mode == "sync":
        llm._client = _sync_client(events)
        return llm.call(MESSAGES, available_functions=available_functions)
    llm._async_client = _async_client(events)
    return await llm.acall(MESSAGES, available_functions=available_functions)


class _EventLog:
    """Record LLM and tool events emitted while the context is active."""

    def __init__(self) -> None:
        self.events: list[Any] = []

    def of(self, event_type: type) -> list[Any]:
        return [event for event in self.events if isinstance(event, event_type)]


def _record_events() -> tuple[_EventLog, Callable[[], None]]:
    log = _EventLog()
    for event_type in (
        LLMCallCompletedEvent,
        LLMCallFailedEvent,
        ToolUsageStartedEvent,
        ToolUsageFinishedEvent,
        ToolUsageErrorEvent,
    ):

        @crewai_event_bus.on(event_type)
        def _record(_source: Any, event: Any) -> None:
            log.events.append(event)

    return log, wait_for_event_handlers


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_fragmented_discovered_calls_match_non_streaming_path(mode: str) -> None:
    llm = _build_llm()
    events, final_response = _completed_discovery_stream()

    with crewai_event_bus.scoped_handlers():
        log, flush = _record_events()
        result = await _stream_once(llm, mode, events)
        flush()

    assert result == llm._extract_function_calls_from_response(final_response)
    assert result == [
        {"id": call_id, "name": name, "arguments": arguments}
        for _, call_id, name, arguments in DISCOVERED_CALLS
    ]
    completed = log.of(LLMCallCompletedEvent)
    assert len(completed) == 1
    assert completed[0].call_type is LLMCallType.TOOL_CALL
    assert completed[0].response == result
    assert completed[0].response_id == "resp_1"


@pytest.mark.parametrize(
    "arguments",
    ['{"city": "Paris"', "null", "[]", '"Paris"', 42, None, {"city": "Paris"}],
    ids=["malformed", "null", "array", "string", "int", "none", "dict"],
)
def test_parse_responses_function_args_only_passes_objects_through(
    arguments: Any,
) -> None:
    parsed = OpenAICompletion._parse_responses_function_args(
        {"id": "call_1", "name": "weather", "arguments": arguments}
    )

    assert parsed == ({"city": "Paris"} if isinstance(arguments, dict) else {})


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_only_catalog_functions_run_and_async_tools_are_awaited(
    mode: str,
) -> None:
    llm = _build_llm()
    events, _ = _completed_discovery_stream()
    calls: list[dict[str, Any]] = []

    def sync_weather(**kwargs: Any) -> str:
        calls.append(kwargs)
        return f"sunny in {kwargs['city']}"

    async def async_weather(**kwargs: Any) -> str:
        calls.append(kwargs)
        return f"sunny in {kwargs['city']}"

    weather = sync_weather if mode == "sync" else async_weather

    with crewai_event_bus.scoped_handlers():
        log, flush = _record_events()
        result = await _stream_once(
            llm, mode, events, available_functions={"weather": weather}
        )
        flush()

    assert result == "sunny in Paris"
    assert calls == [{"city": "Paris"}]
    assert [event.tool_name for event in log.of(ToolUsageStartedEvent)] == ["weather"]
    finished = log.of(ToolUsageFinishedEvent)
    assert [event.output for event in finished] == ["sunny in Paris"]
    tool_completions = [
        event
        for event in log.of(LLMCallCompletedEvent)
        if event.call_type is LLMCallType.TOOL_CALL
    ]
    assert [event.response for event in tool_completions] == ["sunny in Paris"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_discovered_tool_failure_emits_failure_events(mode: str) -> None:
    llm = _build_llm()
    events, _ = _completed_discovery_stream()

    async def broken_lookup(**_kwargs: Any) -> str:
        raise RuntimeError("lookup unavailable")

    def sync_broken_lookup(**_kwargs: Any) -> str:
        raise RuntimeError("lookup unavailable")

    lookup = sync_broken_lookup if mode == "sync" else broken_lookup

    with crewai_event_bus.scoped_handlers():
        log, flush = _record_events()
        await _stream_once(llm, mode, events, available_functions={"lookup": lookup})
        flush()

    errors = log.of(ToolUsageErrorEvent)
    assert [event.tool_name for event in errors] == ["lookup"]
    assert "lookup unavailable" in errors[0].error
    assert any(
        "lookup unavailable" in event.error for event in log.of(LLMCallFailedEvent)
    )
    assert log.of(ToolUsageFinishedEvent) == []


@pytest.mark.asyncio
async def test_async_non_streaming_responses_awaits_discovered_tool() -> None:
    llm = _build_llm(stream=False)
    _, final_response = _completed_discovery_stream()

    async def create(**_kwargs: Any) -> Response:
        return final_response

    async def lookup(query: str) -> str:
        return f"value:{query}"

    llm._async_client = SimpleNamespace(responses=SimpleNamespace(create=create))

    result = await llm.acall(MESSAGES, available_functions={"lookup": lookup})

    assert result == "value:hello"


def _failed_mid_arguments() -> list[Any]:
    builder, _ = _discovery_stream_prefix()
    del builder.events[-4:]
    builder.add(
        "response.failed",
        response=_response(
            [],
            status="failed",
            error={"code": "server_error", "message": "stream broke"},
        ).model_dump(),
    )
    return builder.events


def _incomplete_mid_arguments() -> list[Any]:
    builder, _ = _discovery_stream_prefix()
    del builder.events[-4:]
    builder.add(
        "response.incomplete",
        response=_response(
            [],
            status="incomplete",
            incomplete_details={"reason": "max_output_tokens"},
        ).model_dump(),
    )
    return builder.events


def _incomplete_after_completed_call() -> list[Any]:
    builder, _ = _discovery_stream_prefix()
    del builder.events[-2:]
    builder.add(
        "response.incomplete",
        response=_response(
            [],
            status="incomplete",
            incomplete_details={"reason": "max_output_tokens"},
        ).model_dump(),
    )
    return builder.events


def _truncated_without_terminal() -> list[Any]:
    builder, _ = _discovery_stream_prefix()
    del builder.events[-4:]
    return builder.events


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    ("make_events", "error"),
    [
        (_failed_mid_arguments, "stream broke"),
        (_incomplete_mid_arguments, "status: incomplete, reason: max_output_tokens"),
        (_incomplete_after_completed_call, "status: incomplete"),
        (_truncated_without_terminal, "no terminal event received"),
    ],
    ids=["failed", "incomplete", "incomplete-after-done", "truncated"],
)
async def test_unfinished_stream_never_runs_discovered_calls(
    mode: str, make_events: Callable[[], list[Any]], error: str
) -> None:
    llm = _build_llm()
    ran: list[str] = []

    def record(name: str) -> Callable[..., str]:
        def run(**_kwargs: Any) -> str:
            ran.append(name)
            return name

        return run

    available = {"lookup": record("lookup"), "weather": record("weather")}
    events = make_events()

    with crewai_event_bus.scoped_handlers():
        log, flush = _record_events()
        with pytest.raises(RuntimeError, match=error):
            await _stream_once(llm, mode, events, available_functions=available)
        flush()

    assert ran == []
    assert log.of(ToolUsageStartedEvent) == []
    assert len(log.of(LLMCallFailedEvent)) == 1
    assert log.of(LLMCallCompletedEvent) == []


class LookupTool(BaseTool):
    name: str = "lookup"
    description: str = "Look up a value."
    calls: list[str] = Field(default_factory=list)

    def _run(self, query: str = "default") -> str:
        self.calls.append(query)
        return f"value:{query}"


class WeatherTool(BaseTool):
    name: str = "weather"
    description: str = "Report the weather."
    calls: list[str] = Field(default_factory=list)

    def _run(self, city: str = "nowhere") -> str:
        self.calls.append(city)
        return f"sunny in {city}"


def test_crew_round_trips_streamed_discovered_calls_with_matching_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events, _ = _completed_discovery_stream()
    client = _sync_client(events, _text_stream("Done", "resp_2"))
    monkeypatch.setattr(OpenAICompletion, "_build_sync_client", lambda _self: client)
    lookup, weather = LookupTool(), WeatherTool()
    agent = Agent(
        role="Tester",
        goal="Look up values",
        backstory="Tests streamed tool discovery",
        llm=_build_llm(),
        tools=[lookup, weather],
        max_retry_limit=0,
        cache=False,
        verbose=False,
    )
    task = Task(description="Look up hello", expected_output="A value", agent=agent)

    result = Crew(agents=[agent], tasks=[task], cache=False, verbose=False).kickoff()

    assert result.raw == "Done"
    assert lookup.calls == ["hello"]
    assert weather.calls == ["Paris"]
    assert len(client.requests) == 2
    assert all(request["stream"] is True for request in client.requests)
    followup = client.requests[1]["input"]
    calls = [item for item in followup if item.get("type") == "function_call"]
    outputs = [item for item in followup if item.get("type") == "function_call_output"]
    assert [item["call_id"] for item in calls] == ["call_lookup", "call_weather"]
    assert [item["call_id"] for item in outputs] == ["call_lookup", "call_weather"]
    assert [item["output"] for item in outputs] == ["value:hello", "sunny in Paris"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_completed_response_output_wins_over_missing_done_event(
    mode: str,
) -> None:
    """A call whose done event was lost is still taken from the final response."""
    llm = _build_llm()
    builder, final_response = _discovery_stream_prefix()
    del builder.events[-1]
    builder.add("response.completed", response=final_response.model_dump())

    result = await _stream_once(llm, mode, builder.events)

    assert result == llm._extract_function_calls_from_response(final_response)
    assert [call["id"] for call in result] == ["call_lookup", "call_weather"]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_fragments_are_assembled_when_final_output_has_no_calls(
    mode: str,
) -> None:
    """Without calls in the completed response, the stream fragments are used."""
    llm = _build_llm()
    builder, _ = _discovery_stream_prefix(chunk_size=3)
    builder.add(
        "response.completed",
        response=_response(TOOL_SEARCH_ITEMS).model_dump(),
    )

    result = await _stream_once(llm, mode, builder.events)

    assert result == [
        {"id": call_id, "name": name, "arguments": arguments}
        for _, call_id, name, arguments in DISCOVERED_CALLS
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_parse_tool_outputs_does_not_bypass_incomplete_stream_check(
    mode: str,
) -> None:
    llm = OpenAICompletion(
        model="gpt-5.5",
        api="responses",
        api_key="sk-test",
        stream=True,
        tool_search=True,
        parse_tool_outputs=True,
    )

    with crewai_event_bus.scoped_handlers():
        log, flush = _record_events()
        with pytest.raises(RuntimeError, match="before response.completed"):
            await _stream_once(llm, mode, _incomplete_mid_arguments())
        flush()

    assert log.of(LLMCallCompletedEvent) == []


def _delta_only_events() -> list[Any]:
    """Fragments are the only place the arguments appear.

    The ``function_call_arguments.done`` events are removed, the function
    items in ``output_item.done`` carry empty arguments, and the completed
    response holds no function calls, so breaking the delta assembly fails
    the assertions.
    """
    builder, _ = _discovery_stream_prefix(chunk_size=3)
    events: list[Any] = []
    for event in builder.events:
        if event.type == "response.function_call_arguments.done":
            continue
        if event.type == "response.output_item.done" and event.item.type == (
            "function_call"
        ):
            event = event.model_copy(
                update={"item": event.item.model_copy(update={"arguments": ""})}
            )
        events.append(event)
    builder.events = events
    builder.add("response.completed", response=_response(TOOL_SEARCH_ITEMS).model_dump())
    return builder.events


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", MODES)
async def test_arguments_are_assembled_from_deltas_alone(mode: str) -> None:
    llm = _build_llm()

    result = await _stream_once(llm, mode, _delta_only_events())

    assert result == [
        {"id": call_id, "name": name, "arguments": arguments}
        for _, call_id, name, arguments in DISCOVERED_CALLS
    ]
