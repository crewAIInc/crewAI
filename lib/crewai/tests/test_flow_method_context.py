"""A method's context must remain active until its returned coroutine finishes."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, ClassVar

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.flow_events import (
    FlowInputReceivedEvent,
    FlowInputRequestedEvent,
)
from crewai.flow import Flow, start
from crewai.flow.flow_context import current_flow_method_name
from crewai.flow.types import FlowMethodName
import pytest


class LocalInputProvider:
    def request_input(
        self, message: str, flow: Flow[Any], metadata: dict[str, Any] | None = None
    ) -> str:
        return "a topic"


class MethodContextFlow(Flow):
    _skip_auto_memory: ClassVar[bool] = True

    async def gather_input(self) -> str | None:
        await asyncio.sleep(0)
        return self.ask("Which topic?")


class NativeAsyncFlow(MethodContextFlow):
    @start()
    async def gather(self) -> str | None:
        return await self.gather_input()


class CoroutineReturningFlow(MethodContextFlow):
    @start()
    def gather(self) -> Coroutine[Any, Any, str | None]:
        return self.gather_input()


@pytest.mark.parametrize("mode", ["native_async", "returned_coroutine", "declarative"])
def test_ask_records_method_for_entire_coroutine(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    provider = LocalInputProvider()
    if mode == "declarative":
        flow = MethodContextFlow.from_declaration(
            contents={
                "schema": "crewai.flow/v1",
                "name": "MethodContextFlow",
                "methods": {
                    "gather": {
                        "start": True,
                        "do": {
                            "call": "code",
                            "ref": f"{__name__}:MethodContextFlow.gather_input",
                        },
                    }
                },
            },
            input_provider=provider,
            tracing=False,
        )
    else:
        flow_type = (
            NativeAsyncFlow if mode == "native_async" else CoroutineReturningFlow
        )
        flow = flow_type(input_provider=provider, tracing=False)

    input_events: list[FlowInputRequestedEvent | FlowInputReceivedEvent] = []
    original_emit = crewai_event_bus.emit

    def capture_emit(source: Any, event: Any, **kwargs: Any) -> Any:
        if isinstance(event, (FlowInputRequestedEvent, FlowInputReceivedEvent)):
            input_events.append(event)
        return original_emit(source, event, **kwargs)

    monkeypatch.setattr(crewai_event_bus, "emit", capture_emit)
    assert flow.kickoff() == "a topic"
    assert [entry["method_name"] for entry in flow._input_history] == ["gather"]
    assert [event.method_name for event in input_events] == ["gather", "gather"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, RuntimeError, asyncio.CancelledError])
async def test_returned_coroutine_restores_method_context(
    failure: type[BaseException] | None,
) -> None:
    flow = CoroutineReturningFlow(tracing=False)
    observed: list[str] = []

    async def body() -> None:
        observed.append(current_flow_method_name.get())
        await asyncio.sleep(0)
        if failure is not None:
            raise failure("interrupted method")

    def method() -> Coroutine[Any, Any, None]:
        return body()

    token = current_flow_method_name.set("caller")
    try:
        if failure is None:
            await flow._execute_method(FlowMethodName("gather"), method)
        else:
            with pytest.raises(failure, match="interrupted method"):
                await flow._execute_method(FlowMethodName("gather"), method)
        assert current_flow_method_name.get() == "caller"
        assert observed == ["gather"]
    finally:
        current_flow_method_name.reset(token)
