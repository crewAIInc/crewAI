"""Regression tests for async native tools under kickoff_async's worker threads.

Flow methods run in worker threads (``asyncio.to_thread``), so a native async
tool's coroutine historically went through a bare ``asyncio.run`` inside
``Tool.run`` on a fresh, unrelated loop. Anything bound to the caller's loop
(an ``httpx.AsyncClient``, a caller-loop ``Future``) then failed with
"attached to a different loop" (the post-#8192 shape of the issue #7630 family).

``run_native_tool_result`` bridges async tool coroutines back onto the captured
caller loop (``AgentExecutor._caller_loop``) via ``arun``, and falls back to the
historical ``run()`` behavior for sync tools and sync callers.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai.llms.base_llm import BaseLLM
from crewai.tools import tool
from crewai.utilities.agent_utils import run_native_tool_result


class _FutureHolder:
    """Bridges the test's caller-loop Future into the async tool's scope."""

    value: asyncio.Future | None = None


_HOLDER = _FutureHolder()


@tool("sync_probe")
def sync_probe(x: str) -> str:
    """Sync probe tool."""
    return f"ran:{x}"


@tool("async_probe")
async def async_probe(query: str) -> str:
    """Async probe tool: may await a caller-loop-bound Future when one is armed."""
    caller_bound = _HOLDER.value
    if caller_bound is not None:
        await asyncio.wait_for(caller_bound, timeout=2)
        _HOLDER.value = None
        return f"got:{caller_bound.result()}"
    await asyncio.sleep(0)
    return f"async-ok:{query}"


def test_sync_tool_executes_directly() -> None:
    """Sync tools run through run() regardless of caller_loop."""
    assert run_native_tool_result(sync_probe.run, {"x": "1"}, caller_loop=None) == "ran:1"
    loop = asyncio.new_event_loop()
    try:
        assert run_native_tool_result(sync_probe.run, {"x": "2"}, tool=sync_probe, caller_loop=loop) == "ran:2"
    finally:
        loop.close()

def test_async_tool_without_caller_loop_uses_run_fresh_loop() -> None:
    """Without an async caller, the historical run() fresh-loop behavior holds."""
    _HOLDER.value = None
    assert run_native_tool_result(async_probe.run, {"query": "q"}, tool=async_probe) == "async-ok:q"


def test_async_tool_bridges_to_the_caller_loop() -> None:
    """In a worker thread, an async tool's arun runs on the caller's loop and can
    await caller-loop-bound objects — the regression fresh per-call loops broke."""
    caller_loop = asyncio.new_event_loop()
    try:

        async def scenario() -> str:
            caller_bound: asyncio.Future[str] = (
                asyncio.get_running_loop().create_future()
            )
            asyncio.get_running_loop().call_later(0.05, caller_bound.set_result, "ok")
            _HOLDER.value = caller_bound

            async def run_in_worker() -> str:
                # Mimic the Flow runtime: sync lane executes in a worker thread
                # while the caller loop keeps serving.
                return await asyncio.to_thread(
                    run_native_tool_result,
                    async_probe.run,
                    {"query": "hi"},
                    async_probe,
                    caller_loop=caller_loop,
                )

            return await run_in_worker()

        assert caller_loop.run_until_complete(scenario()) == "got:ok"
    finally:
        caller_loop.close()


class ScriptedLLM(BaseLLM):
    """Stub LLM emitting one native tool call, then a final answer."""

    def __init__(self, responses: list[Any]) -> None:
        super().__init__(model="stub-model")
        self._responses = list(responses)
        self.calls: list[list[dict[str, Any]]] = []

    def supports_function_calling(self) -> bool:
        return True

    def supports_stop_words(self) -> bool:
        return True

    def call(
        self,
        messages: Any,
        tools: Any = None,
        available_functions: Any = None,
        callbacks: Any = None,
        from_task: Any = None,
        from_agent: Any = None,
        response_model: Any = None,
        executor_context: Any = None,
        verbose: bool = False,
    ) -> Any:
        self.calls.append(list(messages))
        step = self._responses.pop(0)
        if isinstance(step, tuple) and step[0] == "tool":
            return step[1]
        return str(step)


def _tool_result_seen_by_llm(llm: ScriptedLLM) -> str:
    if len(llm.calls) < 2:
        return "<no tool message>"
    for message in llm.calls[1]:
        if message.get("role") == "tool":
            return str(message.get("content"))
    return "<no tool message>"


def test_async_native_tool_honors_caller_loop_under_kickoff_async() -> None:
    """End-to-end: the async native tool sees the kickoff_async caller's loop, so a
    caller-loop Future resolves instead of failing with 'attached to a different loop'."""

    tool_call = [
        {
            "id": "call_1",
            "type": "function",
            "function": {
                "name": "async_probe",
                "arguments": json.dumps({"query": "hello"}),
            },
        }
    ]
    llm = ScriptedLLM([("tool", tool_call), "final answer"])
    stub_agent = Agent(
        role="worker",
        goal="use the tools",
        backstory="agent",
        llm=llm,
        tools=[async_probe],
        allow_delegation=False,
        max_iter=6,
    )
    task = Task(description="use the tool", expected_output="the result", agent=stub_agent)
    crew = Crew(agents=[stub_agent], tasks=[task], process=Process.sequential)

    async def scenario() -> None:
        _HOLDER.value = asyncio.get_running_loop().create_future()
        asyncio.get_running_loop().call_later(
            0.05, _HOLDER.value.set_result, "expected-item"
        )
        await crew.kickoff_async()

    asyncio.run(scenario())

    seen = _tool_result_seen_by_llm(llm)
    assert "got:expected-item" in seen, f"async tool failed; LLM saw: {seen!r}"