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
import copy
import json
import os
import threading
from typing import Any

import pytest

from crewai import Agent, Crew, Process, Task
from crewai.llms.base_llm import BaseLLM
from crewai.tools import tool
from crewai.crews.crew_output import CrewOutput
from crewai.types.usage_metrics import UsageMetrics
from crewai.utilities import agent_utils
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


def test_tool_runtime_error_propagates_and_runs_exactly_once() -> None:
    """A RuntimeError raised by the tool's coroutine must propagate unchanged and the
    tool must execute exactly once (it must not be re-run through the sync path)."""
    executions: list[str] = []

    @tool("boom_once")
    async def boom_once(query: str) -> str:
        """Raise RuntimeError after recording a side effect."""
        executions.append(query)
        raise RuntimeError("tool exploded")

    caller_loop = asyncio.new_event_loop()
    try:

        async def scenario() -> Exception | None:
            async def run_in_worker() -> Exception | None:
                try:
                    await asyncio.to_thread(
                        run_native_tool_result,
                        boom_once.run,
                        {"query": "q"},
                        boom_once,
                        caller_loop=caller_loop,
                    )
                except Exception as exc:  # noqa: BLE001 - the exception is the assertion
                    return exc
                return None

            return await run_in_worker()

        exc = caller_loop.run_until_complete(scenario())
        assert isinstance(exc, RuntimeError) and "tool exploded" in str(exc)
        assert executions == ["q"], f"tool ran {len(executions)} times: {executions}"
        assert boom_once.current_usage_count == 1
    finally:
        caller_loop.close()


def test_bridge_is_skipped_when_already_on_the_target_loop() -> None:
    """Calling the helper on the target loop's own thread must not deadlock: it must
    use the provided sync callable instead of scheduling a coroutine it cannot await."""
    arun_calls: list[str] = []

    class _AsyncTool:
        func = None  # class-based tool: detection consults the _arun override

        @staticmethod
        async def _arun(**kwargs: Any) -> str:
            arun_calls.append("arun")
            return "from-arun"

        def arun(self, **kwargs: Any) -> Any:
            return type(self)._arun(**kwargs)

    fake_tool = _AsyncTool()
    caller_loop = asyncio.new_event_loop()
    try:

        async def scenario() -> str:
            # The helper runs ON the very loop it would bridge onto.
            return run_native_tool_result(
                lambda **kwargs: "from-fallback",
                {"x": 1},
                fake_tool,
                caller_loop=asyncio.get_running_loop(),
            )

        assert caller_loop.run_until_complete(scenario()) == "from-fallback"
        assert arun_calls == [], "bridge must not schedule onto the running loop"
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

def test_inherited_arun_is_treated_as_async() -> None:
    """A class-based tool that inherits a real _arun from a parent must still be
    bridged; reading only the concrete class __dict__ would classify it as sync
    and send it back to a fresh asyncio.run loop."""

    class _AsyncParent:
        func = None  # no wrapped callable: force the class-based detection branch

        async def _arun(self, **kwargs: Any) -> str:
            caller_bound = _HOLDER.value
            if caller_bound is not None:
                await asyncio.wait_for(caller_bound, timeout=2)
                _HOLDER.value = None
                return f"got:{caller_bound.result()}"
            return "from-parent-arun"

        def arun(self, **kwargs: Any) -> Any:
            return self._arun(**kwargs)

    class _InheritingTool(_AsyncParent):
        pass

    inherited_tool = _InheritingTool()
    caller_loop = asyncio.new_event_loop()
    try:

        async def scenario() -> str:
            caller_bound: asyncio.Future[str] = asyncio.get_running_loop().create_future()
            asyncio.get_running_loop().call_later(0.05, caller_bound.set_result, "inherited-ok")
            _HOLDER.value = caller_bound
            return await asyncio.to_thread(
                run_native_tool_result,
                lambda **kwargs: "from-sync-fallback",
                {"x": 1},
                inherited_tool,
                caller_loop=caller_loop,
            )

        assert caller_loop.run_until_complete(scenario()) == "got:inherited-ok"
    finally:
        _HOLDER.value = None
        caller_loop.close()


def test_bridge_wait_is_bounded_and_cancels_the_coroutine(monkeypatch: pytest.MonkeyPatch) -> None:
    """A tool that never finishes must surface as a TimeoutError instead of blocking
    the worker thread forever (CWE-833); the abandoned coroutine is cancelled."""

    started = threading.Event()
    cancelled = threading.Event()

    @tool("hangs")
    async def hangs(query: str) -> str:
        """Never finishes."""
        started.set()
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            cancelled.set()
            raise
        return "unreachable"

    monkeypatch.setattr(
        "crewai.utilities.agent_utils.NATIVE_TOOL_BRIDGE_TIMEOUT_SECONDS", 0.2
    )

    caller_loop = asyncio.new_event_loop()
    try:

        async def scenario() -> BaseException | None:
            try:
                await asyncio.to_thread(
                    run_native_tool_result,
                    hangs.run,
                    {"query": "q"},
                    hangs,
                    caller_loop=caller_loop,
                )
            except BaseException as exc:  # noqa: BLE001 - the exception is the assertion
                return exc
            return None

        exc = caller_loop.run_until_complete(scenario())
        assert started.is_set(), "the tool coroutine should have started on the caller loop"
        assert isinstance(exc, TimeoutError), f"expected TimeoutError, got {exc!r}"
        assert "did not finish" in str(exc)
        # let the cancellation land on the loop
        caller_loop.run_until_complete(asyncio.sleep(0))
        assert cancelled.is_set(), "the abandoned coroutine must be cancelled, not left running"
    finally:
        caller_loop.close()


def test_streaming_kickoff_captures_the_caller_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """kickoff_async(stream=True) also runs self.kickoff in a worker thread, so it must
    set the caller-loop context too — otherwise async native tools keep breaking there."""

    captured: list[asyncio.AbstractEventLoop | None] = []

    def fake_kickoff(self: Any, inputs: Any = None, input_files: Any = None) -> Any:
        captured.append(agent_utils._caller_loop_context.get())
        return CrewOutput(raw="streamed", tasks_output=[], token_usage=UsageMetrics())

    async def scenario() -> None:
        stub_agent = Agent(
            role="worker",
            goal="use the tools",
            backstory="agent",
            llm=ScriptedLLM(["unused"]),
            tools=[],
            allow_delegation=False,
        )
        task = Task(description="noop", expected_output="noop", agent=stub_agent)
        crew = Crew(agents=[stub_agent], tasks=[task], process=Process.sequential, stream=True)
        monkeypatch.setattr(Crew, "kickoff", fake_kickoff)

        streaming = await crew.kickoff_async()
        async for _ in streaming:
            pass

    asyncio.run(scenario())

    assert captured, "self.kickoff was never offloaded"
    assert captured[0] is not None, "streaming kickoff lost the caller loop"


def test_concurrent_kickoffs_do_not_starve_the_loop_default_executor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CWE-833 regression: as many concurrent kickoffs as the loop's default executor
    has threads used to wedge the pool, because every parked kickoff thread held a
    default-pool slot while the bridged tool needed one for asyncio.to_thread.

    Kickoffs now run on the dedicated kickoff pool, so the tool still gets a thread.
    Without that pool the bounded bridge wait fires instead (TimeoutError).
    """
    loop_pool_workers = min(32, (os.cpu_count() or 1) + 4)
    offloaded_on_caller_loop: list[str] = []

    @tool("blocking_sdk")
    async def blocking_sdk(query: str) -> str:
        """Emulate a blocking SDK offload (the pattern that starved the pool)."""
        value = await asyncio.to_thread(lambda: f"sdk:{query}")
        offloaded_on_caller_loop.append(value)
        return value

    tool_call = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "blocking_sdk", "arguments": json.dumps({"query": "q"})},
        }
    ]

    def build_crew() -> tuple[Crew, ScriptedLLM]:
        llm = ScriptedLLM([("tool", copy.deepcopy(tool_call)), "final answer"])
        agent = Agent(
            role="worker",
            goal="use the tools",
            backstory="agent",
            llm=llm,
            tools=[blocking_sdk],
            allow_delegation=False,
            max_iter=6,
        )
        task = Task(description="use the tool", expected_output="the result", agent=agent)
        return Crew(agents=[agent], tasks=[task], process=Process.sequential), llm

    # A short bridge bound turns a would-be permanent hang into a fast failure.
    monkeypatch.setattr(
        "crewai.utilities.agent_utils.NATIVE_TOOL_BRIDGE_TIMEOUT_SECONDS", 5.0
    )

    crews = [build_crew() for _ in range(loop_pool_workers)]

    async def scenario() -> None:
        await asyncio.gather(*(crew.kickoff_async() for crew, _ in crews))

    asyncio.run(scenario())

    results = [_tool_result_seen_by_llm(llm) for _, llm in crews]
    assert all("sdk:q" in seen for seen in results), results
    assert len(offloaded_on_caller_loop) == loop_pool_workers


def test_kickoff_worker_thread_is_not_a_loop_default_executor_thread() -> None:
    """The kickoff body runs on the dedicated pool so a parked kickoff never holds a
    thread the loop's default executor needs."""
    observed: list[str] = []

    def fake_kickoff(self: Any, inputs: Any = None, input_files: Any = None) -> Any:
        observed.append(threading.current_thread().name)
        return CrewOutput(raw="done", tasks_output=[], token_usage=UsageMetrics())

    async def scenario() -> None:
        stub_agent = Agent(
            role="worker",
            goal="noop",
            backstory="agent",
            llm=ScriptedLLM(["unused"]),
            tools=[],
            allow_delegation=False,
        )
        task = Task(description="noop", expected_output="noop", agent=stub_agent)
        crew = Crew(agents=[stub_agent], tasks=[task], process=Process.sequential)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(Crew, "kickoff", fake_kickoff)
            await crew.kickoff_async()

    asyncio.run(scenario())

    assert observed and observed[0].startswith("crewai-kickoff"), observed
