"""Benchmarks for crewai's async execution path (``Crew.akickoff``).

Throwaway investigation harness, not a test module (no ``test_`` prefix, so
pytest does not collect it). Every scenario runs in a fresh subprocess so
thread counts, RSS and the global event bus start clean.

Run everything and write a JSON + markdown summary::

    uv run python lib/crewai/tests/perf/bench_async_path.py --label baseline

Run a single scenario in-process (what the driver does per subprocess)::

    uv run python lib/crewai/tests/perf/bench_async_path.py --one scaling \
        --n 50 --latency 0.2 --variant tools
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import contextvars
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import threading
import time
from typing import Any


os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("CREWAI_STORAGE_DIR", "/tmp/crewai-perf-storage")

HERE = Path(__file__).resolve().parent

# --------------------------------------------------------------------------
# Instrumentation shared by all scenarios
# --------------------------------------------------------------------------

_thread_starts = 0
_orig_thread_start = threading.Thread.start


def _counting_start(self: threading.Thread) -> None:
    global _thread_starts
    _thread_starts += 1
    _orig_thread_start(self)


threading.Thread.start = _counting_start  # type: ignore[method-assign]


class Counters:
    lock = threading.Lock()
    sync_llm_calls = 0
    async_llm_calls = 0
    tool_calls = 0
    llm_call_times: list[float] = []
    tool_call_times: list[float] = []
    llm_done_times: list[float] = []
    tool_done_times: list[float] = []

    @classmethod
    def done(cls, kind: str) -> None:
        with cls.lock:
            (cls.llm_done_times if kind == "llm" else cls.tool_done_times).append(
                time.perf_counter()
            )

    @classmethod
    def llm(cls, is_async: bool) -> None:
        with cls.lock:
            if is_async:
                cls.async_llm_calls += 1
            else:
                cls.sync_llm_calls += 1
            cls.llm_call_times.append(time.perf_counter())

    @classmethod
    def tool(cls) -> None:
        with cls.lock:
            cls.tool_calls += 1
            cls.tool_call_times.append(time.perf_counter())


CTX_VAR: contextvars.ContextVar[str] = contextvars.ContextVar(
    "bench_ctx", default="<unset>"
)
SEEN_CTX: dict[str, set[str]] = {"tool": set(), "handler": set(), "llm": set()}


def _build_crew(variant: str, latency: float, tool_latency: float, n_tools: int):
    from crewai import Agent, Crew, Task
    from crewai.llms.base_llm import BaseLLM
    from crewai.tools import BaseTool

    def _has_tool_result(messages: Any) -> bool:
        return isinstance(messages, list) and any(
            isinstance(m, dict) and m.get("role") == "tool" for m in messages
        )

    def _respond(messages: Any, tools: Any, available_functions: Any = None) -> Any:
        SEEN_CTX["llm"].add(CTX_VAR.get())
        if available_functions and "create_reasoning_plan" in available_functions:
            return json.dumps(
                {
                    "plan": "three independent steps",
                    "steps": [
                        {"step_number": k, "description": f"step {k}", "depends_on": []}
                        for k in (1, 2, 3)
                    ],
                    "ready": True,
                }
            )
        if tools and variant in ("tools", "planning") and not _has_tool_result(messages):
            return [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {"name": f"async_tool_{i}", "arguments": "{}"},
                }
                for i in range(n_tools)
            ]
        if tools:
            return "done"
        return "Thought: I know the answer\nFinal Answer: done"

    class MockLLM(BaseLLM):
        def call(self, messages, tools=None, callbacks=None, available_functions=None,
                 from_task=None, from_agent=None, response_model=None):  # noqa: D401
            Counters.llm(is_async=False)
            if latency:
                time.sleep(latency)
            Counters.done("llm")
            return _respond(messages, tools, available_functions)

        async def acall(self, messages, tools=None, callbacks=None,
                        available_functions=None, from_task=None, from_agent=None,
                        response_model=None):
            Counters.llm(is_async=True)
            if latency:
                await asyncio.sleep(latency)
            Counters.done("llm")
            return _respond(messages, tools, available_functions)

        def supports_function_calling(self) -> bool:
            return True

        def supports_stop_words(self) -> bool:
            return False

        def get_context_window_size(self) -> int:
            return 128000

    def make_tool(i: int) -> BaseTool:
        class AsyncTool(BaseTool):
            name: str = f"async_tool_{i}"
            description: str = f"Async benchmark tool {i}"

            async def _run(self) -> str:  # type: ignore[override]
                return await self._arun()

            async def _arun(self) -> str:  # type: ignore[override]
                Counters.tool()
                SEEN_CTX["tool"].add(CTX_VAR.get())
                if tool_latency:
                    await asyncio.sleep(tool_latency)
                Counters.done("tool")
                return f"result {i}"

        return AsyncTool()

    tools = (
        [make_tool(i) for i in range(n_tools)]
        if variant in ("tools", "planning")
        else []
    )
    agents = [
        Agent(
            role=f"Agent {k}",
            goal="Answer",
            backstory="Benchmark agent",
            llm=MockLLM(model="mock-model"),
            tools=tools if k == 0 else [],
            planning=variant == "planning" and k == 0,
            verbose=False,
        )
        for k in range(2)
    ]
    tasks = [
        Task(description=f"Task {k}", expected_output="done", agent=agents[k])
        for k in range(2)
    ]
    return Crew(agents=agents, tasks=tasks, verbose=False)


class LoopProbe:
    """Measure event-loop lag: how late a 10 ms sleep wakes up."""

    def __init__(self) -> None:
        self.lags: list[float] = []
        self._task: asyncio.Task[None] | None = None

    async def _run(self) -> None:
        while True:
            t0 = time.perf_counter()
            await asyncio.sleep(0.01)
            self.lags.append(time.perf_counter() - t0 - 0.01)

    def start(self) -> None:
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> dict[str, float]:
        assert self._task is not None
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        lags = sorted(self.lags) or [0.0]
        return {
            "loop_lag_max_ms": round(lags[-1] * 1000, 1),
            "loop_lag_p99_ms": round(lags[int(len(lags) * 0.99) - 1 if len(lags) > 1 else 0] * 1000, 1),
            "loop_lag_mean_ms": round(statistics.fmean(lags) * 1000, 2),
        }


class ThreadSampler:
    def __init__(self) -> None:
        self.peak = threading.active_count()
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, threading.active_count() - 1)
            time.sleep(0.002)

    def __enter__(self) -> ThreadSampler:
        _orig_thread_start(self._t)
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        self._t.join()


def _rss_mb() -> float:
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1)


# --------------------------------------------------------------------------
# Scenarios (each runs in its own process)
# --------------------------------------------------------------------------


async def scenario_scaling(args: argparse.Namespace) -> dict[str, Any]:
    crews = [
        _build_crew(args.variant, args.latency, args.tool_latency, args.n_tools)
        for _ in range(args.n)
    ]
    if args.small_executor:
        asyncio.get_running_loop().set_default_executor(
            ThreadPoolExecutor(args.small_executor)
        )
    # Warm-up kickoff so import/lazy-init costs are excluded.
    warm = _build_crew(args.variant, 0, 0, args.n_tools)
    await warm.akickoff()
    Counters.sync_llm_calls = Counters.async_llm_calls = Counters.tool_calls = 0
    starts_before = _thread_starts
    rss_before = _rss_mb()

    probe = LoopProbe()
    probe.start()
    with ThreadSampler() as sampler:
        t0 = time.perf_counter()
        results = await asyncio.gather(*(c.akickoff() for c in crews))
        wall = time.perf_counter() - t0
    lag = await probe.stop()
    assert all(r.raw for r in results)
    return {
        "wall_s": round(wall, 3),
        "throughput_kickoffs_per_s": round(args.n / wall, 2),
        "peak_threads": sampler.peak,
        "thread_starts": _thread_starts - starts_before,
        "peak_rss_mb": _rss_mb(),
        "rss_growth_mb": round(_rss_mb() - rss_before, 1),
        "sync_llm_calls": Counters.sync_llm_calls,
        "async_llm_calls": Counters.async_llm_calls,
        "tool_calls": Counters.tool_calls,
        **lag,
    }


async def scenario_overhead(args: argparse.Namespace) -> dict[str, Any]:
    crew = _build_crew(args.variant, 0, 0, args.n_tools)
    await crew.akickoff()  # warm-up
    times = []
    starts_before = _thread_starts
    for _ in range(args.n):
        t0 = time.perf_counter()
        await crew.akickoff()
        times.append(time.perf_counter() - t0)
    return {
        "per_kickoff_ms_median": round(statistics.median(times) * 1000, 2),
        "per_kickoff_ms_mean": round(statistics.fmean(times) * 1000, 2),
        "thread_starts_per_kickoff": round((_thread_starts - starts_before) / args.n, 1),
    }


async def scenario_cancel(args: argparse.Namespace) -> dict[str, Any]:
    crew = _build_crew(args.variant, args.latency, args.tool_latency, args.n_tools)
    await _build_crew(args.variant, 0, 0, args.n_tools).akickoff()  # warm-up
    Counters.sync_llm_calls = Counters.async_llm_calls = Counters.tool_calls = 0
    for lst in (Counters.llm_call_times, Counters.tool_call_times,
                Counters.llm_done_times, Counters.tool_done_times):
        lst.clear()
    task = asyncio.create_task(crew.akickoff())
    await asyncio.sleep(args.cancel_after)
    t_cancel = time.perf_counter()
    task.cancel()
    try:
        await task
        cancel_outcome = "completed (cancel ignored)"
    except asyncio.CancelledError:
        cancel_outcome = "CancelledError"
    except Exception as e:  # noqa: BLE001
        cancel_outcome = f"{type(e).__name__}: {e}"
    t_returned = time.perf_counter() - t_cancel
    await asyncio.sleep(args.latency * 6 + args.tool_latency * 4 + 0.5)
    return {
        "cancel_outcome": cancel_outcome,
        "await_returned_after_ms": round(t_returned * 1000, 1),
        "llm_started_after_cancel": sum(1 for t in Counters.llm_call_times if t > t_cancel),
        "llm_finished_after_cancel": sum(1 for t in Counters.llm_done_times if t > t_cancel),
        "tool_started_after_cancel": sum(1 for t in Counters.tool_call_times if t > t_cancel),
        "tool_finished_after_cancel": sum(1 for t in Counters.tool_done_times if t > t_cancel),
        "llm_started_total": len(Counters.llm_call_times),
        "tool_started_total": len(Counters.tool_call_times),
    }


async def scenario_context(args: argparse.Namespace) -> dict[str, Any]:
    from crewai.events.event_bus import crewai_event_bus
    from crewai.events.types.tool_usage_events import ToolUsageStartedEvent
    from crewai.events.types.llm_events import LLMCallStartedEvent

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(ToolUsageStartedEvent)
        def _h(source: Any, event: Any) -> None:
            SEEN_CTX["handler"].add(CTX_VAR.get())

        @crewai_event_bus.on(LLMCallStartedEvent)
        async def _ah(source: Any, event: Any) -> None:
            SEEN_CTX["handler"].add("async:" + CTX_VAR.get())

        crew = _build_crew("tools", 0.0, 0.0, 2)
        CTX_VAR.set("set-before-akickoff")
        await crew.akickoff()
        crewai_event_bus.flush()
    return {k: sorted(v) for k, v in SEEN_CTX.items()}


SCENARIOS = {
    "scaling": scenario_scaling,
    "overhead": scenario_overhead,
    "cancel": scenario_cancel,
    "context": scenario_context,
}


def run_one(args: argparse.Namespace) -> None:
    result = asyncio.run(SCENARIOS[args.one](args))
    print("RESULT_JSON " + json.dumps(result))


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def _plan(quick: bool) -> list[tuple[str, list[str]]]:
    ns = [1, 10, 50] if quick else [1, 10, 50, 200]
    plan: list[tuple[str, list[str]]] = []
    for variant in ("plain", "tools"):
        for n in ns:
            plan.append(
                (
                    f"scaling/{variant}/lat=200ms/N={n}",
                    ["--one", "scaling", "--variant", variant, "--n", str(n),
                     "--latency", "0.2", "--tool-latency", "0.1"],
                )
            )
    for n in (1, 50):
        plan.append(
            (
                f"scaling/planning/lat=200ms/N={n}",
                ["--one", "scaling", "--variant", "planning", "--n", str(n),
                 "--latency", "0.2", "--tool-latency", "0.1", "--n-tools", "2"],
            )
        )
    plan.append(
        (
            "scaling/plain/lat=500ms/N=200" if not quick else "scaling/plain/lat=500ms/N=50",
            ["--one", "scaling", "--variant", "plain", "--n", "50" if quick else "200",
             "--latency", "0.5"],
        )
    )
    for variant in ("plain", "tools"):
        plan.append(
            (
                f"overhead/{variant}/lat=0",
                ["--one", "overhead", "--variant", variant, "--n", "30"],
            )
        )
    plan.append(
        (
            "small_executor(4)/plain/lat=200ms/N=20",
            ["--one", "scaling", "--variant", "plain", "--n", "20", "--latency", "0.2",
             "--small-executor", "4"],
        )
    )
    for phase, after in (("during-llm", "0.1"), ("during-tools", "0.35")):
        plan.append(
            (
                f"cancel/{phase}/llm=200ms,tool=300ms",
                ["--one", "cancel", "--variant", "tools", "--latency", "0.2",
                 "--tool-latency", "0.3", "--cancel-after", after],
            )
        )
    plan.append(("context/tools", ["--one", "context"]))
    return plan


def drive(args: argparse.Namespace) -> None:
    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)
    results: dict[str, Any] = {}
    for name, extra in _plan(args.quick):
        if args.only and not any(o in name for o in args.only.split(",")):
            continue
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, __file__, *extra],
            capture_output=True,
            text=True,
            timeout=900,
        )
        line = next(
            (ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT_JSON ")),
            None,
        )
        if line is None:
            results[name] = {"error": (proc.stderr or proc.stdout)[-2000:]}
        else:
            results[name] = json.loads(line[len("RESULT_JSON "):])
        print(f"{name:45s} {time.perf_counter() - t0:6.1f}s  {results[name]}", flush=True)
    path = out_dir / f"{args.label}.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"wrote {path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--one", choices=sorted(SCENARIOS))
    p.add_argument("--label", default="run")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--only", default="")
    p.add_argument("--variant", default="plain", choices=["plain", "tools", "planning"])
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--latency", type=float, default=0.2)
    p.add_argument("--tool-latency", type=float, default=0.1)
    p.add_argument("--n-tools", type=int, default=4)
    p.add_argument("--small-executor", type=int, default=0)
    p.add_argument("--cancel-after", type=float, default=0.3)
    args = p.parse_args()
    if args.one:
        run_one(args)
    else:
        drive(args)


if __name__ == "__main__":
    main()
