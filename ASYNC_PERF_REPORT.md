# Async execution path: is going thread-free worth it on CPython?

Throwaway investigation. Prototype commits are on this branch; nothing here is
meant to merge as-is.

## TL;DR

**Recommendation: partial go.** The thread hops listed in the brief (file store,
event bus, parallel tool pool, planning `to_thread`) are worth almost nothing
on their own. But the investigation found a much bigger problem the brief
didn't list: **`Crew.akickoff` with the default executor never calls
`LLM.acall`.** `AgentExecutor` is a `Flow` whose steps are sync methods, and
`Flow` runs every sync method via `asyncio.to_thread`. So every LLM call is a
blocking `LLM.call` on the loop's *default executor*. The number of concurrent
LLM calls across all `akickoff`s in a process is therefore capped at
`min(32, cpu+4)` (8 on the 4-core benchmark box). Fixing that one thing gives:

| 200 concurrent `akickoff` (2 agents, 2 tasks) | baseline | prototype |
|---|---|---|
| LLM latency 200 ms, wall time | 18.5 s | **6.9 s** (2.7×) |
| LLM latency 500 ms, wall time | 35.1 s | **7.2 s** (4.9×) |
| + 4 parallel async tools, 200 ms | 27.9 s | **9.2 s** (3.0×) |
| 50 crews with planning (3 parallel steps) | 16.5 s | **4.7 s** (3.5×) |
| 20 crews, default executor shrunk to 4 workers | 2.69 s | **0.91 s** |
| Cancel mid-run: LLM/tool calls that keep running | 1 LLM / 4 tools | **0 / 0** |
| Threads started per kickoff | 7–11 | **0** |

Two things are **not** worth doing as proposed:

- **Event bus rework.** Measured no effect. On the `akickoff` path, `aemit` is
  never used; `emit()` hands off to a long-lived 10-thread pool and doesn't
  wait. The double hop the brief describes is only taken for handlers
  registered with `depends_on`, and it happens on the bus's private loop, not
  the caller's.
- **Moving the task-output SQLite log off the loop (P6).** It improves loop lag
  but halves throughput at high concurrency.

The caveat: at small concurrency (1–10 crews) with real LLM latency the user
won't notice anything except cancellation. The wins appear once the number of
in-flight LLM calls exceeds the default executor size, i.e. ~8–32 concurrent
calls per process. That threshold is easy to reach for a server that runs
`akickoff` per request.

## Setup

- CPython 3.13.16, 4 vCPU (default executor = 8 workers), `uv sync --all-packages`.
- Harness: `lib/crewai/tests/perf/bench_async_path.py`, one command:
  `uv run python lib/crewai/tests/perf/bench_async_path.py --label <name>`
  (`--quick` / `--only <substr>` available). Each scenario runs in its own
  subprocess. Raw JSON for every run is in `lib/crewai/tests/perf/results/`.
- Mock `BaseLLM` subclass: `call` does `time.sleep(latency)`, `acall` does
  `asyncio.sleep(latency)`; both count invocations, so the harness reports which
  path crewai actually took. The "tools" variant returns 4 native tool calls to
  async `BaseTool`s (`_arun` = `asyncio.sleep(100 ms)`), then a final answer.
  The "planning" variant uses `planning=True` with a 3-step, dependency-free plan.
- Metrics: wall time, throughput, peak `threading.active_count()` (2 ms
  sampler), `Thread.start` calls (patched), peak RSS (`ru_maxrss`), event-loop
  lag (drift of a 10 ms `asyncio.sleep` probe).
- Telemetry/tracing disabled; `verbose=False`.

## Thread hops actually on the `akickoff` path (baseline)

From the brief, confirmed:

1. File store `_run_sync`: new `ThreadPoolExecutor` + `asyncio.run` per call.
   Not 2 calls per kickoff but **~7**: `Crew._prepare_tools`, `Task.prompt()`
   and `Task._aexecute_core`'s `finally` also hit it, every task, even when no
   files were ever stored.
2. Event bus: see above. Measured as not mattering.
3. Planning steps: `asyncio.to_thread(step_executor.execute)`, plus
   `execute_todo_sequential` as a sync Flow step.
4. Parallel native tools: a fresh `ThreadPoolExecutor(≤8)` per batch. Inside
   it, every *async* tool is run via `BaseTool.run` → `asyncio.run(...)`: a new
   event loop per tool call.
5. `Crew.kickoff_async` = `to_thread(kickoff)`: noted, not changed.

Not in the brief, and the ones that matter:

6. **Every `AgentExecutor` step is a sync Flow method → `asyncio.to_thread`**,
   including `call_llm_native_tools` / `call_llm_and_parse`, which call the
   *sync* `get_llm_response` → `LLM.call`. Every pure routing step
   (`check_max_iterations`, `route_by_answer_type`, …) also takes a hop, and
   every step takes a second hop for `_persist_method_completion` even when
   persistence is off. This is the default-executor cap.
7. `_create_crew_output` calls the blocking `crewai_event_bus.flush()` **on the
   loop**, and it waits on handler futures from *all* concurrent crews.
8. `_store_execution_log` (every task) and `TaskOutputStorageHandler.reset`
   (every kickoff) do a SQLite transaction behind a cross-process file lock
   (`store_lock`) **on the loop**. In a cProfile of the prototype this is
   ~60% of per-kickoff wall time at 0 ms LLM latency.

## Prototype changes (one commit each)

| id | change | files |
|---|---|---|
| P1 | `akickoff` uses `aget_all_files` (prefetched into `_prepare_tools`) and `aclear_files` | `crew.py`, `crews/utils.py` |
| P2 | `aflush()` on the bus; `akickoff` awaits it instead of the blocking `flush()`; `_emit_with_dependencies` awaits `wrap_future` (1 hop, not 2) | `event_bus.py`, `crew.py` |
| P3+P4 | `Flow._async_step_override` hook. `AgentExecutor` (only when entered via `invoke_async`) runs async twins of the LLM steps (`acall`, falling back to `to_thread(call)` for LLMs without `acall`) and of `execute_native_tool` (async tools awaited and `asyncio.gather`ed, sync tools `to_thread`). Pure routing steps run inline. Persist hop skipped when persistence is off. Single-call lifecycle refactored into a generator that yields at the one blocking point, so sync and async share one body | `flow/runtime/__init__.py`, `experimental/agent_executor.py`, `utilities/agent_utils.py` |
| P1b | File store remembers which ids ever stored files; lookups and clears for other ids return immediately (benefits the sync path too) | `utilities/file_store.py` |
| P5 | `StepExecutor` converted to the same generator pattern: `execute()` drives it synchronously, `aexecute()` awaits `acall` / async tools. Sequential and parallel plan steps use it on the async path | `agents/step_executor.py`, `experimental/agent_executor.py`, `utilities/agent_utils.py` |
| P6 | `akickoff` writes the per-task output log via `to_thread` | `crew.py` |

P3 (LLM steps) and P4 (parallel tools) were committed together because they
share the `execute_native_tool` restructuring. The "plain" variant has no tool
calls, so its column isolates P3; the "tools" variant shows P3+P4.

Cumulative: each column includes all previous ones. The P1b column's planning
rows were measured on the commit before P5.

### Wall time (s)

| scenario | baseline | P1 | P2 | P3+P4 | P1b | P5 | P6 |
|---|---|---|---|---|---|---|---|
| plain, 200 ms, N=1 | 0.45 | 0.44 | 0.46 | 0.44 | 0.43 | 0.44 | 0.44 |
| plain, 200 ms, N=10 | 0.99 | 0.97 | 0.97 | 0.68 | 0.66 | 0.63 | 1.87 |
| plain, 200 ms, N=50 | 4.23 | 4.08 | 4.10 | 1.80 | 1.57 | 1.65 | 2.24 |
| plain, 200 ms, N=200 | 18.48 | 17.28 | 17.58 | 7.57 | 6.69 | 6.85 | 7.39 |
| plain, 500 ms, N=200 | 35.10 | 32.61 | 32.81 | 7.70 | 7.10 | 7.17 | 8.10 |
| tools, 200 ms, N=1 | 0.77 | 0.76 | 0.76 | 0.76 | 0.77 | 0.74 | 0.75 |
| tools, 200 ms, N=10 | 1.25 | 1.25 | 1.21 | 1.02 | 0.97 | 0.98 | 1.69 |
| tools, 200 ms, N=50 | 6.22 | 6.33 | 6.16 | 2.40 | 2.22 | 2.17 | 3.02 |
| tools, 200 ms, N=200 | 27.89 | 27.07 | 26.96 | 9.34 | 8.90 | 9.22 | 17.21 |
| planning, 200 ms, N=1 | 1.27 | – | – | – | 1.25 | 1.24 | 1.25 |
| planning, 200 ms, N=50 | 16.52 | – | – | – | 15.23 | 4.70 | 5.30 |
| default executor = 4, N=20 | 2.69 | 2.69 | 2.69 | 0.92 | 0.92 | 0.91 | 1.53 |

The ideal for "plain" (2 sequential LLM calls) is ~0.4 s at any N. Once P3 is
in, the residual is crewai's own CPU work, not waiting (see "What's left").

### Per-kickoff overhead (LLM latency 0, sequential kickoffs, median)

| | baseline | P1 | P2 | P3+P4 | P1b | P5 | P6 |
|---|---|---|---|---|---|---|---|
| plain: ms / kickoff | 26.1 | 26.2 | 25.6 | 21.0 | 19.7 | 19.6 | 17.0 |
| plain: threads started / kickoff | 7.1 | 4.0 | 4.0 | 4.0 | 0.0 | 0.1 | 0.1 |
| tools: ms / kickoff | 36.6 | 36.1 | 37.4 | 25.6 | 22.0 | 24.9 | 23.0 |
| tools: threads started / kickoff | 11.1 | 8.1 | 8.0 | 4.2 | 0.2 | 0.2 | 0.2 |

### Threads and memory

| | baseline | after P5 |
|---|---|---|
| peak threads, tools N=200 | 49 | 20 |
| threads started, tools N=200 | 2215 | 13 |
| peak RSS, tools N=200 | 310 MB | 312 MB |

The pools are bounded, so thread counts never exploded, and RSS is unchanged.
Memory is not an argument either way.

### Event-loop lag (ms, p99 / max)

| scenario | baseline | P3+P4 | P5 | P6 |
|---|---|---|---|---|
| plain, 200 ms, N=50 | 89 / 771 | 144 / 758 | 113 / 644 | 14 / 683 |
| plain, 200 ms, N=200 | 72 / 3510 | 761 / 3163 | 499 / 2950 | 97 / 2920 |
| tools, 200 ms, N=200 | 39 / 3829 | 1172 / 3477 | 599 / 3512 | 70 / 3556 |

The multi-second max lag is the same in every column. It comes from the burst
of ~200 crew setups (`prepare_kickoff`, executor construction) running on the
loop at start. That is pure CPU and not thread-related. p99 lag goes **up**
once the executor is native: CPU work that used to be spread over worker threads
(still serialised by the GIL, but interleaved with the loop) now runs on the
loop itself. With P6 (SQLite off the loop) p99 falls back to baseline levels.

### Cancellation (`task.cancel()` on a running `akickoff`)

| | `CancelledError` surfaces after | calls still running to completion after cancel |
|---|---|---|
| baseline, cancel during LLM call | 2.8 ms | 1 LLM call (thread) |
| baseline, cancel during 4 parallel tools | 1.9 ms | 4 tool calls (thread pool + private loops) |
| P3+P4 onward | 0.2–1.5 ms | **0** (the `acall` / tool coroutines are cancelled) |

No new work starts after cancellation even on baseline. The fix is about work
that is *already in flight*: on baseline it keeps running for its full duration.
With real LLMs that means spending tokens on the result of a request the caller
has abandoned. Sync tools are still not cancellable, because they stay in
threads by design.

### Context propagation

`contextvars` set before `akickoff` are visible in the tool, the LLM call, a
sync handler and an async handler, both before and after every prototype
(`context/tools` row in each JSON).

## Which changes are worth it

| change | worth it alone? | risk | test fallout (vs main) |
|---|---|---|---|
| P1: async file store in `akickoff` | Marginal: −3 threads per kickoff, no wall-time change | Low | 0 |
| **P1b: file store short-circuit** | **Yes**, cheap: removes the remaining ~4–7 `ThreadPoolExecutor` + `asyncio.run` per kickoff on *both* sync and async paths | Low. Stale ids after TTL only cost a real lookup. Cross-process stores would need a rethink, but the store is in-memory today | 0 |
| P2: event bus `aflush` / single hop | **No measurable gain** at any concurrency. Keep `aflush` only as hygiene (it stops `akickoff` blocking the loop while slow handlers drain) | Low | 0 |
| **P3: executor LLM steps on the loop** | **Yes, the main win**: 2.7–4.9× throughput at N≥50, cancellation, no default-executor coupling | Medium (see behaviour changes) | 1 (source-inspection test) + 1 fixed during the prototype |
| **P4: gather async tools** | **Yes**, with P3: peak threads 49 → 20, tools N=200 27.9 → 9.2 s | Low/medium | shared with P3 |
| **P5: planning steps native** | **Yes** if planning matters: 3.2× at N=50 | Medium: the generator refactor changes private method shapes | 5 (tests that call or mock `_execute_native` etc. directly) + 1 source-inspection test |
| P6: SQLite log via `to_thread` | **No, as written**: p99 lag 499 → 97 ms but throughput 6.9 → 7.4 s (plain) and **9.2 → 17.2 s** (tools) at N=200, and N=10 gets 3× slower. Threads now contend on the file lock and on default-executor workers. The fix is to make the per-task replay log cheaper (batch per kickoff, async store, or opt-out), not to thread it | – | 0 |

All 7 failing tests are implementation-coupled (`inspect.getsource` checks or
mocks of private methods); none is a behaviour regression. I fixed one real
regression found along the way (`OutputParserError` raised by the LLM call
itself was no longer routed to `parser_error`).

Test-suite note: in this container `pytest lib/crewai/tests` already has 407
failures and 14 errors on `main`, from optional provider packages (litellm,
anthropic, gemini, azure) not being installed. Fallout above is the set
difference against that baseline: main 407 failed / 5198 passed; prototype 414
failed / 5191 passed; nothing fixed or broken beyond the 7 listed.
`ruff check` and `mypy` are clean on all 8 changed source files.

## Behaviour changes users could notice (P3–P5)

- **`LLM.acall` instead of `LLM.call` on `akickoff`.** Provider async paths
  differ from the sync ones (client objects, retry/timeout wiring, streaming).
  Users who customised `call` but not `acall` get the old behaviour via a
  `to_thread` fallback. Users who override both with different behaviour will
  see the `acall` one. The sync `kickoff()` is unchanged.
- **Hooks and callbacks now run on the caller's event loop thread.** Before/after
  LLM-call hooks, tool hooks, `step_callback` and `enforce_rpm_limit` used to
  run in a worker thread. A blocking user callback now stalls the loop, and so
  does RPM throttling (`enforce_rpm_limit` sleeps). The RPM limiter needs an
  async variant before shipping.
- **Async tools run on the caller's loop**, not a private `asyncio.run` loop in a
  thread. This is good for tools holding loop-bound clients (aiohttp sessions
  created on the app loop used to fail). An async tool that does blocking work
  inside `async def` now stalls the caller's loop.
- **`contextvars` writes inside tools and hooks now persist** across steps
  within the same kickoff. Before, each `to_thread` ran on a context copy, so
  mutations were lost. Reads behave the same either way.
- **Parallel tool batch aborts** (`ToolExecutionFailedError`) now cancel
  in-flight async siblings. Before, they could only drop not-yet-started ones.
- **Cancellation** now reaches `acall` / tool coroutines; providers and tools see
  `CancelledError` mid-call.
- **Exceptions**: same types and messages. Tracebacks are shorter (no thread
  boundary).
- **Event handler ordering**: unchanged. `emit()` was not touched, and sync
  handlers still run on the bus pool.
- **`CrewKickoffCompletedEvent` timing**: unchanged ordering. `akickoff` still
  drains memory writes and waits for *all* pending handlers (globally, across
  concurrent crews) before emitting. It now awaits instead of blocking the
  loop. The cross-crew coupling (one crew's completion waits on every other
  crew's handlers) is pre-existing and worth fixing separately.

## What's left after the prototype (the next bottleneck)

With latency out of the picture, a 2-task kickoff costs ~17–20 ms of CPU. In
cProfile, the biggest items are:

- SQLite + file-lock task-output log: ~60% of wall time on the async path.
- Flow listener resolution (`_find_triggered_methods` / `_condition_met`):
  ~10 ms per kickoff, from 24k condition checks for 20 kickoffs.
- `prepare_kickoff` per crew.

These bound throughput at roughly 25–30 kickoffs/s per process regardless of
threads, and they are now what drives loop lag. Reducing them helps both
`kickoff` and `akickoff`.

## Recommendation and PR breakdown

**Partial go.** Do the executor work (it is the real "async path"), skip the
event-bus rework, and fix the storage write differently.

1. **File store short-circuit (P1b + P1).** Small, low risk, helps sync too.
   Ship first.
2. **`Flow._async_step_override` + `AgentExecutor` async LLM steps (P3).**
   Include: `to_thread(call)` fallback for LLMs without `acall`, inline routing
   steps, skipping the persist hop when persistence is off, an async
   `enforce_rpm_limit`, and the `OutputParserError` routing fix. Update the one
   source-inspection test. Document the hooks-on-loop change.
3. **Native tool batches on the loop (P4).** Generator-based single-call
   lifecycle shared by sync and async, gathered async tools, sync tools via
   `to_thread`. Could be folded into PR 2.
4. **Planning `StepExecutor.aexecute` (P5).** Rewrite the 5 tests that mock
   private step methods to go through `execute()`. Then also convert the
   remaining sync planning LLM calls (plan generation, `PlannerObserver`,
   synthesis); 100 of the 450 planning LLM calls in the benchmark still go
   through threads.
5. **Separately:** make the per-task replay log cheap (one write per kickoff, or
   async/opt-out), and reduce Flow listener resolution cost. These are the next
   throughput limit for both sync and async.
6. **`aflush` (P2)**: optional hygiene, can ride along with PR 2. Not worth a
   PR by itself.

Out of scope and unchanged: `kickoff()`, `Task.execute_async`, the memory save
pool, `Crew.kickoff_async`. That last one still wraps the whole sync
`kickoff` in a thread; with PR 2 landed, pointing users at `akickoff` becomes
a real recommendation rather than a cosmetic one.
