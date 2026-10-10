from __future__ import annotations

import pytest

from crewai_pyodide._threads import InlineExecutor, install_inline_threads


def test_submit_runs_immediately_and_returns_a_done_future() -> None:
    calls: list[int] = []
    future = InlineExecutor().submit(lambda x: calls.append(x) or x * 2, 21)

    assert calls == [21]
    assert future.done()
    assert future.result() == 42


def test_exceptions_are_captured_on_the_future() -> None:
    def boom() -> None:
        raise ValueError("nope")

    future = InlineExecutor().submit(boom)

    with pytest.raises(ValueError, match="nope"):
        future.result()


def test_works_as_a_context_manager_and_refuses_work_after_shutdown() -> None:
    with InlineExecutor(max_workers=4) as executor:
        assert executor.submit(sum, [1, 2]).result() == 3

    with pytest.raises(RuntimeError):
        executor.submit(sum, [1])


@pytest.fixture
def inline_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    import threading

    monkeypatch.setattr(threading.Thread, "start", threading.Thread.start)
    monkeypatch.setattr(threading.Thread, "join", threading.Thread.join)
    monkeypatch.setattr(threading.Timer, "start", threading.Timer.start, raising=False)
    import concurrent.futures

    monkeypatch.setattr(
        concurrent.futures, "ThreadPoolExecutor", concurrent.futures.ThreadPoolExecutor
    )
    monkeypatch.setattr(
        concurrent.futures.thread,
        "ThreadPoolExecutor",
        concurrent.futures.thread.ThreadPoolExecutor,
    )
    install_inline_threads()


def test_thread_errors_go_to_excepthook_not_the_caller(
    inline_threads: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    hooked: list[type[BaseException]] = []
    monkeypatch.setattr(threading, "excepthook", lambda args: hooked.append(args.exc_type))

    def boom() -> None:
        raise ValueError("nope")

    threading.Thread(target=boom).start()

    assert hooked == [ValueError]


@pytest.mark.asyncio
async def test_timers_fire_later_without_blocking(inline_threads: None) -> None:
    import asyncio
    import threading

    fired = asyncio.Event()
    threading.Timer(0.01, fired.set).start()
    assert not fired.is_set()
    await asyncio.wait_for(fired.wait(), timeout=5)

    cancelled: list[int] = []
    timer = threading.Timer(0.01, cancelled.append, args=(1,))
    timer.start()
    timer.cancel()
    await asyncio.sleep(0.05)
    assert cancelled == []
