import asyncio
import socket
import threading

import pytest

from crewai.utilities.network_guard import run_in_new_loop, unpatched_socket_connect


@pytest.fixture
def network_guard(monkeypatch):
    def _guard(self, address, *args, **kwargs):
        raise RuntimeError("Network is disabled")

    monkeypatch.setattr(socket.socket, "connect", _guard)
    monkeypatch.setattr(socket.socket, "connect_ex", _guard)


def test_run_in_new_loop_runs_coroutine_and_reinstates_guard(network_guard, monkeypatch):
    async def coro():
        return 42

    assert run_in_new_loop(coro()) == 42

    # The guard must be active again while the coroutine's result is used.
    with pytest.raises(RuntimeError, match="Network is disabled"):
        socket.socket().connect(("127.0.0.1", 1))
    monkeypatch.undo()


def test_run_in_new_loop_propagates_exceptions(network_guard, monkeypatch):
    async def coro():
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        run_in_new_loop(coro())
    monkeypatch.undo()


def test_concurrent_contexts_reinstate_guard(network_guard):
    # Overlapping entry/exit must not leave the pristine connect in place:
    # after every context has exited, the guard is active again.
    errors = []

    def worker():
        try:
            for _ in range(20):
                with unpatched_socket_connect():
                    pass
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert errors == []
    with pytest.raises(RuntimeError, match="Network is disabled"):
        socket.socket().connect(("127.0.0.1", 1))


def test_guard_stays_active_for_other_threads_during_construction(network_guard, monkeypatch):
    # While one thread builds a loop under the exemption, another thread's
    # real connect() attempts must still hit the guard.
    results = {}

    def other_thread():
        try:
            socket.socket().connect(("127.0.0.1", 1))
            results["allowed"] = True
        except RuntimeError as exc:
            results["guarded"] = str(exc)
        except OSError as exc:
            results["pristine"] = str(exc)

    with unpatched_socket_connect():
        loop = asyncio.new_event_loop()
        try:
            t = threading.Thread(target=other_thread)
            t.start()
            t.join(timeout=10)
        finally:
            loop.close()

    assert results.get("guarded") == "Network is disabled"
    monkeypatch.undo()


def test_run_in_new_loop_cancels_pending_tasks(network_guard, monkeypatch):
    finalizer_ran = []

    async def leaving_resource():
        try:
            await asyncio.sleep(30)
        finally:
            finalizer_ran.append(True)

    async def coro():
        task = asyncio.create_task(leaving_resource())
        await asyncio.sleep(0.01)
        return "done"

    assert run_in_new_loop(coro()) == "done"
    assert finalizer_ran == [True]
    monkeypatch.undo()


def test_run_in_new_loop_finalizes_async_generators(network_guard, monkeypatch):
    finalized = []

    async def agen():
        try:
            yield 1
        finally:
            finalized.append(True)

    async def coro():
        gen = agen()
        await gen.__anext__()  # left suspended; asyncio.run would finalize it

    run_in_new_loop(coro())
    assert finalized == [True]
    monkeypatch.undo()


def test_run_in_new_loop_reports_exceptions_from_cancelled_tasks(network_guard, monkeypatch):
    # A pending task that raises (not CancelledError) while handling its
    # cancellation must be reported to the loop's exception handler, mirroring
    # asyncio.run's shutdown behavior.
    reported = []

    async def stubborn():
        try:
            await asyncio.sleep(30)
        finally:
            raise ValueError("cleanup boom")

    async def coro():
        loop = asyncio.get_running_loop()
        loop.set_exception_handler(lambda loop, ctx: reported.append(ctx))
        asyncio.create_task(stubborn())
        await asyncio.sleep(0.01)
        return "done"

    assert run_in_new_loop(coro()) == "done"
    assert any(
        isinstance(ctx.get("exception"), ValueError)
        and "cleanup boom" in str(ctx["exception"])
        for ctx in reported
    )
    monkeypatch.undo()
