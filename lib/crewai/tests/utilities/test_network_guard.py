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
