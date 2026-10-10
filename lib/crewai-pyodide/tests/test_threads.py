from __future__ import annotations

import pytest

from crewai_pyodide._threads import InlineExecutor


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
