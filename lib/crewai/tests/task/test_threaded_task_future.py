from concurrent.futures import wait
from threading import Event, Thread
from unittest.mock import patch

import pytest

from crewai import Agent, Task


@pytest.fixture
def agent() -> Agent:
    return Agent(role="Worker", goal="Finish the task", backstory="A test worker")


@pytest.mark.parametrize("fails", [False, True])
def test_running_task_future_cannot_be_cancelled(agent: Agent, fails: bool) -> None:
    started = Event()
    release = Event()

    def execute_task(**kwargs: object) -> str:
        started.set()
        assert release.wait(timeout=10)
        if fails:
            raise RuntimeError("task failed")
        return "done"

    task = Task(description="Work", expected_output="A result", agent=agent)
    with patch.object(Agent, "execute_task", side_effect=execute_task):
        future = task.execute_async()
        try:
            assert started.wait(timeout=10)
            assert future.running()
            assert not future.cancel()
        finally:
            release.set()
            if fails:
                with pytest.raises(RuntimeError, match="task failed"):
                    future.result(timeout=10)
            else:
                assert future.result(timeout=10).raw == "done"

    assert future.done()
    assert not future.cancelled()


def test_task_cancelled_before_worker_starts_does_not_execute(agent: Agent) -> None:
    release = Event()
    threads: list[Thread] = []

    class DelayedThread(Thread):
        def start(self) -> None:
            threads.append(self)
            super().start()

        def run(self) -> None:
            assert release.wait(timeout=10)
            super().run()

    task = Task(description="Work", expected_output="A result", agent=agent)
    with (
        patch("crewai.task.threading.Thread", DelayedThread),
        patch.object(Agent, "execute_task", return_value="unexpected") as execute_task,
    ):
        future = task.execute_async()
        try:
            assert future.cancel()
        finally:
            release.set()
            for thread in threads:
                thread.join(timeout=10)
                assert not thread.is_alive()

    execute_task.assert_not_called()
    assert task.start_time is None
    assert task.output is None
    assert future.cancelled()
    done, pending = wait([future], timeout=0)
    assert future in done
    assert not pending
