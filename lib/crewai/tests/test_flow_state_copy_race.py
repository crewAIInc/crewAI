"""Regression tests for concurrent state mutation during Flow state copies.

Plain (``def``) flow methods run on worker threads while the event loop
deep-copies the flow state for events. When a plain method adds keys to an
unstructured (dict) state while another method starts, ``copy.deepcopy``
raises ``RuntimeError: dictionary changed size during iteration`` and the
flow fails even though neither method did anything wrong.
"""

import copy
import threading

import pytest

from crewai.flow.flow import Flow, listen, start


class DictStateFlow(Flow[dict]):
    @start()
    def begin(self):
        pass


def test_copy_state_retries_after_concurrent_resize(monkeypatch):
    """A RuntimeError from deepcopy mid-iteration is retried, not raised."""
    flow = DictStateFlow(initial_state={"answer_0": 0, "answer_1": 1})

    real_deepcopy = copy.deepcopy
    calls = {"count": 0}

    def flaky_deepcopy(obj, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("dictionary changed size during iteration")
        return real_deepcopy(obj, *args, **kwargs)

    monkeypatch.setattr("crewai.flow.runtime.copy.deepcopy", flaky_deepcopy)

    copied = flow._copy_state()

    assert copied["answer_0"] == 0
    assert copied["answer_1"] == 1
    assert calls["count"] == 2


def test_copy_state_retry_keeps_shallow_fallback_for_unpickleable_values(monkeypatch):
    """If the retried deepcopy of the snapshot hits unpickleable values
    (TypeError), the shallow snapshot is returned instead of raising."""
    flow = DictStateFlow(initial_state={"answer_0": 0, "answer_1": 1})

    real_deepcopy = copy.deepcopy
    calls = {"count": 0}

    def flaky_deepcopy(obj, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("dictionary changed size during iteration")
        raise TypeError("cannot pickle '_thread.RLock' object")

    monkeypatch.setattr("crewai.flow.runtime.copy.deepcopy", flaky_deepcopy)

    copied = flow._copy_state()

    assert isinstance(copied, dict)
    assert copied["answer_0"] == 0
    assert copied["answer_1"] == 1
    assert calls["count"] == 2


def test_copy_state_survives_concurrent_state_writes():
    """Repeated _copy_state calls while another thread grows the state never raise."""
    flow = DictStateFlow(initial_state={})
    stop = threading.Event()
    total_keys = 5_000

    def writer():
        for index in range(total_keys):
            if stop.is_set():
                return
            flow._state[f"answer_{index}"] = index

    writer_thread = threading.Thread(target=writer)
    writer_thread.start()
    try:
        while writer_thread.is_alive():
            copied = flow._copy_state()
            assert isinstance(copied, dict)
    finally:
        stop.set()
        writer_thread.join()

    assert len(flow._state) >= total_keys
    assert f"answer_{total_keys - 1}" in flow._state


@pytest.mark.timeout(120)
def test_flow_with_concurrent_listener_state_writes_completes():
    """End-to-end: a plain listener growing the state while sibling methods
    start must not fail the flow with 'dictionary changed size during iteration'."""

    class Survey(Flow[dict]):
        @start()
        def begin(self):
            pass

        @listen(begin)
        def collect(self):
            for index in range(20_000):
                self.state[f"answer_{index}"] = index

        @listen(begin)
        def step_1(self):
            pass

        @listen(step_1)
        def step_2(self):
            pass

        @listen(step_2)
        def step_3(self):
            pass

        @listen(step_3)
        def step_4(self):
            pass

    flow = Survey()
    flow.kickoff()

    assert "answer_19999" in flow.state
