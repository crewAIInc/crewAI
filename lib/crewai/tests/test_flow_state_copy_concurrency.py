"""Flow state snapshots must survive a concurrent write to a dict state."""

import copy

import pytest

from crewai.flow import Flow, listen, start


class _Growth:
    """A value whose deepcopy writes a new key into the flow state.

    Listener methods run on worker threads and write into the dict state
    directly, so a snapshot taken while another method starts can walk a dict
    that grows mid-iteration. Growing it from inside the copy reproduces that
    race without depending on thread timing.
    """

    def __init__(self, state):
        self.state = state

    def __deepcopy__(self, memo):
        self.state["added_during_copy"] = True
        return True


def _flow_holding(state):
    class Holder(Flow):
        @start()
        def run(self):
            return "done"

    flow = Holder()
    flow._state = state
    return flow


def test_copy_state_handles_dict_growing_during_iteration():
    state = {"id": "flow-1"}
    state["growth"] = _Growth(state)
    flow = _flow_holding(state)

    snapshot = flow._copy_state()

    assert snapshot["id"] == "flow-1"
    # The write happened, but after the entries were taken, so it is not in the
    # snapshot. What matters is that taking the snapshot did not raise.
    assert state["added_during_copy"] is True
    assert "added_during_copy" not in snapshot
    # The copy is detached from the live state.
    snapshot["id"] = "changed"
    assert flow._state["id"] == "flow-1"


def test_copy_state_keeps_original_without_mutation():
    flow = _flow_holding({"id": "flow-2", "answer": 42})

    snapshot = flow._copy_state()

    assert snapshot == {"id": "flow-2", "answer": 42}
    assert snapshot is not flow._state


def test_copy_state_still_copies_nested_objects_deeply():
    nested = {"inner": [1, 2, 3]}
    flow = _flow_holding({"id": "flow-3", "nested": nested})

    snapshot = flow._copy_state()
    snapshot["nested"]["inner"].append(4)

    assert flow._state["nested"]["inner"] == [1, 2, 3]


def test_copy_state_propagates_unrelated_runtime_errors(monkeypatch):
    flow = _flow_holding({"id": "flow-4"})

    def boom(value, memo=None):
        raise RuntimeError("something else broke")

    monkeypatch.setattr(copy, "deepcopy", boom)

    with pytest.raises(RuntimeError, match="something else broke"):
        flow._copy_state()


def test_flow_completes_when_state_grows_while_snapshotting():
    class Survey(Flow):
        @start()
        def begin(self):
            return True

        @listen(begin)
        def collect(self):
            for i in range(2_000):
                self.state[f"answer_{i}"] = i

        @listen(begin)
        def step_1(self):
            self.state["growth"] = _Growth(self.state)
            return "step_1"

        @listen(step_1)
        def step_2(self):
            return "step_2"

    result = Survey().kickoff()

    assert result == "step_2"

