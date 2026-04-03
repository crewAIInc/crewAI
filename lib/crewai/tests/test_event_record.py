"""Tests for EventRecord data structure and RuntimeState integration."""

from __future__ import annotations

import json

import pytest

from crewai.events.base_events import BaseEvent
from crewai.state.event_record import EventRecord, EventNode


# ── Helpers ──────────────────────────────────────────────────────────


def _event(type: str, **kwargs) -> BaseEvent:
    return BaseEvent(type=type, **kwargs)


def _linear_record(n: int = 5) -> tuple[EventRecord, list[BaseEvent]]:
    """Build a simple chain: e0 → e1 → e2 → ... with previous_event_id."""
    g = EventRecord()
    events: list[BaseEvent] = []
    for i in range(n):
        e = _event(
            f"step_{i}",
            previous_event_id=events[-1].event_id if events else None,
            emission_sequence=i + 1,
        )
        events.append(e)
        g.add(e)
    return g, events


def _tree_record() -> tuple[EventRecord, dict[str, BaseEvent]]:
    """Build a parent/child tree:

        crew_start
        ├── task_start
        │   ├── agent_start
        │   └── agent_complete (started=agent_start)
        └── task_complete (started=task_start)
    """
    g = EventRecord()
    crew_start = _event("crew_kickoff_started", emission_sequence=1)
    task_start = _event(
        "task_started",
        parent_event_id=crew_start.event_id,
        previous_event_id=crew_start.event_id,
        emission_sequence=2,
    )
    agent_start = _event(
        "agent_execution_started",
        parent_event_id=task_start.event_id,
        previous_event_id=task_start.event_id,
        emission_sequence=3,
    )
    agent_complete = _event(
        "agent_execution_completed",
        parent_event_id=task_start.event_id,
        previous_event_id=agent_start.event_id,
        started_event_id=agent_start.event_id,
        emission_sequence=4,
    )
    task_complete = _event(
        "task_completed",
        parent_event_id=crew_start.event_id,
        previous_event_id=agent_complete.event_id,
        started_event_id=task_start.event_id,
        emission_sequence=5,
    )

    for e in [crew_start, task_start, agent_start, agent_complete, task_complete]:
        g.add(e)

    return g, {
        "crew_start": crew_start,
        "task_start": task_start,
        "agent_start": agent_start,
        "agent_complete": agent_complete,
        "task_complete": task_complete,
    }


# ── EventNode tests ─────────────────────────────────────────────────


class TestEventNode:
    def test_add_edge(self):
        node = EventNode(event=_event("test"))
        node.add_edge("child", "abc")
        assert node.neighbors("child") == ["abc"]

    def test_neighbors_empty(self):
        node = EventNode(event=_event("test"))
        assert node.neighbors("parent") == []

    def test_multiple_edges_same_type(self):
        node = EventNode(event=_event("test"))
        node.add_edge("child", "a")
        node.add_edge("child", "b")
        assert node.neighbors("child") == ["a", "b"]


# ── EventRecord core tests ───────────────────────────────────────────


class TestEventRecordCore:
    def test_add_single_event(self):
        g = EventRecord()
        e = _event("test")
        node = g.add(e)
        assert len(g) == 1
        assert e.event_id in g
        assert node.event.type == "test"

    def test_get_existing(self):
        g = EventRecord()
        e = _event("test")
        g.add(e)
        assert g.get(e.event_id) is not None

    def test_get_missing(self):
        g = EventRecord()
        assert g.get("nonexistent") is None

    def test_contains(self):
        g = EventRecord()
        e = _event("test")
        g.add(e)
        assert e.event_id in g
        assert "missing" not in g


# ── Edge wiring tests ───────────────────────────────────────────────


class TestEdgeWiring:
    def test_parent_child_bidirectional(self):
        g = EventRecord()
        parent = _event("parent")
        child = _event("child", parent_event_id=parent.event_id)
        g.add(parent)
        g.add(child)

        parent_node = g.get(parent.event_id)
        child_node = g.get(child.event_id)
        assert child.event_id in parent_node.neighbors("child")
        assert parent.event_id in child_node.neighbors("parent")

    def test_previous_next_bidirectional(self):
        g, events = _linear_record(3)
        node0 = g.get(events[0].event_id)
        node1 = g.get(events[1].event_id)
        node2 = g.get(events[2].event_id)

        assert events[1].event_id in node0.neighbors("next")
        assert events[0].event_id in node1.neighbors("previous")
        assert events[2].event_id in node1.neighbors("next")
        assert events[1].event_id in node2.neighbors("previous")

    def test_trigger_bidirectional(self):
        g = EventRecord()
        cause = _event("cause")
        effect = _event("effect", triggered_by_event_id=cause.event_id)
        g.add(cause)
        g.add(effect)

        assert effect.event_id in g.get(cause.event_id).neighbors("trigger")
        assert cause.event_id in g.get(effect.event_id).neighbors("triggered_by")

    def test_started_completed_by_bidirectional(self):
        g = EventRecord()
        start = _event("start")
        end = _event("end", started_event_id=start.event_id)
        g.add(start)
        g.add(end)

        assert end.event_id in g.get(start.event_id).neighbors("completed_by")
        assert start.event_id in g.get(end.event_id).neighbors("started")

    def test_dangling_reference_ignored(self):
        """Edge to a non-existent node should not be wired."""
        g = EventRecord()
        e = _event("orphan", parent_event_id="nonexistent")
        g.add(e)
        node = g.get(e.event_id)
        assert node.neighbors("parent") == []


# ── Edge symmetry validation ─────────────────────────────────────────


SYMMETRIC_PAIRS = [
    ("parent", "child"),
    ("previous", "next"),
    ("triggered_by", "trigger"),
    ("started", "completed_by"),
]


class TestEdgeSymmetry:
    @pytest.mark.parametrize("forward,reverse", SYMMETRIC_PAIRS)
    def test_symmetry_on_tree(self, forward, reverse):
        g, _ = _tree_record()
        for node_id, node in g.nodes.items():
            for target_id in node.neighbors(forward):
                target_node = g.get(target_id)
                assert target_node is not None, f"{target_id} missing from record"
                assert node_id in target_node.neighbors(reverse), (
                    f"Asymmetric edge: {node_id} --{forward.value}--> {target_id} "
                    f"but {target_id} has no {reverse.value} back to {node_id}"
                )

    @pytest.mark.parametrize("forward,reverse", SYMMETRIC_PAIRS)
    def test_symmetry_on_linear(self, forward, reverse):
        g, _ = _linear_record(10)
        for node_id, node in g.nodes.items():
            for target_id in node.neighbors(forward):
                target_node = g.get(target_id)
                assert target_node is not None
                assert node_id in target_node.neighbors(reverse)


# ── Ordering tests ───────────────────────────────────────────────────


class TestOrdering:
    def test_emission_sequence_monotonic(self):
        g, events = _linear_record(10)
        sequences = [e.emission_sequence for e in events]
        assert sequences == sorted(sequences)
        assert len(set(sequences)) == len(sequences), "Duplicate sequences"

    def test_next_chain_follows_sequence_order(self):
        g, events = _linear_record(5)
        current = g.get(events[0].event_id)
        visited = []
        while current:
            visited.append(current.event.event_id)
            nexts = current.neighbors("next")
            current = g.get(nexts[0]) if nexts else None
        assert visited == [e.event_id for e in events]


# ── Traversal tests ─────────────────────────────────────────────────


class TestTraversal:
    def test_roots_single_root(self):
        g, events = _tree_record()
        roots = g.roots()
        assert len(roots) == 1
        assert roots[0].event.type == "crew_kickoff_started"

    def test_roots_multiple(self):
        g = EventRecord()
        g.add(_event("root1"))
        g.add(_event("root2"))
        assert len(g.roots()) == 2

    def test_descendants_of_crew_start(self):
        g, events = _tree_record()
        desc = g.descendants(events["crew_start"].event_id)
        desc_types = {n.event.type for n in desc}
        assert desc_types == {
            "task_started",
            "task_completed",
            "agent_execution_started",
            "agent_execution_completed",
        }

    def test_descendants_of_leaf(self):
        g, events = _tree_record()
        desc = g.descendants(events["task_complete"].event_id)
        assert desc == []

    def test_descendants_does_not_include_self(self):
        g, events = _tree_record()
        desc = g.descendants(events["crew_start"].event_id)
        desc_ids = {n.event.event_id for n in desc}
        assert events["crew_start"].event_id not in desc_ids


# ── Serialization round-trip tests ──────────────────────────────────


class TestSerialization:
    def test_empty_record_roundtrip(self):
        g = EventRecord()
        restored = EventRecord.model_validate_json(g.model_dump_json())
        assert len(restored) == 0

    def test_linear_record_roundtrip(self):
        g, events = _linear_record(5)
        restored = EventRecord.model_validate_json(g.model_dump_json())
        assert len(restored) == 5
        for e in events:
            assert e.event_id in restored

    def test_tree_record_roundtrip(self):
        g, events = _tree_record()
        restored = EventRecord.model_validate_json(g.model_dump_json())
        assert len(restored) == 5

        # Verify edges survived
        crew_node = restored.get(events["crew_start"].event_id)
        assert len(crew_node.neighbors("child")) == 2

    def test_roundtrip_preserves_edge_symmetry(self):
        g, _ = _tree_record()
        restored = EventRecord.model_validate_json(g.model_dump_json())
        for node_id, node in restored.nodes.items():
            for forward, reverse in SYMMETRIC_PAIRS:
                for target_id in node.neighbors(forward):
                    target_node = restored.get(target_id)
                    assert node_id in target_node.neighbors(reverse)

    def test_roundtrip_preserves_event_data(self):
        g = EventRecord()
        e = _event(
            "test",
            source_type="crew",
            task_id="t1",
            agent_role="researcher",
            emission_sequence=42,
        )
        g.add(e)
        restored = EventRecord.model_validate_json(g.model_dump_json())
        re = restored.get(e.event_id).event
        assert re.type == "test"
        assert re.source_type == "crew"
        assert re.task_id == "t1"
        assert re.agent_role == "researcher"
        assert re.emission_sequence == 42


# ── RuntimeState integration tests ──────────────────────────────────


class TestRuntimeStateIntegration:
    def test_runtime_state_serializes_event_record(self):
        from crewai import Agent, Crew, RuntimeState

        if RuntimeState is None:
            pytest.skip("RuntimeState unavailable (model_rebuild failed)")

        agent = Agent(
            role="test", goal="test", backstory="test", llm="gpt-4o-mini"
        )
        crew = Crew(agents=[agent], tasks=[], verbose=False)
        state = RuntimeState(root=[crew])

        e1 = _event("crew_started", emission_sequence=1)
        e2 = _event(
            "task_started",
            parent_event_id=e1.event_id,
            emission_sequence=2,
        )
        state.event_record.add(e1)
        state.event_record.add(e2)

        dumped = json.loads(state.model_dump_json())
        assert "entities" in dumped
        assert "event_record" in dumped
        assert len(dumped["event_record"]["nodes"]) == 2

    def test_runtime_state_roundtrip_with_record(self):
        from crewai import Agent, Crew, RuntimeState

        if RuntimeState is None:
            pytest.skip("RuntimeState unavailable (model_rebuild failed)")

        agent = Agent(
            role="test", goal="test", backstory="test", llm="gpt-4o-mini"
        )
        crew = Crew(agents=[agent], tasks=[], verbose=False)
        state = RuntimeState(root=[crew])

        e1 = _event("crew_started", emission_sequence=1)
        e2 = _event(
            "task_started",
            parent_event_id=e1.event_id,
            emission_sequence=2,
        )
        state.event_record.add(e1)
        state.event_record.add(e2)

        raw = state.model_dump_json()
        restored = RuntimeState.model_validate_json(
            raw, context={"from_checkpoint": True}
        )

        assert len(restored.event_record) == 2
        assert e1.event_id in restored.event_record
        assert e2.event_id in restored.event_record

        # Verify edges survived
        e2_node = restored.event_record.get(e2.event_id)
        assert e1.event_id in e2_node.neighbors("parent")

    def test_runtime_state_without_record_still_loads(self):
        """Backwards compat: a bare entity list should still validate."""
        from crewai import Agent, Crew, RuntimeState

        if RuntimeState is None:
            pytest.skip("RuntimeState unavailable (model_rebuild failed)")

        agent = Agent(
            role="test", goal="test", backstory="test", llm="gpt-4o-mini"
        )
        crew = Crew(agents=[agent], tasks=[], verbose=False)
        state = RuntimeState(root=[crew])

        # Simulate old-format JSON (just the entity list)
        old_json = json.dumps(
            [json.loads(crew.model_dump_json())]
        )
        restored = RuntimeState.model_validate_json(
            old_json, context={"from_checkpoint": True}
        )
        assert len(restored.root) == 1
        assert len(restored.event_record) == 0