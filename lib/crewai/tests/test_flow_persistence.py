"""Test flow state persistence functionality."""

import os
from typing import Dict, List

import pytest
from crewai.flow.flow import Flow, FlowState, listen, start
from crewai.flow.persistence import persist
from crewai.flow.persistence.sqlite import SQLiteFlowPersistence
from pydantic import BaseModel


class TestState(FlowState):
    """Test state model with required id field."""

    counter: int = 0
    message: str = ""


def test_persist_decorator_saves_state(tmp_path, caplog):
    """Test that @persist decorator saves state in SQLite."""
    db_path = os.path.join(tmp_path, "test_flows.db")
    persistence = SQLiteFlowPersistence(db_path)

    class TestFlow(Flow[Dict[str, str]]):
        initial_state = dict()  # Use dict instance as initial state

        @start()
        @persist(persistence)
        def init_step(self):
            self.state["message"] = "Hello, World!"
            self.state["id"] = "test-uuid"  # Ensure we have an ID for persistence

    # Run flow and verify state is saved
    flow = TestFlow(persistence=persistence)
    flow.kickoff()

    # Load state from DB and verify
    saved_state = persistence.load_state(flow.state["id"])
    assert saved_state is not None
    assert saved_state["message"] == "Hello, World!"


def test_structured_state_persistence(tmp_path):
    """Test persistence with Pydantic model state."""
    db_path = os.path.join(tmp_path, "test_flows.db")
    persistence = SQLiteFlowPersistence(db_path)

    class StructuredFlow(Flow[TestState]):
        initial_state = TestState

        @start()
        @persist(persistence)
        def count_up(self):
            self.state.counter += 1
            self.state.message = f"Count is {self.state.counter}"

    # Run flow and verify state changes are saved
    flow = StructuredFlow(persistence=persistence)
    flow.kickoff()

    # Load and verify state
    saved_state = persistence.load_state(flow.state.id)
    assert saved_state is not None
    assert saved_state["counter"] == 1
    assert saved_state["message"] == "Count is 1"


def test_flow_state_restoration(tmp_path):
    """Test restoring flow state from persistence with various restoration methods."""
    db_path = os.path.join(tmp_path, "test_flows.db")
    persistence = SQLiteFlowPersistence(db_path)

    # First flow execution to create initial state
    class RestorableFlow(Flow[TestState]):
        @start()
        @persist(persistence)
        def set_message(self):
            if self.state.message == "":
                self.state.message = "Original message"
            if self.state.counter == 0:
                self.state.counter = 42

    # Create and persist initial state
    flow1 = RestorableFlow(persistence=persistence)
    flow1.kickoff()
    original_uuid = flow1.state.id

    # Test case 1: Restore using restore_uuid with field override
    flow2 = RestorableFlow(persistence=persistence)
    flow2.kickoff(inputs={"id": original_uuid, "counter": 43})

    # Verify state restoration and selective field override
    assert flow2.state.id == original_uuid
    assert flow2.state.message == "Original message"  # Preserved
    assert flow2.state.counter == 43  # Overridden

    # Test case 2: Restore using kwargs['id']
    flow3 = RestorableFlow(persistence=persistence)
    flow3.kickoff(inputs={"id": original_uuid, "message": "Updated message"})

    # Verify state restoration and selective field override
    assert flow3.state.id == original_uuid
    assert flow3.state.counter == 43  # Preserved
    assert flow3.state.message == "Updated message"  # Overridden


def test_multiple_method_persistence(tmp_path):
    """Test state persistence across multiple method executions."""
    db_path = os.path.join(tmp_path, "test_flows.db")
    persistence = SQLiteFlowPersistence(db_path)

    class MultiStepFlow(Flow[TestState]):
        @start()
        @persist(persistence)
        def step_1(self):
            if self.state.counter == 1:
                self.state.counter = 99999
                self.state.message = "Step 99999"
            else:
                self.state.counter = 1
                self.state.message = "Step 1"

        @listen(step_1)
        @persist(persistence)
        def step_2(self):
            if self.state.counter == 1:
                self.state.counter = 2
                self.state.message = "Step 2"

    flow = MultiStepFlow(persistence=persistence)
    flow.kickoff()

    flow2 = MultiStepFlow(persistence=persistence)
    flow2.kickoff(inputs={"id": flow.state.id})

    # Load final state
    final_state = flow2.state
    assert final_state is not None
    assert final_state.counter == 2
    assert final_state.message == "Step 2"

    class NoPersistenceMultiStepFlow(Flow[TestState]):
        @start()
        @persist(persistence)
        def step_1(self):
            if self.state.counter == 1:
                self.state.counter = 99999
                self.state.message = "Step 99999"
            else:
                self.state.counter = 1
                self.state.message = "Step 1"

        @listen(step_1)
        def step_2(self):
            if self.state.counter == 1:
                self.state.counter = 2
                self.state.message = "Step 2"

    flow = NoPersistenceMultiStepFlow(persistence=persistence)
    flow.kickoff()

    flow2 = NoPersistenceMultiStepFlow(persistence=persistence)
    flow2.kickoff(inputs={"id": flow.state.id})

    # Load final state
    final_state = flow2.state
    assert final_state.counter == 99999
    assert final_state.message == "Step 99999"


def test_persist_decorator_verbose_logging(tmp_path, caplog):
    """Test that @persist decorator's verbose parameter controls logging."""
    # Set logging level to ensure we capture all logs
    caplog.set_level("INFO")

    db_path = os.path.join(tmp_path, "test_flows.db")
    persistence = SQLiteFlowPersistence(db_path)

    # Test with verbose=False (default)
    class QuietFlow(Flow[Dict[str, str]]):
        initial_state = dict()

        @start()
        @persist(persistence)  # Default verbose=False
        def init_step(self):
            self.state["message"] = "Hello, World!"
            self.state["id"] = "test-uuid-1"

    flow = QuietFlow(persistence=persistence)
    flow.kickoff()
    assert "Saving flow state" not in caplog.text

    # Clear the log
    caplog.clear()

    # Test with verbose=True
    class VerboseFlow(Flow[Dict[str, str]]):
        initial_state = dict()

        @start()
        @persist(persistence, verbose=True)
        def init_step(self):
            self.state["message"] = "Hello, World!"
            self.state["id"] = "test-uuid-2"

    flow = VerboseFlow(persistence=persistence)
    flow.kickoff()
    assert "Saving flow state" in caplog.text


def test_persistence_with_base_model(tmp_path):
    db_path = os.path.join(tmp_path, "test_flows.db")
    persistence = SQLiteFlowPersistence(db_path)

    class Message(BaseModel):
        role: str
        type: str
        content: str

    class State(FlowState):
        latest_message: Message | None = None
        history: List[Message] = []

    @persist(persistence)
    class BaseModelFlow(Flow[State]):
        initial_state = State(latest_message=None, history=[])

        @start()
        def init_step(self):
            self.state.latest_message = Message(
                role="user", type="text", content="Hello, World!"
            )
            self.state.history.append(self.state.latest_message)

    flow = BaseModelFlow(persistence=persistence)
    flow.kickoff()

    latest_message = flow.state.latest_message
    (message,) = flow.state.history

    assert latest_message is not None
    assert latest_message.role == "user"
    assert latest_message.type == "text"
    assert latest_message.content == "Hello, World!"

    assert len(flow.state.history) == 1
    assert message.role == "user"
    assert message.type == "text"
    assert message.content == "Hello, World!"
    assert isinstance(flow.state._unwrap(), State)


def test_fork_with_restore_from_state_id(tmp_path):
    """Fork: restore_from_state_id hydrates state from source flow_uuid; new run gets a
    fresh state.id; source's history is preserved (the fork's @persist writes go under
    the new state.id, not the source's)."""
    db_path = os.path.join(tmp_path, "test_flows.db")
    persistence = SQLiteFlowPersistence(db_path)

    class ForkableFlow(Flow[TestState]):
        @start()
        @persist(persistence)
        def step(self):
            self.state.counter += 1

    # Run 1: build up source state. counter goes 0 -> 1.
    flow1 = ForkableFlow(persistence=persistence)
    flow1.kickoff()
    source_uuid = flow1.state.id
    assert flow1.state.counter == 1

    # Resume on the same uuid bumps counter to 2 in the SAME flow_uuid history.
    flow1b = ForkableFlow(persistence=persistence)
    flow1b.kickoff(inputs={"id": source_uuid})
    assert flow1b.state.counter == 2
    assert persistence.load_state(source_uuid)["counter"] == 2

    # Fork: hydrate from source, but persist under a fresh state.id.
    flow2 = ForkableFlow(persistence=persistence)
    flow2.kickoff(restore_from_state_id=source_uuid)

    # Fork has a different state.id from the source.
    assert flow2.state.id != source_uuid
    # Hydrated from source's latest snapshot (counter=2), then incremented to 3.
    assert flow2.state.counter == 3

    # Source's history is unchanged after the fork.
    assert persistence.load_state(source_uuid)["counter"] == 2

    # Fork's writes landed under its own state.id.
    assert persistence.load_state(flow2.state.id)["counter"] == 3


def test_fork_with_pinned_state_id(tmp_path):
    """Fork into a pinned state.id (inputs.id supplied alongside restore_from_state_id):
    the new run uses inputs.id as state.id and hydrates from restore_from_state_id."""
    db_path = os.path.join(tmp_path, "test_flows.db")
    persistence = SQLiteFlowPersistence(db_path)

    class PinnableFlow(Flow[TestState]):
        @start()
        @persist(persistence)
        def step(self):
            self.state.counter += 1

    flow1 = PinnableFlow(persistence=persistence)
    flow1.kickoff()
    source_uuid = flow1.state.id
    assert flow1.state.counter == 1

    pinned_uuid = "pinned-fork-uuid-1234"
    flow2 = PinnableFlow(persistence=persistence)
    flow2.kickoff(
        inputs={"id": pinned_uuid},
        restore_from_state_id=source_uuid,
    )

    # state.id pinned to inputs.id, NOT the source uuid.
    assert flow2.state.id == pinned_uuid
    # Hydrated from source: counter started at 1, step incremented to 2.
    assert flow2.state.counter == 2
    # Source's history is unchanged.
    assert persistence.load_state(source_uuid)["counter"] == 1
    # Fork's writes are under the pinned uuid.
    assert persistence.load_state(pinned_uuid)["counter"] == 2


def test_restore_from_state_id_not_found_silent_fallback(tmp_path):
    """Lookup miss on restore_from_state_id silently falls through to default behavior."""
    db_path = os.path.join(tmp_path, "test_flows.db")
    persistence = SQLiteFlowPersistence(db_path)

    class FallbackFlow(Flow[TestState]):
        @start()
        @persist(persistence)
        def step(self):
            self.state.counter += 1

    flow = FallbackFlow(persistence=persistence)
    # No source UUID exists — should not raise.
    flow.kickoff(restore_from_state_id="no-such-uuid")

    # Default state path: counter starts at 0 and step increments to 1.
    assert flow.state.counter == 1
    # state.id is the auto-generated one, NOT the missing source.
    assert flow.state.id != "no-such-uuid"


def test_restore_from_state_id_none_is_no_op(tmp_path):
    """restore_from_state_id=None (default) preserves baseline kickoff behavior."""
    db_path = os.path.join(tmp_path, "test_flows.db")
    persistence = SQLiteFlowPersistence(db_path)

    class BaselineFlow(Flow[TestState]):
        @start()
        @persist(persistence)
        def step(self):
            self.state.counter += 1

    flow = BaselineFlow(persistence=persistence)
    flow.kickoff(restore_from_state_id=None)
    assert flow.state.counter == 1


def test_fork_conflict_with_from_checkpoint_raises():
    """Passing both from_checkpoint and restore_from_state_id raises ValueError, naming
    both parameters."""
    from crewai.state import CheckpointConfig

    class ConflictFlow(Flow[TestState]):
        @start()
        def step(self):
            pass

    flow = ConflictFlow()
    with pytest.raises(ValueError) as excinfo:
        flow.kickoff(
            from_checkpoint=CheckpointConfig(),
            restore_from_state_id="some-uuid",
        )
    msg = str(excinfo.value)
    assert "from_checkpoint" in msg
    assert "restore_from_state_id" in msg


@pytest.mark.asyncio
async def test_fork_via_kickoff_async(tmp_path):
    """kickoff_async honors restore_from_state_id: hydrates from source, mints fresh
    state.id, persists under the new id, source history preserved."""
    db_path = os.path.join(tmp_path, "test_flows.db")
    persistence = SQLiteFlowPersistence(db_path)

    class AsyncForkableFlow(Flow[TestState]):
        @start()
        @persist(persistence)
        def step(self):
            self.state.counter += 1

    flow1 = AsyncForkableFlow(persistence=persistence)
    await flow1.kickoff_async()
    source_uuid = flow1.state.id
    assert flow1.state.counter == 1

    flow2 = AsyncForkableFlow(persistence=persistence)
    await flow2.kickoff_async(restore_from_state_id=source_uuid)

    assert flow2.state.id != source_uuid
    assert flow2.state.counter == 2
    assert persistence.load_state(source_uuid)["counter"] == 1
    assert persistence.load_state(flow2.state.id)["counter"] == 2


@pytest.mark.asyncio
async def test_fork_via_akickoff(tmp_path):
    """akickoff is the public async alias and must accept restore_from_state_id with
    the same semantics as kickoff_async."""
    db_path = os.path.join(tmp_path, "test_flows.db")
    persistence = SQLiteFlowPersistence(db_path)

    class AkickoffForkableFlow(Flow[TestState]):
        @start()
        @persist(persistence)
        def step(self):
            self.state.counter += 1

    flow1 = AkickoffForkableFlow(persistence=persistence)
    await flow1.akickoff()
    source_uuid = flow1.state.id
    assert flow1.state.counter == 1

    flow2 = AkickoffForkableFlow(persistence=persistence)
    await flow2.akickoff(restore_from_state_id=source_uuid)

    assert flow2.state.id != source_uuid
    assert flow2.state.counter == 2
    assert persistence.load_state(source_uuid)["counter"] == 1
    assert persistence.load_state(flow2.state.id)["counter"] == 2


@pytest.mark.asyncio
async def test_akickoff_pinned_fork(tmp_path):
    """akickoff with both inputs.id and restore_from_state_id pins state.id while
    hydrating from the source."""
    db_path = os.path.join(tmp_path, "test_flows.db")
    persistence = SQLiteFlowPersistence(db_path)

    class PinnableAsyncFlow(Flow[TestState]):
        @start()
        @persist(persistence)
        def step(self):
            self.state.counter += 1

    flow1 = PinnableAsyncFlow(persistence=persistence)
    await flow1.akickoff()
    source_uuid = flow1.state.id

    pinned_uuid = "pinned-akickoff-fork-uuid"
    flow2 = PinnableAsyncFlow(persistence=persistence)
    await flow2.akickoff(
        inputs={"id": pinned_uuid},
        restore_from_state_id=source_uuid,
    )

    assert flow2.state.id == pinned_uuid
    assert flow2.state.counter == 2
    assert persistence.load_state(source_uuid)["counter"] == 1
    assert persistence.load_state(pinned_uuid)["counter"] == 2


@pytest.mark.asyncio
async def test_akickoff_fork_conflict_with_from_checkpoint_raises():
    """akickoff must raise the same conflict ValueError as kickoff/kickoff_async when
    both from_checkpoint and restore_from_state_id are set."""
    from crewai.state import CheckpointConfig

    class AsyncConflictFlow(Flow[TestState]):
        @start()
        def step(self):
            pass

    flow = AsyncConflictFlow()
    with pytest.raises(ValueError) as excinfo:
        await flow.akickoff(
            from_checkpoint=CheckpointConfig(),
            restore_from_state_id="some-uuid",
        )
    msg = str(excinfo.value)
    assert "from_checkpoint" in msg
    assert "restore_from_state_id" in msg
