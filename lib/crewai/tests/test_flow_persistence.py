"""Test flow state persistence functionality."""

import logging
import os
from typing import Dict, List

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


def test_sqlite_persistence_logs_db_path_on_init(tmp_path, caplog):
    """Test that SQLiteFlowPersistence logs its db_path on initialization."""
    caplog.set_level("INFO")

    db_path = os.path.join(tmp_path, "my_custom.db")
    SQLiteFlowPersistence(db_path)

    assert "SQLiteFlowPersistence initialized with db_path" in caplog.text
    assert db_path in caplog.text


def test_sqlite_persistence_default_path_is_logged(caplog):
    """Test that the default persistence path is logged so users can discover it."""
    caplog.set_level("INFO")

    persistence = SQLiteFlowPersistence()

    assert "SQLiteFlowPersistence initialized with db_path" in caplog.text
    assert "flow_states.db" in caplog.text
    # Verify the db_path attribute is accessible for programmatic discovery
    assert persistence.db_path.endswith("flow_states.db")


def test_persist_logs_storage_location_on_save(tmp_path, caplog):
    """Test that the persist decorator logs the storage location when state is saved."""
    caplog.set_level("INFO")

    db_path = os.path.join(tmp_path, "test_flows.db")
    persistence = SQLiteFlowPersistence(db_path)

    class LocationLogFlow(Flow[TestState]):
        @start()
        @persist(persistence)
        def init_step(self):
            self.state.message = "test"

    flow = LocationLogFlow(persistence=persistence)
    flow.kickoff()

    # Verify that the storage location (db_path) is logged after saving
    assert "storage:" in caplog.text
    assert db_path in caplog.text


def test_persist_verbose_shows_storage_location_with_db_path(tmp_path, caplog):
    """Test that verbose persist includes storage location with actual db_path."""
    caplog.set_level("INFO")

    db_path = os.path.join(tmp_path, "verbose_test.db")
    persistence = SQLiteFlowPersistence(db_path)

    class VerboseLocationFlow(Flow[Dict[str, str]]):
        initial_state = dict()

        @start()
        @persist(persistence, verbose=True)
        def init_step(self):
            self.state["message"] = "Hello!"
            self.state["id"] = "verbose-uuid"

    flow = VerboseLocationFlow(persistence=persistence)
    flow.kickoff()

    # Verbose mode should log both save message and storage location
    assert "Saving flow state for ID: verbose-uuid" in caplog.text
    assert f"storage: {db_path}" in caplog.text


def test_persist_class_level_logs_storage_location(tmp_path, caplog):
    """Test that class-level @persist also logs the storage location."""
    caplog.set_level("INFO")

    db_path = os.path.join(tmp_path, "class_level_test.db")
    persistence = SQLiteFlowPersistence(db_path)

    @persist(persistence)
    class ClassLevelFlow(Flow[TestState]):
        @start()
        def init_step(self):
            self.state.message = "class level"

    flow = ClassLevelFlow(persistence=persistence)
    flow.kickoff()

    # Verify storage location is logged even with class-level decorator
    assert "storage:" in caplog.text
    assert db_path in caplog.text
