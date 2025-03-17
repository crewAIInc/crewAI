"""Test flow state persistence functionality."""

import os
from typing import Dict

import pytest
from pydantic import BaseModel

from crewai.flow.flow import Flow, FlowState, listen, start
from crewai.flow.persistence import persist
from crewai.flow.persistence.sqlite import SQLiteFlowPersistence


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
