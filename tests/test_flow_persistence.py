"""Test flow state persistence functionality."""

import os
from typing import Dict, Optional

import pytest
from pydantic import BaseModel

from crewai.flow.flow import Flow, FlowState, start
from crewai.flow.persistence import FlowPersistence, persist
from crewai.flow.persistence.sqlite import SQLiteFlowPersistence


class TestState(FlowState):
    """Test state model with required id field."""
    counter: int = 0
    message: str = ""


def test_persist_decorator_saves_state(tmp_path):
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
    """Test restoring flow state from persistence."""
    db_path = os.path.join(tmp_path, "test_flows.db")
    persistence = SQLiteFlowPersistence(db_path)
    
    # First flow execution to create initial state
    class RestorableFlow(Flow[TestState]):
        initial_state = TestState
        
        @start()
        @persist(persistence)
        def set_message(self):
            self.state.message = "Original message"
            self.state.counter = 42
    
    flow1 = RestorableFlow(persistence=persistence)
    flow1.kickoff()
    original_uuid = flow1.state.id
    
    # Create new flow instance with restored state
    flow2 = RestorableFlow(
        persistence=persistence,
        restore_uuid=original_uuid,
        counter=43,  # Override counter
    )
    
    # Verify state restoration and merging
    assert flow2.state.id == original_uuid
    assert flow2.state.message == "Original message"
    assert flow2.state.counter == 43  # Verify override worked


def test_multiple_method_persistence(tmp_path):
    """Test state persistence across multiple method executions."""
    db_path = os.path.join(tmp_path, "test_flows.db")
    persistence = SQLiteFlowPersistence(db_path)
    
    class MultiStepFlow(Flow[TestState]):
        initial_state = TestState
        
        @start()
        @persist(persistence)
        def step_1(self):
            self.state.counter = 1
            self.state.message = "Step 1"
        
        @start()
        @persist(persistence)
        def step_2(self):
            self.state.counter = 2
            self.state.message = "Step 2"
    
    flow = MultiStepFlow(persistence=persistence)
    flow.kickoff()
    
    # Load final state
    final_state = persistence.load_state(flow.state.id)
    assert final_state is not None
    assert final_state["counter"] == 2
    assert final_state["message"] == "Step 2"


def test_persistence_error_handling(tmp_path):
    """Test error handling in persistence operations."""
    db_path = os.path.join(tmp_path, "test_flows.db")
    persistence = SQLiteFlowPersistence(db_path)
    
    class InvalidFlow(Flow[TestState]):
        # Missing id field in initial state
        class InvalidState(BaseModel):
            value: str = ""
            
        initial_state = InvalidState
        
        @start()
        @persist(persistence)
        def will_fail(self):
            self.state.value = "test"
    
    with pytest.raises(ValueError) as exc_info:
        flow = InvalidFlow(persistence=persistence)
    
    assert "must have an 'id' field" in str(exc_info.value)
