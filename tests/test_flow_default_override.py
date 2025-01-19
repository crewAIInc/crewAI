"""Test that persisted state properly overrides default values."""

import os
from typing import Optional

import pytest
from pydantic import BaseModel

from crewai.flow.flow import Flow, FlowState, start
from crewai.flow.persistence import persist
from crewai.flow.persistence.sqlite import SQLiteFlowPersistence


class PoemState(FlowState):
    """Test state model with default values that should be overridden."""
    sentence_count: int = 1000  # Default that should be overridden
    poem: str = ""
    has_set_count: bool = False  # Track whether we've set the count


def test_default_value_override(tmp_path):
    """Test that persisted state values override class defaults."""
    db_path = os.path.join(tmp_path, "test_flows.db")
    persistence = SQLiteFlowPersistence(db_path)
    
    @persist(persistence)
    class PoemFlow(Flow[PoemState]):
        initial_state = PoemState
        
        @start()
        def set_sentence_count(self):
            # Only set sentence_count on first run, not when loading from persistence
            if not self.state.has_set_count:
                self.state.sentence_count = 2
                self.state.has_set_count = True
    
    # First run - should set sentence_count to 2
    # First run - should set sentence_count to 2
    flow1 = PoemFlow(persistence=persistence)
    flow1.kickoff()
    original_uuid = flow1.state.id
    assert flow1.state.sentence_count == 2
    
    # Second run - should load sentence_count=2 instead of default 1000
    flow2 = PoemFlow(persistence=persistence)
    flow2.kickoff(inputs={"id": original_uuid})
    assert flow2.state.sentence_count == 2  # Should load 2, not default 1000
    
    # Third run - explicit override should work
    flow3 = PoemFlow(
        persistence=persistence,
        id=original_uuid,
        sentence_count=3,  # Override persisted value
    )
    assert flow3.state.sentence_count == 3  # Should use override value
