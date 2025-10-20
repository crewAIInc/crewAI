"""Tests for Flow initialization with Pydantic models having required fields.
Covers https://github.com/crewAIInc/crewAI/issues/3744
"""

import pytest
from pydantic import BaseModel, ValidationError

from crewai.flow.flow import Flow, FlowState, listen, start


class RequiredState(BaseModel):
    """State model with required fields."""
    name: str
    age: int


class RequiredStateFlow(Flow[RequiredState]):
    """Flow with required state fields."""
    
    @start()
    def begin(self):
        return "started"


class MixedState(BaseModel):
    """State model with both required and optional fields."""
    name: str  # Required
    age: int  # Required
    email: str = "default@example.com"  # Optional with default


class MixedStateFlow(Flow[MixedState]):
    """Flow with mixed required and optional state fields."""
    
    @start()
    def begin(self):
        return f"Started with {self.state.name}, {self.state.age}, {self.state.email}"


class RequiredStateWithFlowState(FlowState):
    """State model extending FlowState with required fields."""
    name: str
    age: int


class RequiredFlowStateFlow(Flow[RequiredStateWithFlowState]):
    """Flow with required FlowState fields."""
    
    @start()
    def begin(self):
        return "started"


def test_flow_initialization_without_kwargs_raises_validation_error():
    """Test that Flow initialization without kwargs raises ValidationError for required fields."""
    with pytest.raises(ValidationError) as exc_info:
        RequiredStateFlow()
    
    error_str = str(exc_info.value)
    assert "name" in error_str
    assert "age" in error_str


def test_flow_initialization_with_kwargs_passes_and_sets_state():
    """Test that Flow initialization with kwargs properly sets state values."""
    flow = RequiredStateFlow(name="John", age=30)
    assert flow.state.name == "John"
    assert flow.state.age == 30
    assert hasattr(flow.state, "id")
    assert flow.state.id is not None


def test_flow_initialization_with_partial_kwargs_raises_validation_error():
    """Test that Flow initialization with only some required kwargs raises ValidationError."""
    with pytest.raises(ValidationError) as exc_info:
        RequiredStateFlow(name="John")
    
    error_str = str(exc_info.value)
    assert "age" in error_str


def test_flow_initialization_with_mixed_required_and_optional_fields():
    """Test Flow initialization with both required and optional fields."""
    flow1 = MixedStateFlow(name="Alice", age=25)
    assert flow1.state.name == "Alice"
    assert flow1.state.age == 25
    assert flow1.state.email == "default@example.com"
    
    flow2 = MixedStateFlow(name="Bob", age=35, email="bob@example.com")
    assert flow2.state.name == "Bob"
    assert flow2.state.age == 35
    assert flow2.state.email == "bob@example.com"


def test_flow_initialization_with_flowstate_and_required_fields():
    """Test Flow initialization with FlowState subclass having required fields."""
    flow = RequiredFlowStateFlow(name="Charlie", age=40)
    assert flow.state.name == "Charlie"
    assert flow.state.age == 40
    assert hasattr(flow.state, "id")
    assert flow.state.id is not None


def test_flow_execution_with_required_state():
    """Test that Flow execution works correctly with required state fields."""
    flow = RequiredStateFlow(name="David", age=45)
    result = flow.kickoff()
    assert result == "started"
    assert flow.state.name == "David"
    assert flow.state.age == 45


def test_flow_with_state_modification():
    """Test that state can be modified during flow execution."""
    
    class ModifiableState(BaseModel):
        counter: int
        name: str
    
    class ModifiableFlow(Flow[ModifiableState]):
        @start()
        def increment(self):
            self.state.counter += 10
            return "incremented"
        
        @listen(increment)
        def check_value(self):
            assert self.state.counter == 15
            return "checked"
    
    flow = ModifiableFlow(counter=5, name="Test")
    result = flow.kickoff()
    assert result == "checked"
    assert flow.state.counter == 15


def test_flow_initialization_preserves_id_field():
    """Test that the automatically generated id field is preserved."""
    flow = RequiredStateFlow(name="Eve", age=28)
    original_id = flow.state.id
    
    assert isinstance(original_id, str)
    assert len(original_id) == 36  # UUID format with hyphens
    
    assert flow.state.id == original_id
