"""Tests for Pydantic version compatibility issues."""

from unittest.mock import patch, MagicMock
from pydantic import BaseModel

from crewai.flow.flow_trackable import FlowTrackable
from crewai.flow import Flow


class TestFlowTrackable(FlowTrackable, BaseModel):
    """Test class that inherits from FlowTrackable for testing."""
    name: str = "test"


class MockFlow(Flow):
    """Mock Flow class for testing."""
    
    def __init__(self):
        super().__init__()


def test_flow_trackable_instantiation():
    """Test that FlowTrackable can be instantiated without ValidationInfo errors."""
    trackable = TestFlowTrackable()
    assert trackable.name == "test"
    assert trackable.parent_flow is None


def test_flow_trackable_with_parent_flow():
    """Test that FlowTrackable correctly identifies parent flow from call stack."""
    mock_flow = MockFlow()
    
    def create_trackable_in_flow():
        return TestFlowTrackable()
    
    with patch('inspect.currentframe') as mock_frame:
        mock_current_frame = MagicMock()
        mock_parent_frame = MagicMock()
        mock_flow_frame = MagicMock()
        
        mock_current_frame.f_back = mock_parent_frame
        mock_parent_frame.f_back = mock_flow_frame
        mock_flow_frame.f_back = None
        
        mock_parent_frame.f_locals = {}
        mock_flow_frame.f_locals = {"self": mock_flow}
        
        mock_frame.return_value = mock_current_frame
        
        trackable = create_trackable_in_flow()
        assert trackable.parent_flow == mock_flow


def test_flow_trackable_no_parent_flow():
    """Test that FlowTrackable handles case where no parent flow is found."""
    with patch('inspect.currentframe') as mock_frame:
        mock_current_frame = MagicMock()
        mock_parent_frame = MagicMock()
        
        mock_current_frame.f_back = mock_parent_frame
        mock_parent_frame.f_back = None
        mock_parent_frame.f_locals = {"self": "not_a_flow"}
        
        mock_frame.return_value = mock_current_frame
        
        trackable = TestFlowTrackable()
        assert trackable.parent_flow is None


def test_flow_trackable_max_depth_limit():
    """Test that FlowTrackable respects max_depth limit when searching for parent flow."""
    with patch('inspect.currentframe') as mock_frame:
        mock_frames = []
        for i in range(10):
            frame = MagicMock()
            frame.f_locals = {"self": f"frame_{i}"}
            mock_frames.append(frame)
        
        for i in range(len(mock_frames) - 1):
            mock_frames[i].f_back = mock_frames[i + 1]
        mock_frames[-1].f_back = None
        
        mock_frame.return_value = mock_frames[0]
        
        trackable = TestFlowTrackable()
        assert trackable.parent_flow is None


def test_flow_trackable_none_frame():
    """Test that FlowTrackable handles None frame gracefully."""
    with patch('inspect.currentframe', return_value=None):
        trackable = TestFlowTrackable()
        assert trackable.parent_flow is None


def test_pydantic_model_validator_signature():
    """Test that the model validator has the correct signature for Pydantic compatibility."""
    import inspect
    from crewai.flow.flow_trackable import FlowTrackable
    
    validator_method = FlowTrackable._set_parent_flow
    sig = inspect.signature(validator_method)
    
    params = list(sig.parameters.keys())
    assert params == ['self'], f"Expected ['self'], got {params}"
    
    assert sig.return_annotation == "FlowTrackable"


def test_crew_instantiation_with_flow_trackable():
    """Test that Crew can be instantiated without ValidationInfo errors (reproduces issue #3011)."""
    from crewai import Crew, Agent, Task
    
    agent = Agent(
        role="Test Agent",
        goal="Test goal",
        backstory="Test backstory"
    )
    
    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task]
    )
    
    assert crew is not None
    assert len(crew.agents) == 1
    assert len(crew.tasks) == 1
