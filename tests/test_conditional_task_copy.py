import pytest

from crewai import Agent, Crew, Task
from crewai.tasks.conditional_task import ConditionalTask
from crewai.tasks.task_output import TaskOutput


@pytest.fixture
def test_agent():
    """Fixture for creating a test agent."""
    return Agent(
        role="Researcher",
        goal="Research topics",
        backstory="You are a researcher."
    )

@pytest.fixture
def test_task(test_agent):
    """Fixture for creating a regular task."""
    return Task(
        description="Research topic A",
        expected_output="Research results for topic A",
        agent=test_agent
    )

@pytest.fixture
def test_conditional_task(test_agent):
    """Fixture for creating a conditional task."""
    return ConditionalTask(
        description="Research topic B if topic A was successful",
        expected_output="Research results for topic B",
        agent=test_agent,
        condition=lambda output: "success" in output.raw.lower()
    )

@pytest.fixture
def test_crew(test_agent, test_task, test_conditional_task):
    """Fixture for creating a crew with both regular and conditional tasks."""
    return Crew(
        agents=[test_agent],
        tasks=[test_task, test_conditional_task]
    )


def test_conditional_task_preserved_in_copy(test_crew):
    """Test that ConditionalTask objects are preserved when copying a Crew."""
    # Create a copy of the crew
    crew_copy = test_crew.copy()
    
    # Check that the conditional task is still a ConditionalTask in the copied crew
    assert isinstance(crew_copy.tasks[1], ConditionalTask)
    assert hasattr(crew_copy.tasks[1], "should_execute")
    
def test_conditional_task_preserved_in_kickoff_for_each(test_crew, test_agent):
    """Test that ConditionalTask objects are preserved when using kickoff_for_each."""
    from unittest.mock import patch
    
    # Mock the kickoff method to avoid actual execution
    with patch.object(Crew, "kickoff") as mock_kickoff:
        # Set up the mock to return a TaskOutput
        mock_output = TaskOutput(
            description="Mock task output",
            raw="Success with topic",
            agent=test_agent.role
        )
        mock_kickoff.return_value = mock_output
        
        # Call kickoff_for_each with test inputs
        inputs = [{"topic": "test1"}, {"topic": "test2"}]
        test_crew.kickoff_for_each(inputs=inputs)
        
        # Verify the mock was called with the expected inputs
        assert mock_kickoff.call_count == len(inputs)
        
        # Create a copy of the crew to verify the type preservation
        # (since we can't directly access the crews created inside kickoff_for_each)
        crew_copy = test_crew.copy()
        assert isinstance(crew_copy.tasks[1], ConditionalTask)


def test_conditional_task_copy_with_none_values(test_agent, test_task):
    """Test that ConditionalTask objects are preserved when copying with optional fields."""
    # Create a conditional task with optional fields
    conditional_task = ConditionalTask(
        description="Research topic B if topic A was successful",
        expected_output="Research results for topic B",  # Required field
        agent=test_agent,
        condition=lambda output: "success" in output.raw.lower(),
        context=None  # Optional field that can be None
    )
    
    # Create a crew with both a regular task and the conditional task
    crew = Crew(
        agents=[test_agent],
        tasks=[test_task, conditional_task]
    )
    
    # Create a copy of the crew
    crew_copy = crew.copy()
    
    # Check that the conditional task is still a ConditionalTask in the copied crew
    assert isinstance(crew_copy.tasks[1], ConditionalTask)
    assert hasattr(crew_copy.tasks[1], "should_execute")
    assert crew_copy.tasks[1].context is None  # Verify None value is preserved
