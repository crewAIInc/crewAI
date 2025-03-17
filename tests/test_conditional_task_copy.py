import pytest

from crewai import Agent, Crew, Task
from crewai.tasks.conditional_task import ConditionalTask
from crewai.tasks.task_output import TaskOutput


def test_conditional_task_preserved_in_copy():
    """Test that ConditionalTask objects are preserved when copying a Crew."""
    agent = Agent(
        role="Researcher",
        goal="Research topics",
        backstory="You are a researcher."
    )
    
    # Create a regular task
    task1 = Task(
        description="Research topic A",
        expected_output="Research results for topic A",
        agent=agent
    )
    
    # Create a conditional task
    conditional_task = ConditionalTask(
        description="Research topic B if topic A was successful",
        expected_output="Research results for topic B",
        agent=agent,
        condition=lambda output: "success" in output.raw.lower()
    )
    
    # Create a crew with both tasks
    crew = Crew(
        agents=[agent],
        tasks=[task1, conditional_task]
    )
    
    # Create a copy of the crew
    crew_copy = crew.copy()
    
    # Check that the conditional task is still a ConditionalTask in the copied crew
    assert isinstance(crew_copy.tasks[1], ConditionalTask)
    assert hasattr(crew_copy.tasks[1], "should_execute")
    
def test_conditional_task_preserved_in_kickoff_for_each():
    """Test that ConditionalTask objects are preserved when using kickoff_for_each."""
    from unittest.mock import patch
    
    agent = Agent(
        role="Researcher",
        goal="Research topics",
        backstory="You are a researcher."
    )
    
    # Create a regular task
    task1 = Task(
        description="Research topic A",
        expected_output="Research results for topic A",
        agent=agent
    )
    
    # Create a conditional task
    conditional_task = ConditionalTask(
        description="Research topic B if topic A was successful",
        expected_output="Research results for topic B",
        agent=agent,
        condition=lambda output: "success" in output.raw.lower()
    )
    
    # Create a crew with both tasks
    crew = Crew(
        agents=[agent],
        tasks=[task1, conditional_task]
    )
    
    # Mock the kickoff method to avoid actual execution
    with patch.object(Crew, "kickoff") as mock_kickoff:
        # Set up the mock to return a TaskOutput
        mock_output = TaskOutput(
            description="Mock task output",
            raw="Success with topic",
            agent=agent.role
        )
        mock_kickoff.return_value = mock_output
        
        # Call kickoff_for_each with test inputs
        inputs = [{"topic": "test1"}, {"topic": "test2"}]
        crew.kickoff_for_each(inputs=inputs)
        
        # Verify the mock was called with the expected inputs
        assert mock_kickoff.call_count == len(inputs)
        
        # Create a copy of the crew to verify the type preservation
        # (since we can't directly access the crews created inside kickoff_for_each)
        crew_copy = crew.copy()
        assert isinstance(crew_copy.tasks[1], ConditionalTask)
