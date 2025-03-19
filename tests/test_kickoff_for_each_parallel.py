"""Test for the kickoff_for_each_parallel method in Crew class."""

import concurrent.futures
import pytest
from unittest.mock import patch, MagicMock

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.crews.crew_output import CrewOutput
from crewai.task import Task


def test_kickoff_for_each_parallel_single_input():
    """Tests if kickoff_for_each_parallel works with a single input."""
    
    inputs = [{"topic": "dog"}]
    
    agent = Agent(
        role="{topic} Researcher",
        goal="Express hot takes on {topic}.",
        backstory="You have a lot of experience with {topic}.",
    )
    
    task = Task(
        description="Give me an analysis around {topic}.",
        expected_output="1 bullet point about {topic} that's under 15 words.",
        agent=agent,
    )
    
    crew = Crew(agents=[agent], tasks=[task])
    
    # Mock the kickoff method to avoid API calls
    expected_output = CrewOutput(raw="Dogs are loyal companions.")
    with patch.object(Crew, "kickoff", return_value=expected_output):
        results = crew.kickoff_for_each_parallel(inputs=inputs)
        
        assert len(results) == 1
        assert results[0].raw == "Dogs are loyal companions."


def test_kickoff_for_each_parallel_multiple_inputs():
    """Tests if kickoff_for_each_parallel works with multiple inputs."""
    
    inputs = [
        {"topic": "dog"},
        {"topic": "cat"},
        {"topic": "apple"},
    ]
    
    agent = Agent(
        role="{topic} Researcher",
        goal="Express hot takes on {topic}.",
        backstory="You have a lot of experience with {topic}.",
    )
    
    task = Task(
        description="Give me an analysis around {topic}.",
        expected_output="1 bullet point about {topic} that's under 15 words.",
        agent=agent,
    )
    
    crew = Crew(agents=[agent], tasks=[task])
    
    # Mock the kickoff method to avoid API calls
    expected_outputs = [
        CrewOutput(raw="Dogs are loyal companions."),
        CrewOutput(raw="Cats are independent pets."),
        CrewOutput(raw="Apples are nutritious fruits."),
    ]
    
    with patch.object(Crew, "copy") as mock_copy:
        # Setup mock crew copies
        crew_copies = []
        for i in range(len(inputs)):
            crew_copy = MagicMock()
            crew_copy.kickoff.return_value = expected_outputs[i]
            crew_copies.append(crew_copy)
        mock_copy.side_effect = crew_copies
        
        results = crew.kickoff_for_each_parallel(inputs=inputs)
        
        assert len(results) == len(inputs)
        # Since ThreadPoolExecutor returns results in completion order, not input order,
        # we just check that all expected outputs are in the results
        result_texts = [result.raw for result in results]
        expected_texts = [output.raw for output in expected_outputs]
        for expected_text in expected_texts:
            assert expected_text in result_texts


def test_kickoff_for_each_parallel_empty_input():
    """Tests if kickoff_for_each_parallel handles an empty input list."""
    agent = Agent(
        role="{topic} Researcher",
        goal="Express hot takes on {topic}.",
        backstory="You have a lot of experience with {topic}.",
    )
    
    task = Task(
        description="Give me an analysis around {topic}.",
        expected_output="1 bullet point about {topic} that's under 15 words.",
        agent=agent,
    )
    
    crew = Crew(agents=[agent], tasks=[task])
    results = crew.kickoff_for_each_parallel(inputs=[])
    assert results == []


def test_kickoff_for_each_parallel_invalid_input():
    """Tests if kickoff_for_each_parallel raises TypeError for invalid input types."""
    
    agent = Agent(
        role="{topic} Researcher",
        goal="Express hot takes on {topic}.",
        backstory="You have a lot of experience with {topic}.",
    )
    
    task = Task(
        description="Give me an analysis around {topic}.",
        expected_output="1 bullet point about {topic} that's under 15 words.",
        agent=agent,
    )
    
    crew = Crew(agents=[agent], tasks=[task])
    
    # No need to mock here since we're testing input validation which happens before any API calls
    with pytest.raises(TypeError):
        # Pass a string instead of a list
        crew.kickoff_for_each_parallel("invalid input")


def test_kickoff_for_each_parallel_error_handling():
    """Tests error handling in kickoff_for_each_parallel when kickoff raises an error."""
    
    inputs = [
        {"topic": "dog"},
        {"topic": "cat"},
        {"topic": "apple"},
    ]
    
    agent = Agent(
        role="{topic} Researcher",
        goal="Express hot takes on {topic}.",
        backstory="You have a lot of experience with {topic}.",
    )
    
    task = Task(
        description="Give me an analysis around {topic}.",
        expected_output="1 bullet point about {topic} that's under 15 words.",
        agent=agent,
    )
    
    crew = Crew(agents=[agent], tasks=[task])
    
    with patch.object(Crew, "copy") as mock_copy:
        # Setup mock crew copies
        crew_copies = []
        for i in range(len(inputs)):
            crew_copy = MagicMock()
            # Make the third crew copy raise an exception
            if i == 2:
                crew_copy.kickoff.side_effect = Exception("Simulated kickoff error")
            else:
                crew_copy.kickoff.return_value = f"Output for {inputs[i]['topic']}"
            crew_copies.append(crew_copy)
        mock_copy.side_effect = crew_copies
        
        with pytest.raises(Exception, match="Simulated kickoff error"):
            crew.kickoff_for_each_parallel(inputs=inputs)


def test_kickoff_for_each_parallel_max_workers():
    """Tests if kickoff_for_each_parallel respects the max_workers parameter."""
    
    inputs = [
        {"topic": "dog"},
        {"topic": "cat"},
        {"topic": "apple"},
    ]
    
    agent = Agent(
        role="{topic} Researcher",
        goal="Express hot takes on {topic}.",
        backstory="You have a lot of experience with {topic}.",
    )
    
    task = Task(
        description="Give me an analysis around {topic}.",
        expected_output="1 bullet point about {topic} that's under 15 words.",
        agent=agent,
    )
    
    crew = Crew(agents=[agent], tasks=[task])
    
    # Mock both ThreadPoolExecutor and crew.copy to avoid API calls
    with patch.object(concurrent.futures, "ThreadPoolExecutor", wraps=concurrent.futures.ThreadPoolExecutor) as mock_executor:
        with patch.object(Crew, "copy") as mock_copy:
            # Setup mock crew copies
            crew_copies = []
            for _ in range(len(inputs)):
                crew_copy = MagicMock()
                crew_copy.kickoff.return_value = CrewOutput(raw="Test output")
                crew_copies.append(crew_copy)
            mock_copy.side_effect = crew_copies
            
            crew.kickoff_for_each_parallel(inputs=inputs, max_workers=2)
            mock_executor.assert_called_once_with(max_workers=2)
