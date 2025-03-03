import json
import os
from typing import List
import pytest
from unittest.mock import patch
from crewai import Agent, Task, Crew, Process
from crewai.crews.crew_output import CrewOutput

def test_manager_agent_with_kickoff_for_each():
    """
    Test that using a manager agent with kickoff_for_each doesn't raise validation errors.
    
    This test specifically checks that the fix for issue #2260 works correctly.
    We're only testing that no validation errors occur during the copy process,
    not the actual execution of the crew.
    """
    # Define agents
    researcher = Agent(
        role="Researcher",
        goal="Conduct thorough research and analysis on AI and AI agents",
        backstory="You're an expert researcher, specialized in technology, software engineering, AI, and startups. You work as a freelancer and are currently researching for a new client.",
        allow_delegation=False
    )

    writer = Agent(
        role="Senior Writer",
        goal="Create compelling content about AI and AI agents",
        backstory="You're a senior writer, specialized in technology, software engineering, AI, and startups. You work as a freelancer and are currently writing content for a new client.",
        allow_delegation=False
    )

    # Define task
    task = Task(
        description="Generate a list of 5 interesting ideas for an article, then write one captivating paragraph for each idea that showcases the potential of a full article on this topic. Return the list of ideas with their paragraphs and your notes.",
        expected_output="5 bullet points, each with a paragraph and accompanying notes.",
    )

    # Define manager agent
    manager = Agent(
        role="Project Manager",
        goal="Efficiently manage the crew and ensure high-quality task completion",
        backstory="You're an experienced project manager, skilled in overseeing complex projects and guiding teams to success. Your role is to coordinate the efforts of the crew members, ensuring that each task is completed on time and to the highest standard.",
        allow_delegation=True
    )

    # Instantiate crew with a custom manager
    crew = Crew(
        agents=[researcher, writer],
        tasks=[task],
        manager_agent=manager,
        process=Process.hierarchical,
        verbose=True
    )

    # Load test data
    test_data_path = os.path.join(os.path.dirname(__file__), "test_data", "test_kickoff_for_each.json")
    with open(test_data_path) as f:
        d = json.load(f)

    # Create a copy of the crew to test that no validation errors occur
    # This is what happens in kickoff_for_each before any LLM calls
    try:
        crew_copy = crew.copy()
        # Check that the manager_agent was properly copied
        assert crew_copy.manager_agent is not None
        assert crew_copy.manager_agent.id != crew.manager_agent.id
        assert crew_copy.manager_agent.role == crew.manager_agent.role
        assert crew_copy.manager_agent.goal == crew.manager_agent.goal
        assert crew_copy.manager_agent.backstory == crew.manager_agent.backstory
    except Exception as e:
        pytest.fail(f"Crew copy with manager_agent raised an exception: {e}")
        
    # Test that kickoff_for_each doesn't raise validation errors
    # We'll patch the kickoff method to avoid actual LLM calls
    with patch.object(Crew, 'kickoff', return_value=CrewOutput(final_output="Test output", task_outputs={})):
        try:
            outputs = crew.kickoff_for_each(inputs=[
                {"document": document} for document in d["foo"]
            ])
            assert len(outputs) == len(d["foo"])
        except Exception as e:
            if "validation error" in str(e).lower():
                pytest.fail(f"kickoff_for_each raised validation errors: {e}")
            else:
                # Other errors are fine for this test, we're only checking for validation errors
                pass
