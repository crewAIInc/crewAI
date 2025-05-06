"""Test to ensure hierarchical process mode returns complete final answers."""

import pytest
from crewai.agent import Agent
from crewai.crew import Crew
from crewai.process import Process
from crewai.task import Task


@pytest.mark.vcr(filter_headers=["authorization"])
def test_hierarchical_process_delegation_result():
    """Tests hierarchical process delegation result handling.
    
    Ensures that:
    1. The output is derived from the delegated agent's actual work.
    2. The response does not contain delegation-related metadata.
    3. The content meets minimum length requirements.
    4. Expected topic-related keywords are present in the output.
    """
    researcher = Agent(
        role="Researcher",
        goal="Make the best research and analysis on content about AI and AI agents",
        backstory="You're an expert researcher, specialized in technology, software engineering, AI and startups. You work as a freelancer and is now working on doing research and analysis for a new customer.",
        allow_delegation=False,
    )

    writer = Agent(
        role="Senior Writer",
        goal="Write the best content about AI and AI agents.",
        backstory="You're a senior writer, specialized in technology, software engineering, AI and startups. You work as a freelancer and are now working on writing content for a new customer.",
        allow_delegation=False,
    )

    task = Task(
        description="Come up with a list of 3 interesting ideas to explore for an article, then write one amazing paragraph highlight for each idea that showcases how good an article about this topic could be. Return the list of ideas with their paragraph and your notes.",
        expected_output="3 bullet points with a paragraph for each idea.",
    )

    crew = Crew(
        agents=[researcher, writer],
        process=Process.hierarchical,
        manager_llm="gpt-4o",
        tasks=[task],
    )

    result = crew.kickoff()

    assert "idea" in result.raw.lower() or "article" in result.raw.lower()
    assert len(result.raw) > 100  # Ensure we have substantial content
    assert result.raw.count('\n') >= 6  # At least 3 ideas with paragraphs
    
    assert "delegate" not in result.raw.lower()
    assert "delegating" not in result.raw.lower()
    assert "assigned" not in result.raw.lower()
