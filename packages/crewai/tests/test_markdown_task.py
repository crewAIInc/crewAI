"""Test the markdown attribute in Task class."""

import pytest
from pydantic import BaseModel

from crewai import Agent, Task


@pytest.mark.parametrize(
    "markdown_enabled,should_contain_instructions",
    [
        (True, True),
        (False, False),
    ],
)
def test_markdown_option_in_task_prompt(markdown_enabled, should_contain_instructions):
    """Test that markdown flag correctly controls the inclusion of markdown formatting instructions."""
    
    researcher = Agent(
        role="Researcher",
        goal="Research a topic",
        backstory="You're a researcher specialized in providing well-formatted content.",
        allow_delegation=False,
    )

    task = Task(
        description="Research advances in AI in 2023",
        expected_output="A summary of key AI advances in 2023",
        markdown=markdown_enabled,
        agent=researcher,
    )

    prompt = task.prompt()
    
    assert "Research advances in AI in 2023" in prompt
    assert "A summary of key AI advances in 2023" in prompt
    
    if should_contain_instructions:
        assert "Your final answer MUST be formatted in Markdown syntax." in prompt
        assert "Use # for headers" in prompt
        assert "Use ** for bold text" in prompt
    else:
        assert "Your final answer MUST be formatted in Markdown syntax." not in prompt


def test_markdown_with_empty_description():
    """Test markdown formatting with empty description."""
    
    researcher = Agent(
        role="Researcher",
        goal="Research a topic",
        backstory="You're a researcher.",
        allow_delegation=False,
    )

    task = Task(
        description="",
        expected_output="A summary",
        markdown=True,
        agent=researcher,
    )

    prompt = task.prompt()
    
    assert prompt.strip() != ""
    assert "A summary" in prompt
    assert "Your final answer MUST be formatted in Markdown syntax." in prompt


def test_markdown_with_complex_output_format():
    """Test markdown with JSON output format to ensure compatibility."""
    
    class ResearchOutput(BaseModel):
        title: str
        findings: list[str]
    
    researcher = Agent(
        role="Researcher",
        goal="Research a topic",
        backstory="You're a researcher.",
        allow_delegation=False,
    )

    task = Task(
        description="Research topic",
        expected_output="Research results",
        markdown=True,
        output_json=ResearchOutput,
        agent=researcher,
    )

    prompt = task.prompt()
    
    assert "Your final answer MUST be formatted in Markdown syntax." in prompt
    assert "Research topic" in prompt
    assert "Research results" in prompt
