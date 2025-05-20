"""Test the markdown attribute in Task class."""

import pytest
from unittest.mock import patch

from crewai import Agent, Task


def test_markdown_option_in_task_prompt():
    """Test that when markdown=True, the task prompt includes markdown formatting instructions."""
    
    researcher = Agent(
        role="Researcher",
        goal="Research a topic",
        backstory="You're a researcher specialized in providing well-formatted content.",
        allow_delegation=False,
    )

    task = Task(
        description="Research advances in AI in 2023",
        expected_output="A summary of key AI advances in 2023",
        markdown=True,
        agent=researcher,
    )

    prompt = task.prompt()
    
    assert "Research advances in AI in 2023" in prompt
    assert "A summary of key AI advances in 2023" in prompt
    assert "Your final answer MUST be formatted in Markdown syntax." in prompt


def test_markdown_option_not_in_task_prompt_by_default():
    """Test that by default (markdown=False), the task prompt does not include markdown formatting instructions."""
    
    researcher = Agent(
        role="Researcher",
        goal="Research a topic",
        backstory="You're a researcher specialized in providing well-formatted content.",
        allow_delegation=False,
    )

    task = Task(
        description="Research advances in AI in 2023",
        expected_output="A summary of key AI advances in 2023",
        agent=researcher,
    )

    prompt = task.prompt()
    
    assert "Research advances in AI in 2023" in prompt
    assert "A summary of key AI advances in 2023" in prompt
    assert "Your final answer MUST be formatted in Markdown syntax." not in prompt
