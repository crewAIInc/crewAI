"""Test knowledge utils functionality."""

from typing import Dict, List, Any

import pytest

from crewai.knowledge.utils.knowledge_utils import extract_knowledge_context


def test_extract_knowledge_context_with_valid_snippets():
    """Test extracting knowledge context with valid snippets."""
    snippets = [
        {"context": "Fact 1: The sky is blue", "score": 0.9},
        {"context": "Fact 2: Water is wet", "score": 0.8},
    ]
    result = extract_knowledge_context(snippets)
    expected = "Additional Information: Fact 1: The sky is blue\nFact 2: Water is wet"
    assert result == expected


def test_extract_knowledge_context_with_empty_snippets():
    """Test extracting knowledge context with empty snippets."""
    snippets: List[Dict[str, Any]] = []
    result = extract_knowledge_context(snippets)
    assert result == ""


def test_extract_knowledge_context_with_none_snippets():
    """Test extracting knowledge context with None snippets."""
    snippets = [None, {"context": "Valid context"}]  # type: ignore
    result = extract_knowledge_context(snippets)
    assert result == "Additional Information: Valid context"


def test_extract_knowledge_context_with_missing_context():
    """Test extracting knowledge context with missing context."""
    snippets = [{"score": 0.9}, {"context": "Valid context"}]
    result = extract_knowledge_context(snippets)
    assert result == "Important Context (You MUST use this information to complete your task accurately and effectively):\nValid context\n\nMake sure to incorporate the above context into your response."


def test_knowledge_effectiveness():
    """Test that knowledge is effectively used in agent execution."""
    from crewai import Agent, Task, Crew
    from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource

    content = "The capital of France is Paris. The Eiffel Tower is located in Paris."
    string_source = StringKnowledgeSource(content=content)
    
    agent = Agent(
        role="Geography Expert",
        goal="Answer questions about geography accurately",
        backstory="You are an expert in geography",
        knowledge_sources=[string_source],
    )
    
    task = Task(
        description="What is the capital of France?",
        expected_output="The capital of France",
        agent=agent,
    )
    
    crew = Crew(agents=[agent], tasks=[task])
    result = crew.kickoff()
    
    assert "paris" in result.raw.lower()
    assert "capital" in result.raw.lower()
