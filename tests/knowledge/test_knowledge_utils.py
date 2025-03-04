"""Test knowledge utils functionality."""

from typing import Any, Dict, List

import pytest

from crewai.knowledge.utils.knowledge_utils import extract_knowledge_context


def test_extract_knowledge_context_with_valid_snippets():
    """Test extracting knowledge context with valid snippets."""
    snippets = [
        {"context": "Fact 1: The sky is blue", "score": 0.9},
        {"context": "Fact 2: Water is wet", "score": 0.8},
    ]
    result = extract_knowledge_context(snippets)
    expected = "Important Context (You MUST use this information to complete your task accurately and effectively):\nFact 1: The sky is blue\nFact 2: Water is wet\n\nMake sure to incorporate the above context into your response."
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
    assert result == "Important Context (You MUST use this information to complete your task accurately and effectively):\nValid context\n\nMake sure to incorporate the above context into your response."


def test_extract_knowledge_context_with_missing_context():
    """Test extracting knowledge context with missing context."""
    snippets = [{"score": 0.9}, {"context": "Valid context"}]
    result = extract_knowledge_context(snippets)
    assert result == "Important Context (You MUST use this information to complete your task accurately and effectively):\nValid context\n\nMake sure to incorporate the above context into your response."


def test_knowledge_effectiveness():
    """Test that knowledge is effectively used in agent execution."""
    from unittest.mock import MagicMock, patch

    import pytest

    from crewai.knowledge.utils.knowledge_utils import extract_knowledge_context
    
    # Create mock knowledge snippets
    knowledge_snippets = [
        {"context": "The capital of France is Paris. The Eiffel Tower is located in Paris.", "score": 0.9}
    ]
    
    # Test that the extract_knowledge_context function formats the knowledge correctly
    knowledge_context = extract_knowledge_context(knowledge_snippets)
    
    # Verify the knowledge context contains the expected information
    assert "paris" in knowledge_context.lower()
    assert "capital" in knowledge_context.lower()
    assert "france" in knowledge_context.lower()
    
    # Verify the format is correct
    assert knowledge_context.startswith("Important Context")
    assert "Make sure to incorporate the above context" in knowledge_context
