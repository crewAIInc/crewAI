"""Test CustomStorageKnowledgeSource functionality."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from crewai.knowledge.knowledge import Knowledge
from crewai.knowledge.source.custom_storage_knowledge_source import CustomStorageKnowledgeSource
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage


@pytest.fixture
def custom_storage():
    """Create a custom KnowledgeStorage instance."""
    storage = KnowledgeStorage(collection_name="test_collection")
    return storage


def test_custom_storage_knowledge_source(custom_storage):
    """Test that a CustomStorageKnowledgeSource can be created with a pre-existing storage."""
    source = CustomStorageKnowledgeSource(collection_name="test_collection")
    
    assert source is not None
    assert source.collection_name == "test_collection"


def test_custom_storage_knowledge_source_with_knowledge(custom_storage):
    """Test that a CustomStorageKnowledgeSource can be used with Knowledge."""
    source = CustomStorageKnowledgeSource(collection_name="test_collection")
    
    with patch.object(KnowledgeStorage, 'initialize_knowledge_storage'):
        with patch.object(CustomStorageKnowledgeSource, 'add'):
            knowledge = Knowledge(
                sources=[source],
                storage=custom_storage,
                collection_name="test_collection"
            )
    
    assert knowledge is not None
    assert knowledge.sources[0] == source
    assert knowledge.storage == custom_storage


def test_custom_storage_knowledge_source_with_crew():
    """Test that a CustomStorageKnowledgeSource can be used with Crew."""
    from crewai.agent import Agent
    from crewai.crew import Crew
    from crewai.task import Task
    
    storage = KnowledgeStorage(collection_name="test_collection")
    
    source = CustomStorageKnowledgeSource(collection_name="test_collection")
    source.storage = storage
    
    agent = Agent(role="test", goal="test", backstory="test")
    task = Task(description="test", expected_output="test", agent=agent)
    
    with patch.object(KnowledgeStorage, 'initialize_knowledge_storage'):
        with patch.object(CustomStorageKnowledgeSource, 'add'):
            crew = Crew(
                agents=[agent],
                tasks=[task],
                knowledge_sources=[source]
            )
    
    assert crew is not None
    assert crew.knowledge_sources[0] == source
