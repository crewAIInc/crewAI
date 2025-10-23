"""Test lazy loading of knowledge sources to prevent premature authentication errors."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crewai import Agent, Crew, Task
from crewai.knowledge.knowledge import Knowledge
from crewai.knowledge.source.text_file_knowledge_source import TextFileKnowledgeSource


def test_knowledge_sources_not_loaded_during_initialization(tmpdir):
    """Test that knowledge sources are not loaded during agent/crew initialization."""
    # Create a test file
    test_file = Path(tmpdir) / "test.txt"
    test_file.write_text("Test content")
    
    # Create knowledge source
    knowledge_source = TextFileKnowledgeSource(file_paths=[test_file])
    
    # Mock the storage to avoid actual database operations
    with patch('crewai.knowledge.knowledge.KnowledgeStorage'):
        # Create Knowledge object
        knowledge = Knowledge(
            collection_name="test",
            sources=[knowledge_source],
            embedder=None
        )
        
        # Verify that sources are not loaded yet
        assert knowledge._sources_loaded is False


def test_knowledge_sources_loaded_on_first_query(tmpdir):
    """Test that knowledge sources are loaded only when first queried."""
    # Create a test file
    test_file = Path(tmpdir) / "test.txt"
    test_file.write_text("Test content")
    
    # Create knowledge source
    knowledge_source = TextFileKnowledgeSource(file_paths=[test_file])
    
    # Mock the storage to avoid actual database operations
    with patch('crewai.knowledge.knowledge.KnowledgeStorage') as MockStorage:
        mock_storage = MagicMock()
        mock_storage.search.return_value = []
        MockStorage.return_value = mock_storage
        
        # Create Knowledge object
        knowledge = Knowledge(
            collection_name="test",
            sources=[knowledge_source],
            embedder=None
        )
        
        # Verify sources not loaded yet
        assert knowledge._sources_loaded is False
        
        with patch.object(Knowledge, 'add_sources', wraps=knowledge.add_sources) as mock_add_sources:
            # Query should trigger loading
            knowledge.query(["test query"])
            
            # Verify add_sources was called
            mock_add_sources.assert_called_once()
        
        # Verify sources are now marked as loaded
        assert knowledge._sources_loaded is True
        
        # Query again - add_sources should not be called again
        with patch.object(Knowledge, 'add_sources', wraps=knowledge.add_sources) as mock_add_sources:
            knowledge.query(["another query"])
            mock_add_sources.assert_not_called()


def test_agent_with_knowledge_sources_no_immediate_loading(tmpdir):
    """Test that creating an agent with knowledge sources doesn't immediately load them."""
    # Create a test file
    test_file = Path(tmpdir) / "test.txt"
    test_file.write_text("Test content")
    
    # Create knowledge source
    knowledge_source = TextFileKnowledgeSource(file_paths=[test_file])
    
    # Mock the storage to avoid authentication errors
    with patch('crewai.knowledge.knowledge.KnowledgeStorage'):
        # Create agent with knowledge source
        agent = Agent(
            role="Test Agent",
            goal="Test goal",
            backstory="Test backstory",
            knowledge_sources=[knowledge_source],
        )
        
        # Create task and crew
        task = Task(
            description="Test task",
            expected_output="Test output",
            agent=agent
        )
        
        crew = Crew(
            agents=[agent],
            tasks=[task],
        )
        
        # but sources should not be loaded yet
        if agent.knowledge is not None:
            assert agent.knowledge._sources_loaded is False


def test_knowledge_add_sources_can_still_be_called_explicitly():
    """Test that add_sources can still be called explicitly if needed."""
    # Create a mock knowledge source
    mock_source = MagicMock()
    mock_source.add = MagicMock()
    
    # Mock the storage
    with patch('crewai.knowledge.knowledge.KnowledgeStorage') as MockStorage:
        mock_storage = MagicMock()
        MockStorage.return_value = mock_storage
        
        # Create Knowledge object
        knowledge = Knowledge(
            collection_name="test",
            sources=[mock_source],
            embedder=None
        )
        
        # Explicitly call add_sources
        knowledge.add_sources()
        
        # Verify add was called
        mock_source.add.assert_called_once()
        
        # Verify sources are marked as loaded
        assert knowledge._sources_loaded is True
