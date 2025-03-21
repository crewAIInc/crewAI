import os
from unittest.mock import MagicMock, patch

import pytest
from mem0.memory.main import Memory
from mem0.client.main import MemoryClient

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.memory.storage.mem0_storage import Mem0Storage
from crewai.task import Task


@pytest.fixture
def mock_mem0_memory():
    """Fixture to create a mock Memory instance"""
    mock_memory = MagicMock(spec=Memory)
    return mock_memory

@pytest.fixture
def mem0_storage_with_mocked_config(mock_mem0_memory):
    """Fixture to create a Mem0Storage instance with mocked dependencies"""
    
    # Use minimal config with just required structure
    config = {
        "vector_store": {
            "provider": "mock_vector_store",
            "config": {
                "host": "localhost",
                "port": 6333
            }
        },
        "llm": {
            "provider": "mock_llm",
            "config": {
                "api_key": "mock-api-key",
                "model": "mock-model"
            }
        },
        "embedder": {
            "provider": "mock_embedder",
            "config": {
                "api_key": "mock-api-key",
                "model": "mock-model"
            }
        },
        "graph_store": {
            "provider": "mock_graph_store",
            "config": {
                "url": "mock-url",
                "username": "mock-user",
                "password": "mock-password"
            }
        },
        "history_db_path": "/mock/path",
        "version": "test-version",
        "custom_fact_extraction_prompt": "mock prompt 1",
        "custom_update_memory_prompt": "mock prompt 2"
    }

    agent = Agent(
        role="Researcher",
        goal="Test goal",
        backstory="Test backstory",
        tools=[],
        verbose=False,
    )

    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False,
        memory=True,
        memory_config={
            "provider": "mem0",
            "config": {"user_id": "test_user", 'local_mem0_config': config},
        },
    )
    
    # Patch the Memory class to return our mock
    with patch('mem0.memory.Memory', return_value=mock_mem0_memory):
        with patch('crewai.memory.mem0_storage.validate_local_mem0_config', return_value=True):
            mem0_storage = Mem0Storage(type="short_term", crew=crew)
            return mem0_storage

def test_mem0_storage_initialization(mem0_storage_with_mocked_config, mock_mem0_memory):
    """Test that Mem0Storage initializes correctly with the mocked config"""
    assert mem0_storage_with_mocked_config.memory_type == "short_term"
    assert mem0_storage_with_mocked_config.memory is mock_mem0_memory

def test_mem0_storage_save(mem0_storage_with_mocked_config, mock_mem0_memory):
    """Test the save method of Mem0Storage"""
    mem0_storage_with_mocked_config.save(
        value="test value " * 12, 
        metadata={"task": "test_task"}
    )
    mock_mem0_memory.save.assert_called_once()

def test_mem0_storage_query(mem0_storage_with_mocked_config, mock_mem0_memory):
    """Test the query method of Mem0Storage"""
    test_query = "Test query"
    mock_mem0_memory.query.return_value = "Mock query result"
    
    result = mem0_storage_with_mocked_config.query(test_query)
    
    mock_mem0_memory.query.assert_called_once_with(test_query)
    assert result == "Mock query result"

@pytest.fixture
def mock_mem0_memory_client():
    """Fixture to create a mock MemoryClient instance"""
    mock_memory = MagicMock(spec=MemoryClient)
    return mock_memory

@pytest.fixture
def mem0_storage_with_memory_client(mock_mem0_memory_client):
    """Fixture to create a Mem0Storage instance with mocked dependencies"""

    agent = Agent(
        role="Researcher",
        goal="Test goal",
        backstory="Test backstory",
        tools=[],
        verbose=False,
    )

    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False,
        memory=True,
        memory_config={
            "provider": "mem0",
            "config": {"user_id": "test_user", 'api_key': "mock-api-key"},
        },
    )
    
    with patch('mem0.client.main.MemoryClient', return_value=mock_mem0_memory_client):
        mem0_storage = Mem0Storage(type="short_term", crew=crew)
        return mem0_storage

def test_mem0_storage_initialization_client(mem0_storage_with_memory_client, mock_mem0_memory_client):
    """Test Mem0Storage initialization with MemoryClient"""
    assert mem0_storage_with_memory_client.memory_type == "short_term"
    assert mem0_storage_with_memory_client.memory is mock_mem0_memory_client

def test_mem0_storage_save_client(mem0_storage_with_memory_client, mock_mem0_memory_client):
    """Test the save method with MemoryClient"""
    mem0_storage_with_memory_client.save(
        value="test value " * 12, 
        metadata={"task": "test_task"}
    )
    mock_mem0_memory_client.save.assert_called_once()

def test_mem0_storage_query_client(mem0_storage_with_memory_client, mock_mem0_memory_client):
    """Test the query method with MemoryClient"""
    test_query = "Test query"
    mock_mem0_memory_client.query.return_value = "Mock query result"
    
    result = mem0_storage_with_memory_client.query(test_query)
    
    mock_mem0_memory_client.query.assert_called_once_with(test_query)
    assert result == "Mock query result"
