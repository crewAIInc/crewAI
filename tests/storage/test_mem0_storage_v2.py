import os
from unittest.mock import MagicMock, patch

import pytest
from mem0.client.main import MemoryClient
from mem0.memory.main import Memory

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.memory.storage.mem0_storage import Mem0Storage
from crewai.task import Task


class MockCrew:
    def __init__(self, memory_config):
        self.memory_config = memory_config
        self.agents = [MagicMock(role="Test Agent")]


@pytest.fixture
def mock_mem0_memory_client():
    """Fixture to create a mock MemoryClient instance"""
    mock_memory = MagicMock(spec=MemoryClient)
    return mock_memory


@pytest.fixture
def mem0_storage_with_v2_api(mock_mem0_memory_client):
    """Fixture to create a Mem0Storage instance with v2 API configuration"""
    
    with patch.object(MemoryClient, "__new__", return_value=mock_mem0_memory_client):
        crew = MockCrew(
            memory_config={
                "provider": "mem0",
                "config": {
                    "user_id": "test_user",
                    "api_key": "ABCDEFGH",
                    "version": "v2",  # Explicitly set to v2
                },
            }
        )
        
        mem0_storage = Mem0Storage(type="short_term", crew=crew)
        return mem0_storage, mock_mem0_memory_client


@pytest.fixture
def mem0_storage_with_run_id(mock_mem0_memory_client):
    """Fixture to create a Mem0Storage instance with run_id configuration"""
    
    with patch.object(MemoryClient, "__new__", return_value=mock_mem0_memory_client):
        crew = MockCrew(
            memory_config={
                "provider": "mem0",
                "config": {
                    "user_id": "test_user",
                    "api_key": "ABCDEFGH",
                    "version": "v2",
                    "run_id": "test-session-123",  # Set run_id
                },
            }
        )
        
        mem0_storage = Mem0Storage(type="short_term", crew=crew)
        return mem0_storage, mock_mem0_memory_client


def test_mem0_storage_v2_initialization(mem0_storage_with_v2_api):
    """Test that Mem0Storage initializes correctly with v2 API configuration"""
    mem0_storage, _ = mem0_storage_with_v2_api
    
    assert mem0_storage.version == "v2"
    
    assert mem0_storage.run_id is None


def test_mem0_storage_with_run_id_initialization(mem0_storage_with_run_id):
    """Test that Mem0Storage initializes correctly with run_id configuration"""
    mem0_storage, _ = mem0_storage_with_run_id
    
    assert mem0_storage.version == "v2"
    
    assert mem0_storage.run_id == "test-session-123"


def test_save_method_with_v2_api(mem0_storage_with_v2_api):
    """Test save method with v2 API"""
    mem0_storage, mock_memory_client = mem0_storage_with_v2_api
    mock_memory_client.add = MagicMock()
    
    test_value = "This is a test memory"
    test_metadata = {"key": "value"}
    
    mem0_storage.save(test_value, test_metadata)
    
    mock_memory_client.add.assert_called_once()
    call_args = mock_memory_client.add.call_args[1]
    
    assert call_args["version"] == "v2"
    assert "run_id" not in call_args
    assert call_args["agent_id"] == "Test_Agent"
    assert call_args["metadata"] == {"type": "short_term", "key": "value"}


def test_save_method_with_run_id(mem0_storage_with_run_id):
    """Test save method with run_id"""
    mem0_storage, mock_memory_client = mem0_storage_with_run_id
    mock_memory_client.add = MagicMock()
    
    test_value = "This is a test memory"
    test_metadata = {"key": "value"}
    
    mem0_storage.save(test_value, test_metadata)
    
    mock_memory_client.add.assert_called_once()
    call_args = mock_memory_client.add.call_args[1]
    
    assert call_args["version"] == "v2"
    assert call_args["run_id"] == "test-session-123"
    assert call_args["agent_id"] == "Test_Agent"
    assert call_args["metadata"] == {"type": "short_term", "key": "value"}


def test_search_method_with_v2_api(mem0_storage_with_v2_api):
    """Test search method with v2 API"""
    mem0_storage, mock_memory_client = mem0_storage_with_v2_api
    mock_results = {"results": [{"score": 0.9, "content": "Result 1"}, {"score": 0.4, "content": "Result 2"}]}
    mock_memory_client.search = MagicMock(return_value=mock_results)
    
    results = mem0_storage.search("test query", limit=5, score_threshold=0.5)
    
    mock_memory_client.search.assert_called_once()
    call_args = mock_memory_client.search.call_args[1]
    
    assert call_args["version"] == "v2"
    assert "run_id" not in call_args
    assert call_args["query"] == "test query"
    assert call_args["limit"] == 5
    
    assert len(results) == 1
    assert results[0]["content"] == "Result 1"


def test_search_method_with_run_id(mem0_storage_with_run_id):
    """Test search method with run_id"""
    mem0_storage, mock_memory_client = mem0_storage_with_run_id
    mock_results = {"results": [{"score": 0.9, "content": "Result 1"}, {"score": 0.4, "content": "Result 2"}]}
    mock_memory_client.search = MagicMock(return_value=mock_results)
    
    results = mem0_storage.search("test query", limit=5, score_threshold=0.5)
    
    mock_memory_client.search.assert_called_once()
    call_args = mock_memory_client.search.call_args[1]
    
    assert call_args["version"] == "v2"
    assert call_args["run_id"] == "test-session-123"
    assert call_args["query"] == "test query"
    assert call_args["limit"] == 5
    
    assert len(results) == 1
    assert results[0]["content"] == "Result 1"
