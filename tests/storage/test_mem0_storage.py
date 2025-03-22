import os
from unittest.mock import MagicMock, patch

import pytest
from mem0.client.main import MemoryClient
from mem0.memory.main import Memory

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.memory.storage.mem0_storage import Mem0Storage
from crewai.task import Task


# Define the class (if not already defined)
class MockCrew:
    def __init__(self, memory_config):
        self.memory_config = memory_config


@pytest.fixture
def mock_mem0_memory():
    """Fixture to create a mock Memory instance"""
    mock_memory = MagicMock(spec=Memory)
    return mock_memory


@pytest.fixture
def mem0_storage_with_mocked_config(mock_mem0_memory):
    """Fixture to create a Mem0Storage instance with mocked dependencies"""

    # Patch the Memory class to return our mock
    with patch('mem0.memory.main.Memory.from_config', return_value=mock_mem0_memory):
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

        # Instantiate the class with memory_config
        crew = MockCrew(
            memory_config={
                "provider": "mem0",
                "config": {"user_id": "test_user", "local_mem0_config": config},
            }
        )

        mem0_storage = Mem0Storage(type="short_term", crew=crew)
        return mem0_storage


def test_mem0_storage_initialization(mem0_storage_with_mocked_config, mock_mem0_memory):
    """Test that Mem0Storage initializes correctly with the mocked config"""
    assert mem0_storage_with_mocked_config.memory_type == "short_term"
    assert mem0_storage_with_mocked_config.memory is mock_mem0_memory


@pytest.fixture
def mock_mem0_memory_client():
    """Fixture to create a mock MemoryClient instance"""
    mock_memory = MagicMock(spec=MemoryClient)
    return mock_memory


@pytest.fixture
def mem0_storage_with_memory_client(mock_mem0_memory_client):
    """Fixture to create a Mem0Storage instance with mocked dependencies"""

    # We need to patch the MemoryClient before it's instantiated
    with patch.object(MemoryClient, '__new__', return_value=mock_mem0_memory_client):
            crew = MockCrew(
                memory_config={
                    "provider": "mem0",
                    "config": {"user_id": "test_user", "api_key": "ABCDEFGH"},
                }
            )

            mem0_storage = Mem0Storage(type="short_term", crew=crew)
            return mem0_storage


def test_mem0_storage_with_memory_client_initialization(mem0_storage_with_memory_client, mock_mem0_memory_client):
    """Test Mem0Storage initialization with MemoryClient"""
    assert mem0_storage_with_memory_client.memory_type == "short_term"
    assert mem0_storage_with_memory_client.memory is mock_mem0_memory_client
