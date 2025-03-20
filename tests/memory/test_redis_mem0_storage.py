import os
import pytest
from unittest.mock import MagicMock, patch

from crewai.memory.storage.mem0_storage import Mem0Storage


class TestMem0RedisIntegration:
    @pytest.fixture
    def mock_memory(self):
        with patch("mem0.memory.main.Memory") as mock_memory:
            mock_memory_instance = MagicMock()
            mock_memory.from_config.return_value = mock_memory_instance
            yield mock_memory

    def test_mem0_with_redis_config(self, mock_memory):
        # Create a mock crew with Redis vector store configuration
        mock_crew = MagicMock()
        mock_crew.memory_config = {
            "provider": "mem0",
            "config": {
                "user_id": "test-user",
                "api_key": "test-api-key",
                "vector_store": {
                    "provider": "redis",
                    "config": {
                        "collection_name": "test_collection",
                        "embedding_model_dims": 1536,
                        "redis_url": "redis://localhost:6379/0"
                    }
                }
            }
        }

        # Create Mem0Storage instance
        with patch("crewai.memory.storage.mem0_storage.MemoryClient"):
            storage = Mem0Storage(type="user", crew=mock_crew)

        # Check that Memory.from_config was called with correct parameters
        mock_memory.from_config.assert_called_once()
        config_arg = mock_memory.from_config.call_args[0][0]
        assert "vector_store" in config_arg
        assert config_arg["vector_store"]["provider"] == "redis"
        assert config_arg["vector_store"]["config"]["redis_url"] == "redis://localhost:6379/0"

    def test_fallback_to_memory_client(self):
        # Create a mock crew without vector store configuration
        mock_crew = MagicMock()
        mock_crew.memory_config = {
            "provider": "mem0",
            "config": {
                "user_id": "test-user",
                "api_key": "test-api-key"
            }
        }

        # Mock MemoryClient
        with patch("crewai.memory.storage.mem0_storage.MemoryClient") as mock_client:
            mock_client_instance = MagicMock()
            mock_client.return_value = mock_client_instance
            
            # Create Mem0Storage instance
            storage = Mem0Storage(type="user", crew=mock_crew)
            
            # Check that MemoryClient was called (fallback path)
            mock_client.assert_called_once()
            assert mock_client.call_args[1]["api_key"] == "test-api-key"
