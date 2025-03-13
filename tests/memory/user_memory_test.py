from unittest.mock import patch, PropertyMock

import pytest

from crewai.memory.storage.rag_storage import RAGStorage
from crewai.memory.user.user_memory import UserMemory


@patch('crewai.memory.storage.mem0_storage.Mem0Storage')
@patch('crewai.memory.storage.mem0_storage.MemoryClient')
def test_user_memory_provider_selection(mock_memory_client, mock_mem0_storage):
    """Test that UserMemory selects the correct storage provider based on config."""
    # Setup - Mock Mem0Storage to avoid API key requirement
    mock_mem0_storage.return_value = mock_mem0_storage
    
    # Test with mem0 provider
    with patch('crewai.memory.user.user_memory.RAGStorage'):
        # Create UserMemory with mem0 provider
        memory_config = {"provider": "mem0"}
        user_memory = UserMemory(memory_config=memory_config)
        
        # Verify Mem0Storage was used
        mock_mem0_storage.assert_called_once()
    
    # Reset mocks
    mock_mem0_storage.reset_mock()
    
    # Test with default provider (RAGStorage)
    with patch('crewai.memory.user.user_memory.RAGStorage', return_value=mock_mem0_storage) as mock_rag:
        # Create UserMemory with no provider specified
        user_memory = UserMemory()
        
        # Verify RAGStorage was used
        mock_rag.assert_called_once()


@patch('crewai.memory.user.user_memory.UserMemory._memory_provider', new_callable=PropertyMock)
def test_user_memory_save_formatting(mock_memory_provider):
    """Test that UserMemory formats data correctly based on provider."""
    # Test with mem0 provider
    mock_memory_provider.return_value = "mem0"
    with patch('crewai.memory.memory.Memory.save') as mock_save:
        user_memory = UserMemory()
        user_memory.save("test data")
        
        # Verify data was formatted for mem0
        args, _ = mock_save.call_args
        assert "Remember the details about the user: test data" in args[0]
    
    # Test with RAG provider
    mock_memory_provider.return_value = None
    with patch('crewai.memory.memory.Memory.save') as mock_save:
        user_memory = UserMemory()
        user_memory.save("test data")
        
        # Verify data was not formatted
        args, _ = mock_save.call_args
        assert args[0] == "test data"
