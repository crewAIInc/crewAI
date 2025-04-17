from unittest.mock import MagicMock, patch

import pytest

from crewai.memory.contextual.contextual_memory import ContextualMemory


def test_contextual_memory_with_mem0_provider():
    """Test that contextual memory properly handles Mem0 results."""
    stm_mock = MagicMock()
    ltm_mock = MagicMock()
    em_mock = MagicMock()
    um_mock = MagicMock()
    exm_mock = MagicMock()  # External memory mock
    
    mock_result = {
        'id': 'test-id',
        'metadata': 'some metadata',
        'context': 'test context data',
        'score': 0.95
    }
    
    stm_mock.search.return_value = [mock_result]
    em_mock.search.return_value = [mock_result]
    um_mock.search.return_value = [{'memory': 'user memory'}]  # User memory has different structure
    exm_mock.search.return_value = [{'memory': 'external memory'}]  # External memory structure
    
    memory_config = {"provider": "mem0"}
    context_memory = ContextualMemory(
        memory_config=memory_config,
        stm=stm_mock,
        ltm=ltm_mock,
        em=em_mock,
        um=um_mock,
        exm=exm_mock
    )
    
    result = context_memory._fetch_stm_context("test query")
    
    stm_mock.search.assert_called_once_with("test query")
    
    assert "test context data" in result
    
    entity_result = context_memory._fetch_entity_context("test query")
    em_mock.search.assert_called_once_with("test query")
    assert "test context data" in entity_result
