from unittest.mock import patch, MagicMock

import pytest

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.memory.memory import Memory, MemoryOperationError
from crewai.memory.short_term.short_term_memory import ShortTermMemory
from crewai.memory.short_term.short_term_memory_item import ShortTermMemoryItem
from crewai.task import Task
from crewai.utilities.logger import Logger


def test_memory_verbose_flag_in_crew():
    """Test that memory_verbose flag is correctly set in Crew"""
    agent = Agent(
        role="Researcher",
        goal="Research goal",
        backstory="Researcher backstory",
    )
    task = Task(
        description="Test task",
        expected_output="Test output",
        agent=agent,
    )
    crew = Crew(agents=[agent], tasks=[task], memory=True, memory_verbose=True)
    assert crew.memory_verbose is True


def test_memory_verbose_logging_in_memory():
    """Test that memory operations are logged when memory_verbose is enabled"""
    storage = MagicMock()
    
    mock_logger = MagicMock(spec=Logger)
    
    memory = Memory(storage=storage, memory_verbose=True)
    
    memory._logger = mock_logger
    
    memory.save("test value", {"test": "metadata"}, "test_agent")
    mock_logger.log.assert_called_once()
    args = mock_logger.log.call_args[0]
    assert args[0] == "info"
    assert "Saving" in args[1]
    
    mock_logger.log.reset_mock()
    memory.search("test query")
    assert mock_logger.log.call_count == 2
    first_call_args = mock_logger.log.call_args_list[0][0]
    assert first_call_args[0] == "info"
    assert "Searching" in first_call_args[1]
    second_call_args = mock_logger.log.call_args_list[1][0]
    assert "Found" in second_call_args[1]


def test_no_logging_when_memory_verbose_disabled():
    """Test that no logging occurs when memory_verbose is disabled"""
    storage = MagicMock()
    
    mock_logger = MagicMock(spec=Logger)
    
    memory = Memory(storage=storage, memory_verbose=False)
    
    memory._logger = mock_logger
    
    memory.save("test value", {"test": "metadata"}, "test_agent")
    mock_logger.log.assert_not_called()
    
    memory.search("test query")
    mock_logger.log.assert_not_called()


def test_memory_verbose_in_short_term_memory():
    """Test that memory_verbose flag is correctly passed to ShortTermMemory"""
    with patch('crewai.memory.short_term.short_term_memory.RAGStorage') as mock_storage_class:
        mock_storage = MagicMock()
        mock_storage_class.return_value = mock_storage
        
        memory = ShortTermMemory(memory_verbose=True)
        assert memory.memory_verbose is True
        
        mock_logger = MagicMock()
        memory._logger = mock_logger
        
        memory.save("test value", {"test": "metadata"}, "test_agent")
        assert mock_logger.log.call_count >= 1


def test_memory_verbose_passed_from_crew_to_memory():
    """Test that memory_verbose flag is correctly passed from Crew to memory instances"""
    with patch('crewai.crew.LongTermMemory') as mock_ltm, \
         patch('crewai.crew.ShortTermMemory') as mock_stm, \
         patch('crewai.crew.EntityMemory') as mock_em, \
         patch('crewai.crew.UserMemory') as mock_um:
        
        mock_ltm_instance = MagicMock()
        mock_stm_instance = MagicMock()
        mock_em_instance = MagicMock()
        mock_um_instance = MagicMock()
        
        mock_ltm.return_value = mock_ltm_instance
        mock_stm.return_value = mock_stm_instance
        mock_em.return_value = mock_em_instance
        mock_um.return_value = mock_um_instance
        
        agent = Agent(
            role="Researcher",
            goal="Research goal",
            backstory="Researcher backstory",
        )
        task = Task(
            description="Test task",
            expected_output="Test output",
            agent=agent,
        )
        
        crew = Crew(agents=[agent], tasks=[task], memory=True, memory_verbose=True, memory_config={})
        
        mock_ltm.assert_called_once_with(memory_verbose=True)
        mock_stm.assert_called_with(crew=crew, embedder_config=None, memory_verbose=True)
        mock_em.assert_called_with(crew=crew, embedder_config=None, memory_verbose=True)
        mock_um.assert_called_with(crew=crew, memory_verbose=True)


def test_memory_verbose_error_handling():
    """Test that memory operations errors are properly handled when memory_verbose is enabled"""
    storage = MagicMock()
    storage.save.side_effect = Exception("Test error")
    storage.search.side_effect = Exception("Test error")
    
    mock_logger = MagicMock()
    
    with patch('crewai.memory.memory.Logger', return_value=mock_logger):
        memory = Memory(storage=storage, memory_verbose=True)
        
        with pytest.raises(MemoryOperationError) as exc_info:
            memory.save("test value", {"test": "metadata"}, "test_agent")
        
        assert "save" in str(exc_info.value)
        assert "Test error" in str(exc_info.value)
        assert "Memory" in str(exc_info.value)
        
        with pytest.raises(MemoryOperationError) as exc_info:
            memory.search("test query")
        
        assert "search" in str(exc_info.value)
        assert "Test error" in str(exc_info.value)
