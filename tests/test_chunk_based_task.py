import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from crewai.tasks.chunk_based_task import ChunkBasedTask
from crewai.agent import Agent
from crewai.tasks.task_output import TaskOutput


class TestChunkBasedTask:
    
    def test_chunk_based_task_creation(self):
        """Test creating a ChunkBasedTask with valid parameters."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("Test content for chunking")
            temp_path = f.name
        
        try:
            task = ChunkBasedTask(
                description="Analyze the content",
                expected_output="Analysis summary",
                file_path=temp_path,
                chunk_size=100,
                chunk_overlap=20
            )
            assert task.file_path == Path(temp_path)
            assert task.chunk_size == 100
            assert task.chunk_overlap == 20
        finally:
            Path(temp_path).unlink()
    
    def test_file_path_validation(self):
        """Test file path validation."""
        with pytest.raises(ValueError, match="File not found"):
            ChunkBasedTask(
                description="Test",
                expected_output="Test output",
                file_path="/nonexistent/file.txt"
            )
    
    def test_chunk_text_method(self):
        """Test text chunking functionality."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("dummy content")
            temp_path = f.name
        
        try:
            task = ChunkBasedTask(
                description="Test",
                expected_output="Test output", 
                file_path=temp_path,
                chunk_size=10,
                chunk_overlap=2
            )
            
            text = "This is a test text that should be chunked properly"
            chunks = task._chunk_text(text)
            
            assert len(chunks) > 1
            assert all(len(chunk) <= 10 for chunk in chunks)
            assert chunks[1].startswith(chunks[0][-2:])
        finally:
            Path(temp_path).unlink()
    
    def test_empty_file_handling(self):
        """Test handling of empty files."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("")
            temp_path = f.name
        
        try:
            task = ChunkBasedTask(
                description="Test",
                expected_output="Test output",
                file_path=temp_path
            )
            
            chunks = task._chunk_text("")
            assert chunks == []
        finally:
            Path(temp_path).unlink()
    
    @patch('crewai.task.Task._execute_core')
    def test_chunk_processing_execution(self, mock_execute):
        """Test the sequential chunk processing execution."""
        test_content = "A" * 100 + "B" * 100 + "C" * 100
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(test_content)
            temp_path = f.name
        
        try:
            mock_result = TaskOutput(
                description="Test",
                expected_output="Test output",
                raw="Chunk analysis result",
                agent="test_agent"
            )
            mock_execute.return_value = mock_result
            
            task = ChunkBasedTask(
                description="Analyze content",
                expected_output="Analysis summary",
                file_path=temp_path,
                chunk_size=80,
                chunk_overlap=10
            )
            
            mock_agent = Mock()
            mock_agent.role = "test_agent"
            mock_agent.crew = None
            
            result = task._execute_core(mock_agent, None, None)
            
            assert len(task.chunk_results) > 1
            assert mock_execute.call_count > len(task.chunk_results)
            
        finally:
            Path(temp_path).unlink()
    
    def test_memory_integration(self):
        """Test integration with agent memory system."""
        test_content = "Test content for memory integration"
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(test_content)
            temp_path = f.name
        
        try:
            task = ChunkBasedTask(
                description="Test memory integration",
                expected_output="Memory test output",
                file_path=temp_path
            )
            
            mock_agent = Mock()
            mock_agent.role = "test_agent"
            mock_crew = Mock()
            mock_memory = Mock()
            mock_crew._short_term_memory = mock_memory
            mock_agent.crew = mock_crew
            
            with patch.object(task, '_aggregate_results') as mock_aggregate:
                mock_aggregate.return_value = TaskOutput(
                    description="Test",
                    expected_output="Test output", 
                    raw="Final result",
                    agent="test_agent"
                )
                
                with patch('crewai.task.Task._execute_core') as mock_execute:
                    mock_execute.return_value = TaskOutput(
                        description="Test",
                        expected_output="Test output",
                        raw="Chunk result",
                        agent="test_agent"
                    )
                    
                    task._execute_core(mock_agent, None, None)
                    
                    mock_memory.save.assert_called()
                    
        finally:
            Path(temp_path).unlink()

    def test_get_chunk_results(self):
        """Test accessing individual chunk results."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("test content")
            temp_path = f.name
        
        try:
            task = ChunkBasedTask(
                description="Test",
                expected_output="Test output",
                file_path=temp_path
            )
            
            mock_result = TaskOutput(
                description="Test",
                expected_output="Test output",
                raw="Test result",
                agent="test_agent"
            )
            task.chunk_results = [mock_result]
            
            results = task.get_chunk_results()
            assert len(results) == 1
            assert results[0] == mock_result
            
        finally:
            Path(temp_path).unlink()

    def test_custom_aggregation_prompt(self):
        """Test using custom aggregation prompt."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write("test content")
            temp_path = f.name
        
        try:
            custom_prompt = "Custom aggregation instructions"
            task = ChunkBasedTask(
                description="Test",
                expected_output="Test output",
                file_path=temp_path,
                aggregation_prompt=custom_prompt
            )
            
            assert task.aggregation_prompt == custom_prompt
            
        finally:
            Path(temp_path).unlink()
