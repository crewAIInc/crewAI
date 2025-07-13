import pytest
import tempfile
from pathlib import Path

from crewai.tasks.chunk_based_task import ChunkBasedTask
from crewai.agent import Agent
from crewai.crew import Crew


class TestChunkBasedTaskIntegration:
    
    def test_end_to_end_chunk_processing(self):
        """Test complete chunk-based processing workflow."""
        test_content = """
        This is the first section of a large document that needs to be analyzed.
        It contains important information about the topic at hand.
        
        This is the second section that builds upon the first section.
        It provides additional context and details that are relevant.
        
        This is the third section that concludes the document.
        It summarizes the key points and provides final insights.
        """
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(test_content)
            temp_path = f.name
        
        try:
            agent = Agent(
                role="Document Analyzer",
                goal="Analyze document content thoroughly",
                backstory="Expert at analyzing and summarizing documents"
            )
            
            task = ChunkBasedTask(
                description="Analyze this document and identify key themes",
                expected_output="A comprehensive analysis of the document's main themes",
                file_path=temp_path,
                chunk_size=200,
                chunk_overlap=50,
                agent=agent
            )
            
            crew = Crew(
                agents=[agent],
                tasks=[task],
                memory=True
            )
            
            assert task.file_path == Path(temp_path)
            assert task.chunk_size == 200
            assert task.chunk_overlap == 50
            
        finally:
            Path(temp_path).unlink()

    def test_chunk_based_task_with_crew_structure(self):
        """Test that ChunkBasedTask integrates properly with Crew structure."""
        test_content = "Sample content for testing crew integration with chunk-based tasks."
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            f.write(test_content)
            temp_path = f.name
        
        try:
            analyzer_agent = Agent(
                role="Content Analyzer",
                goal="Analyze content effectively",
                backstory="Experienced content analyst"
            )
            
            chunk_task = ChunkBasedTask(
                description="Analyze the content for key insights",
                expected_output="Summary of key insights found",
                file_path=temp_path,
                chunk_size=50,
                chunk_overlap=10,
                agent=analyzer_agent
            )
            
            crew = Crew(
                agents=[analyzer_agent],
                tasks=[chunk_task]
            )
            
            assert len(crew.tasks) == 1
            assert isinstance(crew.tasks[0], ChunkBasedTask)
            assert crew.tasks[0].file_path == Path(temp_path)
            
        finally:
            Path(temp_path).unlink()
