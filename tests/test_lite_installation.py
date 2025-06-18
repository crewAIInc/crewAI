"""Test that crewAI lite installation works with minimal dependencies."""

import pytest
import subprocess
import sys
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock


def test_core_imports_work_without_optional_deps():
    """Test that core crewAI functionality can be imported without optional dependencies."""
    
    try:
        from crewai import Agent, Crew, Task, LLM
        from crewai.lite_agent import LiteAgent
        from crewai.process import Process
        assert Agent is not None
        assert Crew is not None
        assert Task is not None
        assert LLM is not None
        assert LiteAgent is not None
        assert Process is not None
    except ImportError as e:
        pytest.fail(f"Core imports should work without optional dependencies: {e}")


def test_optional_memory_import_error():
    """Test that memory functionality raises helpful error without chromadb."""
    with patch.dict('sys.modules', {'chromadb': None}):
        with patch('crewai.memory.storage.rag_storage.CHROMADB_AVAILABLE', False):
            from crewai.memory.storage.rag_storage import RAGStorage
            
            with pytest.raises(ImportError) as exc_info:
                RAGStorage("test")
            
            assert "ChromaDB is required" in str(exc_info.value)
            assert "crewai[memory]" in str(exc_info.value)


def test_optional_knowledge_import_error():
    """Test that knowledge functionality raises helpful error without dependencies."""
    with patch.dict('sys.modules', {'chromadb': None}):
        with patch('crewai.knowledge.storage.knowledge_storage.CHROMADB_AVAILABLE', False):
            from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
            
            with pytest.raises(ImportError) as exc_info:
                KnowledgeStorage()
            
            assert "ChromaDB is required" in str(exc_info.value)
            assert "crewai[knowledge]" in str(exc_info.value)


def test_optional_pdf_import_error():
    """Test that PDF knowledge source raises helpful error without pdfplumber."""
    with patch.dict('sys.modules', {'pdfplumber': None}):
        with patch('crewai.knowledge.source.pdf_knowledge_source.PDFPLUMBER_AVAILABLE', False):
            from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
            
            knowledge_dir = Path("knowledge/tmp")
            knowledge_dir.mkdir(parents=True, exist_ok=True)
            
            test_file = knowledge_dir / "test.pdf"
            test_file.touch()
            
            try:
                with pytest.raises(ImportError) as exc_info:
                    PDFKnowledgeSource(file_paths=["tmp/test.pdf"])
                
                assert "pdfplumber is required" in str(exc_info.value)
                assert "crewai[knowledge]" in str(exc_info.value)
            finally:
                if test_file.exists():
                    test_file.unlink()
                if knowledge_dir.exists() and not any(knowledge_dir.iterdir()):
                    knowledge_dir.rmdir()


def test_optional_visualization_import_error():
    """Test that flow visualization raises helpful error without pyvis."""
    with patch.dict('sys.modules', {'pyvis': None}):
        with patch('crewai.flow.flow_visualizer.PYVIS_AVAILABLE', False):
            from crewai.flow.flow_visualizer import plot_flow
            
            mock_flow = Mock()
            
            with pytest.raises(ImportError) as exc_info:
                plot_flow(mock_flow)
            
            assert "Pyvis is required" in str(exc_info.value)
            assert "crewai[visualization]" in str(exc_info.value)


def test_telemetry_disabled_without_opentelemetry():
    """Test that telemetry is disabled gracefully without opentelemetry."""
    from crewai.telemetry.telemetry import Telemetry
    
    telemetry = Telemetry()
    
    assert isinstance(telemetry.ready, bool)
    assert isinstance(telemetry._is_telemetry_disabled(), bool)


def test_lite_agent_works_without_optional_deps():
    """Test that LiteAgent can be created and used without optional dependencies."""
    from crewai.lite_agent import LiteAgent
    from crewai import LLM
    from unittest.mock import Mock
    
    mock_llm = Mock(spec=LLM)
    mock_llm.call.return_value = "Test response"
    mock_llm.model = "test-model"
    
    agent = LiteAgent(
        role="Test Agent",
        goal="Test Goal", 
        backstory="Test Backstory",
        llm=mock_llm,
        verbose=False
    )
    
    assert agent.role == "Test Agent"
    assert agent.goal == "Test Goal"
    assert agent.backstory == "Test Backstory"


def test_basic_crew_creation_without_optional_deps():
    """Test that basic Crew can be created without optional dependencies."""
    from crewai import Agent, Crew, Task, LLM
    from unittest.mock import Mock
    
    mock_llm = Mock(spec=LLM)
    mock_llm.call.return_value = "Test response"
    mock_llm.model = "test-model"
    
    agent = Agent(
        role="Test Agent",
        goal="Test Goal",
        backstory="Test Backstory", 
        llm=mock_llm,
        verbose=False
    )
    
    task = Task(
        description="Test task",
        agent=agent,
        expected_output="Test output"
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False
    )
    
    assert crew.agents[0].role == "Test Agent"
    assert crew.tasks[0].description == "Test task"


def test_core_functionality_without_optional_deps():
    """Test that core crewAI functionality works without optional dependencies."""
    from crewai import Agent, Task, Crew, LLM
    from unittest.mock import Mock
    
    mock_llm = Mock(spec=LLM)
    mock_llm.call.return_value = "Test response"
    mock_llm.model = "test-model"
    
    agent = Agent(
        role="Test Agent",
        goal="Test Goal",
        backstory="Test Backstory",
        llm=mock_llm,
        verbose=False
    )
    
    task = Task(
        description="Test task description",
        agent=agent,
        expected_output="Test expected output"
    )
    
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False
    )
    
    assert agent.role == "Test Agent"
    assert task.description == "Test task description"
    assert len(crew.agents) == 1
    assert len(crew.tasks) == 1
