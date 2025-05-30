import pytest
import importlib
import sys
from unittest.mock import patch


def test_import_without_chromadb():
    """Test that crewai can be imported without chromadb."""
    with patch.dict(sys.modules, {"chromadb": None}):
        if "crewai.memory.storage.rag_storage" in sys.modules:
            importlib.reload(sys.modules["crewai.memory.storage.rag_storage"])
        if "crewai.knowledge.storage.knowledge_storage" in sys.modules:
            importlib.reload(sys.modules["crewai.knowledge.storage.knowledge_storage"])
        if "crewai.utilities.embedding_configurator" in sys.modules:
            importlib.reload(sys.modules["crewai.utilities.embedding_configurator"])
            
        import crewai
        
        from crewai import Agent, Task, Crew, Process
        
        agent = Agent(role="Test Agent", goal="Test Goal", backstory="Test Backstory")
        task = Task(description="Test Task", agent=agent)
        crew = Crew(agents=[agent], tasks=[task], process=Process.sequential)


def test_memory_storage_without_chromadb():
    """Test that memory storage raises appropriate error when chromadb is not available."""
    with patch.dict(sys.modules, {"chromadb": None}):
        if "crewai.memory.storage.rag_storage" in sys.modules:
            importlib.reload(sys.modules["crewai.memory.storage.rag_storage"])
            
        from crewai.memory.storage.rag_storage import RAGStorage, HAS_CHROMADB
        
        assert not HAS_CHROMADB
        
        with pytest.raises(ImportError) as excinfo:
            storage = RAGStorage()
            storage._initialize_app()
            
        assert "ChromaDB is required for memory storage features" in str(excinfo.value)


def test_knowledge_storage_without_chromadb():
    """Test that knowledge storage raises appropriate error when chromadb is not available."""
    with patch.dict(sys.modules, {"chromadb": None}):
        if "crewai.knowledge.storage.knowledge_storage" in sys.modules:
            importlib.reload(sys.modules["crewai.knowledge.storage.knowledge_storage"])
            
        from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage, HAS_CHROMADB
        
        assert not HAS_CHROMADB
        
        with pytest.raises(ImportError) as excinfo:
            storage = KnowledgeStorage()
            storage.initialize_knowledge_storage()
            
        assert "ChromaDB is required for knowledge storage features" in str(excinfo.value)


def test_embedding_configurator_without_chromadb():
    """Test that embedding configurator raises appropriate error when chromadb is not available."""
    with patch.dict(sys.modules, {"chromadb": None}):
        if "crewai.utilities.embedding_configurator" in sys.modules:
            importlib.reload(sys.modules["crewai.utilities.embedding_configurator"])
            
        from crewai.utilities.embedding_configurator import EmbeddingConfigurator, HAS_CHROMADB
        
        assert not HAS_CHROMADB
        
        with pytest.raises(ImportError) as excinfo:
            configurator = EmbeddingConfigurator()
            configurator.configure_embedder()
            
        assert "ChromaDB is required for embedding functionality" in str(excinfo.value)
