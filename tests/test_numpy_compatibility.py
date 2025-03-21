import importlib
import pytest
import sys
import warnings

def test_crew_import_with_numpy():
    """Test that crewai can be imported even with NumPy compatibility issues."""
    try:
        # Force reload to ensure we test our fix
        if "crewai" in sys.modules:
            importlib.reload(sys.modules["crewai"])
        
        # This should not raise an exception
        from crewai import Crew
        assert Crew is not None
    except Exception as e:
        pytest.fail(f"Failed to import Crew: {e}")

def test_embedding_configurator_with_numpy():
    """Test that EmbeddingConfigurator can be imported with NumPy."""
    try:
        # Force reload
        if "crewai.utilities.embedding_configurator" in sys.modules:
            importlib.reload(sys.modules["crewai.utilities.embedding_configurator"])
        
        from crewai.utilities.embedding_configurator import EmbeddingConfigurator
        configurator = EmbeddingConfigurator()
        # Test that we can create an embedder (might be unavailable but shouldn't crash)
        embedder = configurator.configure_embedder()
        assert embedder is not None
    except Exception as e:
        pytest.fail(f"Failed to use EmbeddingConfigurator: {e}")

def test_rag_storage_with_numpy():
    """Test that RAGStorage can be imported and used with NumPy."""
    try:
        # Force reload
        if "crewai.memory.storage.rag_storage" in sys.modules:
            importlib.reload(sys.modules["crewai.memory.storage.rag_storage"])
        
        from crewai.memory.storage.rag_storage import RAGStorage
        # Initialize with minimal config to avoid actual DB operations
        storage = RAGStorage(type="test", crew=None)
        # Just verify we can create the object without errors
        assert storage is not None
    except Exception as e:
        pytest.fail(f"Failed to use RAGStorage: {e}")

def test_knowledge_storage_with_numpy():
    """Test that KnowledgeStorage can be imported and used with NumPy."""
    try:
        # Force reload
        if "crewai.knowledge.storage.knowledge_storage" in sys.modules:
            importlib.reload(sys.modules["crewai.knowledge.storage.knowledge_storage"])
        
        from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
        # Initialize with minimal config
        storage = KnowledgeStorage()
        # Just verify we can create the object without errors
        assert storage is not None
    except Exception as e:
        pytest.fail(f"Failed to use KnowledgeStorage: {e}")
