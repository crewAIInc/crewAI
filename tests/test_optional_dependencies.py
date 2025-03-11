import importlib
import sys
from unittest import mock
import pytest

def test_rag_storage_without_chromadb():
    # Mock the import to simulate chromadb not being installed
    with mock.patch.dict(sys.modules, {'chromadb': None}):
        # Force reload to ensure our mock takes effect
        if 'crewai.memory.storage.rag_storage' in sys.modules:
            importlib.reload(sys.modules['crewai.memory.storage.rag_storage'])
        
        # Now import and test
        from crewai.memory.storage.rag_storage import RAGStorage
        
        # Should not raise an exception
        storage = RAGStorage(type="test", allow_reset=True)
        
        # Methods should handle the case when chromadb is not available
        assert storage.app is None
        assert storage.collection is None
        
        # These methods should not raise exceptions
        storage.save("test", {})
        results = storage.search("test")
        assert results == []
        storage.reset()

def test_knowledge_storage_without_chromadb():
    # Mock the import to simulate chromadb not being installed
    with mock.patch.dict(sys.modules, {'chromadb': None}):
        # Force reload to ensure our mock takes effect
        if 'crewai.knowledge.storage.knowledge_storage' in sys.modules:
            importlib.reload(sys.modules['crewai.knowledge.storage.knowledge_storage'])
        
        # Now import and test
        from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
        
        # Should not raise an exception
        storage = KnowledgeStorage()
        
        # Methods should handle the case when chromadb is not available
        assert storage.app is None
        assert storage.collection is None
        
        # These methods should not raise exceptions
        storage.initialize_knowledge_storage()
        results = storage.search(["test"])
        assert results == []
        storage.reset()
