import unittest
from unittest.mock import patch, MagicMock
import sys
import pytest
from typing import Any, Dict, List, Optional


class TestOptionalChromadb(unittest.TestCase):
    def test_rag_storage_import_error(self):
        """Test that RAGStorage raises an ImportError when chromadb is not installed."""
        with patch.dict(sys.modules, {"chromadb": None}):
            with pytest.raises(ImportError) as excinfo:
                from crewai.memory.storage.rag_storage import RAGStorage
                storage = RAGStorage(type="test")
            
            assert "ChromaDB is not installed" in str(excinfo.value)

    def test_knowledge_storage_import_error(self):
        """Test that KnowledgeStorage raises an ImportError when chromadb is not installed."""
        with patch.dict(sys.modules, {"chromadb": None}):
            with pytest.raises(ImportError) as excinfo:
                from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage
                storage = KnowledgeStorage()
            
            assert "ChromaDB is not installed" in str(excinfo.value)
