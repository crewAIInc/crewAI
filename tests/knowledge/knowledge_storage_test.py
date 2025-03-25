import os
from unittest.mock import MagicMock, patch

import pytest

from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage


class TestKnowledgeStorage:
    @patch("crewai.knowledge.storage.knowledge_storage.chromadb")
    @patch("crewai.knowledge.storage.knowledge_storage.shutil")
    def test_reset_with_default_collection(self, mock_shutil, mock_chromadb):
        # Setup
        mock_app = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_app
        
        # Execute
        storage = KnowledgeStorage()
        storage.reset()
        
        # Verify
        mock_app.reset.assert_called_once()
        mock_shutil.rmtree.assert_called_once()
        
    @patch("crewai.knowledge.storage.knowledge_storage.chromadb")
    @patch("crewai.knowledge.storage.knowledge_storage.shutil")
    def test_reset_with_custom_collection(self, mock_shutil, mock_chromadb):
        # Setup
        mock_app = MagicMock()
        mock_chromadb.PersistentClient.return_value = mock_app
        
        # Execute
        storage = KnowledgeStorage(collection_name="custom_collection")
        storage.reset()
        
        # Verify
        mock_app.delete_collection.assert_called_once_with(name="knowledge_custom_collection")
        mock_app.reset.assert_not_called()
        mock_shutil.rmtree.assert_not_called()
        
    @patch("crewai.knowledge.storage.knowledge_storage.chromadb")
    @patch("crewai.knowledge.storage.knowledge_storage.shutil")
    def test_reset_with_custom_collection_fallback(self, mock_shutil, mock_chromadb):
        # Setup
        mock_app = MagicMock()
        mock_app.delete_collection.side_effect = Exception("Collection not found")
        mock_chromadb.PersistentClient.return_value = mock_app
        
        # Execute
        storage = KnowledgeStorage(collection_name="custom_collection")
        storage.reset()
        
        # Verify
        mock_app.delete_collection.assert_called_once_with(name="knowledge_custom_collection")
        mock_app.reset.assert_called_once()
        mock_shutil.rmtree.assert_called_once()
