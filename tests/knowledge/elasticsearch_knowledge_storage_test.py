"""Test Elasticsearch knowledge storage functionality."""

import os
import unittest
from unittest.mock import MagicMock, patch

import pytest

from crewai.knowledge.storage.elasticsearch_knowledge_storage import (
    ElasticsearchKnowledgeStorage,
)


@pytest.mark.skipif(
    os.environ.get("RUN_ELASTICSEARCH_TESTS") != "true",
    reason="Elasticsearch tests require RUN_ELASTICSEARCH_TESTS=true"
)
class TestElasticsearchKnowledgeStorage(unittest.TestCase):
    """Test Elasticsearch knowledge storage functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.es_mock = MagicMock()
        self.es_mock.indices.exists.return_value = False
        
        self.embedder_mock = MagicMock()
        self.embedder_mock.embed_documents.return_value = [[0.1, 0.2, 0.3]]
        
        self.es_patcher = patch(
            "crewai.knowledge.storage.elasticsearch_knowledge_storage.Elasticsearch",
            return_value=self.es_mock
        )
        self.es_class_mock = self.es_patcher.start()
        
        self.storage = ElasticsearchKnowledgeStorage(
            embedder_config=self.embedder_mock,
            collection_name="test"
        )
        self.storage.initialize_knowledge_storage()
        
    def tearDown(self):
        """Tear down test fixtures."""
        self.es_patcher.stop()
        
    def test_initialization(self):
        """Test initialization of Elasticsearch knowledge storage."""
        self.es_class_mock.assert_called_once()
        
        self.es_mock.indices.create.assert_called_once()
        
    def test_save(self):
        """Test saving to Elasticsearch knowledge storage."""
        self.storage.save(["Test document 1", "Test document 2"], {"source": "test"})
        
        self.assertEqual(self.es_mock.index.call_count, 2)
        
        self.assertEqual(self.embedder_mock.embed_documents.call_count, 2)
        
    def test_search(self):
        """Test searching in Elasticsearch knowledge storage."""
        self.es_mock.search.return_value = {
            "hits": {
                "hits": [
                    {
                        "_id": "test_id",
                        "_score": 1.5,  # Score between 1-2 (Elasticsearch range)
                        "_source": {
                            "text": "Test document",
                            "metadata": {"source": "test"},
                        }
                    }
                ]
            }
        }
        
        results = self.storage.search(["test query"])
        
        self.es_mock.search.assert_called_once()
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "test_id")
        self.assertEqual(results[0]["context"], "Test document")
        self.assertEqual(results[0]["metadata"], {"source": "test"})
        self.assertEqual(results[0]["score"], 0.5)  # Adjusted to 0-1 range
        
    def test_reset(self):
        """Test resetting Elasticsearch knowledge storage."""
        self.es_mock.indices.exists.return_value = True
        
        self.storage.reset()
        
        self.es_mock.indices.delete.assert_called_once()
        
        self.assertEqual(self.es_mock.indices.create.call_count, 2)
