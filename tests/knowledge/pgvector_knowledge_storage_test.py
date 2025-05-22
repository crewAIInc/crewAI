import os
import pytest
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock

from crewai.knowledge.storage.pgvector_knowledge_storage import PGVectorKnowledgeStorage

class MockSession:
    def __init__(self):
        self.queries = []
        self.commits = 0
        self.rollbacks = 0
        self.closes = 0
    
    def query(self, *args, **kwargs):
        return self
    
    def filter(self, *args, **kwargs):
        return self
    
    def first(self):
        return None
    
    def add(self, *args, **kwargs):
        pass
    
    def commit(self):
        self.commits += 1
    
    def rollback(self):
        self.rollbacks += 1
    
    def close(self):
        self.closes += 1
    
    def execute(self, *args, **kwargs):
        return self
    
    def fetchall(self):
        return [
            ("doc1", "This is a test document", '{"source": "test"}', 0.9),
            ("doc2", "Another test document", '{"source": "test"}', 0.8),
            ("doc3", "A third test document", '{"source": "test"}', 0.7),
        ]

@pytest.fixture
def mock_embedder():
    return lambda x: [[0.1] * 1536 for _ in range(len(x))]

@pytest.fixture
def mock_session():
    return MockSession()

@pytest.fixture
def mock_session_maker(mock_session):
    def session_maker():
        return mock_session
    return session_maker

@pytest.fixture
def mock_engine():
    return MagicMock()

@pytest.fixture
def pgvector_storage(mock_embedder, mock_session_maker, mock_engine):
    with patch("crewai.knowledge.storage.pgvector_knowledge_storage.create_engine", return_value=mock_engine), \
         patch("crewai.knowledge.storage.pgvector_knowledge_storage.sessionmaker", return_value=mock_session_maker), \
         patch("crewai.knowledge.storage.pgvector_knowledge_storage.Base.metadata.create_all"):
        storage = PGVectorKnowledgeStorage(connection_string="postgresql://test:test@localhost:5432/test")
        storage.embedder = mock_embedder
        return storage

def test_search(pgvector_storage, mock_session):
    results = pgvector_storage.search(["test query"], limit=3, score_threshold=0.5)
    
    assert len(results) == 3
    assert results[0]["id"] == "doc1"
    assert results[0]["context"] == "This is a test document"
    assert results[0]["score"] == 0.9
    
    assert mock_session.closes == 1

def test_save(pgvector_storage, mock_session):
    documents = ["Document 1", "Document 2"]
    metadata = [{"source": "test1"}, {"source": "test2"}]
    
    pgvector_storage.save(documents, metadata)
    
    assert mock_session.commits == 1
    assert mock_session.closes == 1
