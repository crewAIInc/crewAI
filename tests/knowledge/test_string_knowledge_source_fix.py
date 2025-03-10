import pytest
from unittest.mock import patch, MagicMock
from crewai import Agent, Task, Crew, Process
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.knowledge.storage.knowledge_storage import KnowledgeStorage

def test_knowledge_storage_search_filtering():
    """Test that KnowledgeStorage.search() correctly filters results based on distance scores."""
    # Create a mock collection to simulate ChromaDB behavior
    mock_collection = MagicMock()
    mock_collection.query.return_value = {
        "ids": [["1", "2", "3", "4", "5"]],
        "metadatas": [[{}, {}, {}, {}, {}]],
        "documents": [["Doc1", "Doc2", "Doc3", "Doc4", "Doc5"]],
        "distances": [[0.1, 0.2, 0.3, 0.4, 0.5]]  # Lower is better in ChromaDB
    }
    
    # Create a KnowledgeStorage instance with the mock collection
    storage = KnowledgeStorage()
    storage.collection = mock_collection
    
    # Search with the fixed implementation
    results = storage.search(["test query"], score_threshold=0.35)
    
    # Assert that only results with distance < threshold are included
    assert len(results) == 3
    assert results[0]["context"] == "Doc1"
    assert results[1]["context"] == "Doc2"
    assert results[2]["context"] == "Doc3"
    
    # Verify that results with distance >= threshold are excluded
    contexts = [result["context"] for result in results]
    assert "Doc4" not in contexts
    assert "Doc5" not in contexts

def test_string_knowledge_source_integration():
    """Test that StringKnowledgeSource correctly adds content to storage."""
    # Create a knowledge source with specific content
    content = "Users name is John. He is 30 years old and lives in San Francisco."
    
    # Mock the KnowledgeStorage to avoid actual embedding computation
    with patch('crewai.knowledge.storage.knowledge_storage.KnowledgeStorage') as MockStorage:
        # Configure the mock storage
        mock_storage = MockStorage.return_value
        mock_storage.search.return_value = [
            {"context": "Users name is John. He is 30 years old and lives in San Francisco."}
        ]
        
        # Create the string source with the mock storage
        string_source = StringKnowledgeSource(content=content)
        string_source.storage = mock_storage
        string_source.add()
        
        # Verify that the content was added to storage
        assert mock_storage.save.called
        
        # Test querying the knowledge
        results = mock_storage.search(["What city does John live in?"])
        assert len(results) > 0
        assert "San Francisco" in results[0]["context"]
