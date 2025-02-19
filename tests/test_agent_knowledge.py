from unittest.mock import MagicMock, patch
import pytest

from chromadb.api.types import EmbeddingFunction

from crewai import Agent, Crew, Task
from crewai.knowledge.source.string_knowledge_source import StringKnowledgeSource
from crewai.process import Process

class MockEmbeddingFunction(EmbeddingFunction):
    def __call__(self, texts):
        return [[0.0] * 1536 for _ in texts]

@pytest.fixture(autouse=True)
def mock_vector_db():
    """Mock vector database operations."""
    with patch("crewai.knowledge.storage.knowledge_storage.KnowledgeStorage") as mock, \
         patch("chromadb.PersistentClient") as mock_chroma:
        # Mock ChromaDB client and collection
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["1"]],
            "distances": [[0.1]],
            "metadatas": [[{"source": "test"}]],
            "documents": [["Test content"]]
        }
        mock_chroma.return_value.get_or_create_collection.return_value = mock_collection
        
        # Mock the query method to return a predefined response
        instance = mock.return_value
        instance.query.return_value = [
            {
                "context": "Test content",
                "score": 0.9,
            }
        ]
        instance.reset.return_value = None
        yield instance

def test_agent_invalid_embedder_config():
    """Test that an invalid embedder configuration raises a ValueError."""
    with pytest.raises(ValueError, match="embedder_config must be a dictionary"):
        Agent(
            role="test role",
            goal="test goal",
            backstory="test backstory",
            knowledge_sources=[StringKnowledgeSource(content="test content")],
            embedder_config="invalid"
        )

    with pytest.raises(ValueError, match="embedder_config must contain 'provider' key"):
        Agent(
            role="test role",
            goal="test goal",
            backstory="test backstory",
            knowledge_sources=[StringKnowledgeSource(content="test content")],
            embedder_config={"invalid": "config"}
        )

    with pytest.raises(ValueError, match="embedder_config must contain 'config' key"):
        Agent(
            role="test role",
            goal="test goal",
            backstory="test backstory",
            knowledge_sources=[StringKnowledgeSource(content="test content")],
            embedder_config={"provider": "custom"}
        )

    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        Agent(
            role="test role",
            goal="test goal",
            backstory="test backstory",
            knowledge_sources=[StringKnowledgeSource(content="test content")],
            embedder_config={"provider": "invalid", "config": {}}
        )

def test_agent_knowledge_with_custom_embedder(mock_vector_db):
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        knowledge_sources=[StringKnowledgeSource(content="test content")],
        embedder_config={
            "provider": "custom",
            "config": {
                "embedder": MockEmbeddingFunction()
            }
        }
    )
    assert agent.knowledge is not None
    assert agent.knowledge.storage.embedder is not None

def test_agent_inherits_crew_embedder(mock_vector_db):
    test_agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory"
    )
    test_task = Task(
        description="test task",
        expected_output="test output",
        agent=test_agent
    )
    crew = Crew(
        agents=[test_agent],
        tasks=[test_task],
        process=Process.sequential,
        embedder_config={
            "provider": "custom",
            "config": {
                "embedder": MockEmbeddingFunction()
            }
        }
    )
    agent = Agent(
        role="test role",
        goal="test goal",
        backstory="test backstory",
        knowledge_sources=[StringKnowledgeSource(content="test content")],
        crew=crew
    )
    assert agent.knowledge is not None
    assert agent.knowledge.storage.embedder is not None

def test_agent_knowledge_without_embedder_raises_error(mock_vector_db):
    with pytest.raises(ValueError, match="No embedder configuration provided"):
        agent = Agent(
            role="test role",
            goal="test goal",
            backstory="test backstory",
            knowledge_sources=[StringKnowledgeSource(content="test content")]
        )
