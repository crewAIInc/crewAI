"""Tests for RAGStorage custom path functionality."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from crewai.agent import Agent
from crewai.crew import Crew
from crewai.memory.entity.entity_memory import EntityMemory
from crewai.memory.short_term.short_term_memory import ShortTermMemory
from crewai.memory.storage.rag_storage import RAGStorage
from crewai.task import Task


@pytest.fixture
def crew():
    """Fixture to create a simple Crew instance."""
    agent = Agent(
        role="Researcher",
        goal="Search relevant data",
        backstory="You are a researcher.",
        tools=[],
    )
    task = Task(
        description="Perform a search.",
        expected_output="A list of results.",
        agent=agent,
    )
    return Crew(agents=[agent], tasks=[task])


@pytest.fixture
def fake_embedder_config():
    """Fixture to provide a fake embedder config that doesn't hit the network."""
    def fake_embedding_function(texts):
        """Fake embedding function that returns constant vectors."""
        if isinstance(texts, str):
            texts = [texts]
        return [[0.1] * 384 for _ in texts]
    
    return fake_embedding_function


class TestRAGStorageCustomPath:
    """Test suite for RAGStorage custom path functionality."""

    def test_rag_storage_with_custom_path_normalizes_to_absolute(self, crew, tmp_path):
        """Test that RAGStorage normalizes relative paths to absolute paths."""
        relative_path = "relative/test/path"
        
        with patch("crewai.memory.storage.rag_storage.build_embedder") as mock_embedder:
            mock_embedder.return_value = lambda texts: [[0.1] * 384 for _ in texts]
            
            storage = RAGStorage(
                type="short_term",
                embedder_config={"provider": "openai"},
                crew=crew,
                path=relative_path,
            )
            
            assert storage.path is not None
            assert os.path.isabs(storage.path)
            assert storage.path.endswith(relative_path)

    def test_rag_storage_with_empty_string_path_uses_default(self, crew):
        """Test that empty string path falls back to default behavior."""
        with patch("crewai.memory.storage.rag_storage.build_embedder") as mock_embedder:
            mock_embedder.return_value = lambda texts: [[0.1] * 384 for _ in texts]
            
            storage = RAGStorage(
                type="short_term",
                embedder_config={"provider": "openai"},
                crew=crew,
                path="",
            )
            
            assert storage.path is None

    def test_rag_storage_with_whitespace_path_uses_default(self, crew):
        """Test that whitespace-only path falls back to default behavior."""
        with patch("crewai.memory.storage.rag_storage.build_embedder") as mock_embedder:
            mock_embedder.return_value = lambda texts: [[0.1] * 384 for _ in texts]
            
            storage = RAGStorage(
                type="short_term",
                embedder_config={"provider": "openai"},
                crew=crew,
                path="   ",
            )
            
            assert storage.path is None

    def test_rag_storage_with_none_path_uses_default(self, crew):
        """Test that None path uses default behavior."""
        with patch("crewai.memory.storage.rag_storage.build_embedder") as mock_embedder:
            mock_embedder.return_value = lambda texts: [[0.1] * 384 for _ in texts]
            
            storage = RAGStorage(
                type="short_term",
                embedder_config={"provider": "openai"},
                crew=crew,
                path=None,
            )
            
            assert storage.path is None

    def test_rag_storage_sets_persist_directory_when_path_provided(self, crew, tmp_path):
        """Test that RAGStorage sets ChromaDB persist_directory when path is provided."""
        custom_path = str(tmp_path / "custom_storage")
        
        with patch("crewai.memory.storage.rag_storage.build_embedder") as mock_embedder, \
             patch("crewai.memory.storage.rag_storage.create_client") as mock_create_client:
            
            mock_embedder.return_value = lambda texts: [[0.1] * 384 for _ in texts]
            mock_create_client.return_value = MagicMock()
            
            storage = RAGStorage(
                type="short_term",
                embedder_config={"provider": "openai"},
                crew=crew,
                path=custom_path,
            )
            
            assert mock_create_client.called
            config = mock_create_client.call_args[0][0]
            assert config.settings.persist_directory == os.path.abspath(custom_path)

    def test_rag_storage_does_not_override_persist_directory_when_no_path(self, crew):
        """Test that RAGStorage doesn't override persist_directory when path is None."""
        with patch("crewai.memory.storage.rag_storage.build_embedder") as mock_embedder, \
             patch("crewai.memory.storage.rag_storage.create_client") as mock_create_client:
            
            mock_embedder.return_value = lambda texts: [[0.1] * 384 for _ in texts]
            mock_create_client.return_value = MagicMock()
            
            storage = RAGStorage(
                type="short_term",
                embedder_config={"provider": "openai"},
                crew=crew,
                path=None,
            )
            
            assert mock_create_client.called
            config = mock_create_client.call_args[0][0]
            assert "CrewAI" in config.settings.persist_directory or "crewai" in config.settings.persist_directory.lower()


class TestShortTermMemoryCustomPath:
    """Test suite for ShortTermMemory with custom path."""

    def test_short_term_memory_with_custom_path(self, crew, tmp_path):
        """Test that ShortTermMemory accepts and uses custom path."""
        custom_path = str(tmp_path / "short_term_storage")
        
        with patch("crewai.memory.storage.rag_storage.build_embedder") as mock_embedder, \
             patch("crewai.memory.storage.rag_storage.create_client") as mock_create_client:
            
            mock_embedder.return_value = lambda texts: [[0.1] * 384 for _ in texts]
            mock_client = MagicMock()
            mock_create_client.return_value = mock_client
            
            memory = ShortTermMemory(
                crew=crew,
                embedder_config={"provider": "openai"},
                path=custom_path,
            )
            
            assert memory.storage.path == os.path.abspath(custom_path)
            
            assert mock_create_client.called
            config = mock_create_client.call_args[0][0]
            assert config.settings.persist_directory == os.path.abspath(custom_path)

    def test_short_term_memory_default_behavior_unchanged(self, crew):
        """Test that ShortTermMemory default behavior (no path) is unchanged."""
        with patch("crewai.memory.storage.rag_storage.build_embedder") as mock_embedder, \
             patch("crewai.memory.storage.rag_storage.create_client") as mock_create_client:
            
            mock_embedder.return_value = lambda texts: [[0.1] * 384 for _ in texts]
            mock_client = MagicMock()
            mock_create_client.return_value = mock_client
            
            memory = ShortTermMemory(
                crew=crew,
                embedder_config={"provider": "openai"},
            )
            
            assert memory.storage.path is None


class TestEntityMemoryCustomPath:
    """Test suite for EntityMemory with custom path."""

    def test_entity_memory_with_custom_path(self, crew, tmp_path):
        """Test that EntityMemory accepts and uses custom path."""
        custom_path = str(tmp_path / "entity_storage")
        
        with patch("crewai.memory.storage.rag_storage.build_embedder") as mock_embedder, \
             patch("crewai.memory.storage.rag_storage.create_client") as mock_create_client:
            
            mock_embedder.return_value = lambda texts: [[0.1] * 384 for _ in texts]
            mock_client = MagicMock()
            mock_create_client.return_value = mock_client
            
            memory = EntityMemory(
                crew=crew,
                embedder_config={"provider": "openai"},
                path=custom_path,
            )
            
            assert memory.storage.path == os.path.abspath(custom_path)
            
            assert mock_create_client.called
            config = mock_create_client.call_args[0][0]
            assert config.settings.persist_directory == os.path.abspath(custom_path)

    def test_entity_memory_default_behavior_unchanged(self, crew):
        """Test that EntityMemory default behavior (no path) is unchanged."""
        with patch("crewai.memory.storage.rag_storage.build_embedder") as mock_embedder, \
             patch("crewai.memory.storage.rag_storage.create_client") as mock_create_client:
            
            mock_embedder.return_value = lambda texts: [[0.1] * 384 for _ in texts]
            mock_client = MagicMock()
            mock_create_client.return_value = mock_client
            
            memory = EntityMemory(
                crew=crew,
                embedder_config={"provider": "openai"},
            )
            
            assert memory.storage.path is None


class TestPathIsolation:
    """Test suite for verifying path isolation between different storage instances."""

    def test_different_paths_create_different_storage_instances(self, crew, tmp_path):
        """Test that different paths result in isolated storage."""
        path1 = str(tmp_path / "storage1")
        path2 = str(tmp_path / "storage2")
        
        with patch("crewai.memory.storage.rag_storage.build_embedder") as mock_embedder, \
             patch("crewai.memory.storage.rag_storage.create_client") as mock_create_client:
            
            mock_embedder.return_value = lambda texts: [[0.1] * 384 for _ in texts]
            mock_create_client.return_value = MagicMock()
            
            storage1 = RAGStorage(
                type="short_term",
                embedder_config={"provider": "openai"},
                crew=crew,
                path=path1,
            )
            
            storage2 = RAGStorage(
                type="short_term",
                embedder_config={"provider": "openai"},
                crew=crew,
                path=path2,
            )
            
            assert storage1.path != storage2.path
            assert storage1.path == os.path.abspath(path1)
            assert storage2.path == os.path.abspath(path2)
            
            assert mock_create_client.call_count == 2
            config1 = mock_create_client.call_args_list[0][0][0]
            config2 = mock_create_client.call_args_list[1][0][0]
            assert config1.settings.persist_directory != config2.settings.persist_directory
