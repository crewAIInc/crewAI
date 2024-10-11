# tests/memory/test_entity_memory.py

from unittest.mock import MagicMock, patch

import pytest
from crewai.memory.entity.entity_memory import EntityMemory
from crewai.memory.entity.entity_memory_item import EntityMemoryItem
from crewai.memory.storage.mem0_storage import Mem0Storage
from crewai.memory.storage.rag_storage import RAGStorage


@pytest.fixture
def mock_rag_storage():
    """Fixture to create a mock RAGStorage instance"""
    return MagicMock(spec=RAGStorage)


@pytest.fixture
def mock_mem0_storage():
    """Fixture to create a mock Mem0Storage instance"""
    return MagicMock(spec=Mem0Storage)


@pytest.fixture
def entity_memory_rag(mock_rag_storage):
    """Fixture to create an EntityMemory instance with RAGStorage"""
    with patch(
        "crewai.memory.entity.entity_memory.RAGStorage", return_value=mock_rag_storage
    ):
        return EntityMemory()


@pytest.fixture
def entity_memory_mem0(mock_mem0_storage):
    """Fixture to create an EntityMemory instance with Mem0Storage"""
    with patch(
        "crewai.memory.entity.entity_memory.Mem0Storage", return_value=mock_mem0_storage
    ):
        return EntityMemory(memory_provider="mem0")


def test_save_rag_storage(entity_memory_rag, mock_rag_storage):
    item = EntityMemoryItem(
        name="John Doe",
        type="Person",
        description="A software engineer",
        relationships="Works at TechCorp",
    )
    entity_memory_rag.save(item)

    expected_data = "John Doe(Person): A software engineer"
    mock_rag_storage.save.assert_called_once_with(expected_data, item.metadata)


def test_save_mem0_storage(entity_memory_mem0, mock_mem0_storage):
    item = EntityMemoryItem(
        name="John Doe",
        type="Person",
        description="A software engineer",
        relationships="Works at TechCorp",
    )
    entity_memory_mem0.save(item)

    expected_data = """
            Remember details about the following entity:
            Name: John Doe
            Type: Person
            Entity Description: A software engineer
            """
    mock_mem0_storage.save.assert_called_once_with(expected_data, item.metadata)


def test_search(entity_memory_rag, mock_rag_storage):
    query = "software engineer"
    limit = 5
    filters = {"type": "Person"}
    score_threshold = 0.7

    entity_memory_rag.search(query, limit, filters, score_threshold)

    mock_rag_storage.search.assert_called_once_with(
        query=query, limit=limit, filters=filters, score_threshold=score_threshold
    )


def test_reset(entity_memory_rag, mock_rag_storage):
    entity_memory_rag.reset()
    mock_rag_storage.reset.assert_called_once()


def test_reset_error(entity_memory_rag, mock_rag_storage):
    mock_rag_storage.reset.side_effect = Exception("Reset error")

    with pytest.raises(Exception) as exc_info:
        entity_memory_rag.reset()

    assert (
        str(exc_info.value)
        == "An error occurred while resetting the entity memory: Reset error"
    )


@pytest.mark.parametrize("memory_provider", [None, "other"])
def test_init_with_rag_storage(memory_provider):
    with patch("crewai.memory.entity.entity_memory.RAGStorage") as mock_rag_storage:
        EntityMemory(memory_provider=memory_provider)
        mock_rag_storage.assert_called_once()


def test_init_with_mem0_storage():
    with patch("crewai.memory.entity.entity_memory.Mem0Storage") as mock_mem0_storage:
        EntityMemory(memory_provider="mem0")
        mock_mem0_storage.assert_called_once()


def test_init_with_custom_storage():
    custom_storage = MagicMock()
    entity_memory = EntityMemory(storage=custom_storage)
    assert entity_memory.storage == custom_storage
