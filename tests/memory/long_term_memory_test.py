# tests/memory/long_term_memory_test.py

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from crewai.memory.long_term.long_term_memory import LongTermMemory
from crewai.memory.long_term.long_term_memory_item import LongTermMemoryItem
from crewai.memory.storage.ltm_sqlite_storage import LTMSQLiteStorage


@pytest.fixture
def mock_storage():
    """Fixture to create a mock LTMSQLiteStorage instance"""
    return MagicMock(spec=LTMSQLiteStorage)


@pytest.fixture
def long_term_memory(mock_storage):
    """Fixture to create a LongTermMemory instance with mock storage"""
    return LongTermMemory(storage=mock_storage)


def test_save(long_term_memory, mock_storage):
    memory = LongTermMemoryItem(
        agent="test_agent",
        task="test_task",
        expected_output="test_output",
        datetime="2023-01-01 12:00:00",
        quality=0.5,
        metadata={"additional_info": "test_info"},
    )
    long_term_memory.save(memory)

    expected_metadata = {
        "additional_info": "test_info",
        "agent": "test_agent",
        "expected_output": "test_output",
        "quality": 0.5,  # Include quality in expected metadata
    }
    mock_storage.save.assert_called_once_with(
        task_description="test_task",
        score=0.5,
        metadata=expected_metadata,
        datetime="2023-01-01 12:00:00",
    )


def test_search(long_term_memory, mock_storage):
    mock_storage.load.return_value = [
        {
            "metadata": {
                "agent": "test_agent",
                "expected_output": "test_output",
                "task": "test_task",
            },
            "datetime": "2023-01-01 12:00:00",
            "score": 0.5,
        }
    ]

    result = long_term_memory.search("test_task", latest_n=5)

    mock_storage.load.assert_called_once_with("test_task", 5)
    assert len(result) == 1
    assert result[0]["metadata"]["agent"] == "test_agent"
    assert result[0]["metadata"]["expected_output"] == "test_output"
    assert result[0]["metadata"]["task"] == "test_task"
    assert result[0]["datetime"] == "2023-01-01 12:00:00"
    assert result[0]["score"] == 0.5


def test_save_with_minimal_metadata(long_term_memory, mock_storage):
    memory = LongTermMemoryItem(
        agent="minimal_agent",
        task="minimal_task",
        expected_output="minimal_output",
        datetime="2023-01-01 12:00:00",
        quality=0.3,
        metadata={},
    )
    long_term_memory.save(memory)

    expected_metadata = {
        "agent": "minimal_agent",
        "expected_output": "minimal_output",
        "quality": 0.3,  # Include quality in expected metadata
    }
    mock_storage.save.assert_called_once_with(
        task_description="minimal_task",
        score=0.3,
        metadata=expected_metadata,
        datetime="2023-01-01 12:00:00",
    )


def test_reset(long_term_memory, mock_storage):
    long_term_memory.reset()
    mock_storage.reset.assert_called_once()


def test_search_with_no_results(long_term_memory, mock_storage):
    mock_storage.load.return_value = []
    result = long_term_memory.search("nonexistent_task")
    assert result == []


def test_init_with_default_storage():
    with patch(
        "crewai.memory.long_term.long_term_memory.LTMSQLiteStorage"
    ) as mock_storage_class:
        LongTermMemory()
        mock_storage_class.assert_called_once()


def test_init_with_custom_storage():
    custom_storage = MagicMock()
    memory = LongTermMemory(storage=custom_storage)
    assert memory.storage == custom_storage


@pytest.mark.parametrize("latest_n", [1, 3, 5, 10])
def test_search_with_different_latest_n(long_term_memory, mock_storage, latest_n):
    long_term_memory.search("test_task", latest_n=latest_n)
    mock_storage.load.assert_called_once_with("test_task", latest_n)
