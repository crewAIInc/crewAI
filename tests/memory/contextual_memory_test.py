from unittest.mock import MagicMock, patch

import pytest
from crewai.memory import EntityMemory, LongTermMemory, ShortTermMemory, UserMemory
from crewai.memory.contextual.contextual_memory import ContextualMemory


@pytest.fixture
def mock_memories():
    return {
        "stm": MagicMock(spec=ShortTermMemory),
        "ltm": MagicMock(spec=LongTermMemory),
        "em": MagicMock(spec=EntityMemory),
        "um": MagicMock(spec=UserMemory),
    }


@pytest.fixture
def contextual_memory_mem0(mock_memories):
    return ContextualMemory(
        memory_provider="mem0",
        stm=mock_memories["stm"],
        ltm=mock_memories["ltm"],
        em=mock_memories["em"],
        um=mock_memories["um"],
    )


@pytest.fixture
def contextual_memory_other(mock_memories):
    return ContextualMemory(
        memory_provider="other",
        stm=mock_memories["stm"],
        ltm=mock_memories["ltm"],
        em=mock_memories["em"],
        um=mock_memories["um"],
    )


@pytest.fixture
def contextual_memory_none(mock_memories):
    return ContextualMemory(
        memory_provider=None,
        stm=mock_memories["stm"],
        ltm=mock_memories["ltm"],
        em=mock_memories["em"],
        um=mock_memories["um"],
    )


def test_build_context_for_task_mem0(contextual_memory_mem0, mock_memories):
    task = MagicMock(description="Test task")
    context = "Additional context"

    mock_memories["stm"].search.return_value = ["Recent insight"]
    mock_memories["ltm"].search.return_value = [
        {"metadata": {"suggestions": ["Historical data"]}}
    ]
    mock_memories["em"].search.return_value = [{"memory": "Entity memory"}]
    mock_memories["um"].search.return_value = [{"memory": "User memory"}]

    result = contextual_memory_mem0.build_context_for_task(task, context)

    assert "Recent Insights:" in result
    assert "Historical Data:" in result
    assert "Entities:" in result
    assert "User memories/preferences:" in result


def test_build_context_for_task_other_provider(contextual_memory_other, mock_memories):
    task = MagicMock(description="Test task")
    context = "Additional context"

    mock_memories["stm"].search.return_value = ["Recent insight"]
    mock_memories["ltm"].search.return_value = [
        {"metadata": {"suggestions": ["Historical data"]}}
    ]
    mock_memories["em"].search.return_value = [{"context": "Entity context"}]
    mock_memories["um"].search.return_value = [{"memory": "User memory"}]

    result = contextual_memory_other.build_context_for_task(task, context)

    assert "Recent Insights:" in result
    assert "Historical Data:" in result
    assert "Entities:" in result
    assert "User memories/preferences:" not in result


def test_build_context_for_task_none_provider(contextual_memory_none, mock_memories):
    task = MagicMock(description="Test task")
    context = "Additional context"

    mock_memories["stm"].search.return_value = ["Recent insight"]
    mock_memories["ltm"].search.return_value = [
        {"metadata": {"suggestions": ["Historical data"]}}
    ]
    mock_memories["em"].search.return_value = [{"context": "Entity context"}]
    mock_memories["um"].search.return_value = [{"memory": "User memory"}]

    result = contextual_memory_none.build_context_for_task(task, context)

    assert "Recent Insights:" in result
    assert "Historical Data:" in result
    assert "Entities:" in result
    assert "User memories/preferences:" not in result


def test_fetch_entity_context_mem0(contextual_memory_mem0, mock_memories):
    mock_memories["em"].search.return_value = [
        {"memory": "Entity 1"},
        {"memory": "Entity 2"},
    ]
    result = contextual_memory_mem0._fetch_entity_context("query")
    expected_result = "Entities:\n- Entity 1\n- Entity 2"
    assert result == expected_result


def test_fetch_entity_context_other_provider(contextual_memory_other, mock_memories):
    mock_memories["em"].search.return_value = [
        {"context": "Entity 1"},
        {"context": "Entity 2"},
    ]
    result = contextual_memory_other._fetch_entity_context("query")
    expected_result = "Entities:\n- Entity 1\n- Entity 2"
    assert result == expected_result


def test_user_memories_only_for_mem0(contextual_memory_mem0, mock_memories):
    mock_memories["um"].search.return_value = [{"memory": "User memory"}]

    # Test for mem0 provider
    result_mem0 = contextual_memory_mem0._fetch_user_memories("query")
    assert "User memories/preferences:" in result_mem0
    assert "User memory" in result_mem0

    # Additional test to ensure user memories are included/excluded in the full context
    task = MagicMock(description="Test task")
    context = "Additional context"
    mock_memories["stm"].search.return_value = ["Recent insight"]
    mock_memories["ltm"].search.return_value = [
        {"metadata": {"suggestions": ["Historical data"]}}
    ]
    mock_memories["em"].search.return_value = [{"memory": "Entity memory"}]

    full_context_mem0 = contextual_memory_mem0.build_context_for_task(task, context)
    assert "User memories/preferences:" in full_context_mem0
    assert "User memory" in full_context_mem0
