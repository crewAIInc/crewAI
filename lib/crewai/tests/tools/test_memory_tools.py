"""Tests for memory tool input validation and hardening."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from crewai.memory.types import MemoryMatch, MemoryRecord
from crewai.tools.memory_tools import RecallMemoryTool, RememberTool


@pytest.fixture
def mock_memory() -> MagicMock:
    """Create a mock Memory instance."""
    memory = MagicMock()
    memory.read_only = False
    memory.recall.return_value = []
    memory.remember.return_value = MemoryRecord(content="test")
    memory.remember_many.return_value = None
    return memory


@pytest.fixture
def recall_tool(mock_memory: MagicMock) -> RecallMemoryTool:
    return RecallMemoryTool(memory=mock_memory, description="test recall")


@pytest.fixture
def remember_tool(mock_memory: MagicMock) -> RememberTool:
    return RememberTool(memory=mock_memory, description="test remember")


# --- RecallMemoryTool ---


class TestRecallMemoryToolValidation:
    """Tests for RecallMemoryTool input validation."""

    def test_none_queries_returns_error(self, recall_tool: RecallMemoryTool) -> None:
        result = recall_tool._run(queries=None)
        assert "Error" in result

    def test_empty_list_queries_returns_error(
        self, recall_tool: RecallMemoryTool
    ) -> None:
        result = recall_tool._run(queries=[])
        assert "Error" in result

    def test_list_of_empty_strings_returns_error(
        self, recall_tool: RecallMemoryTool,
    ) -> None:
        result = recall_tool._run(queries=["", "  ", ""])
        assert "Error" in result

    def test_string_input_converted_to_list(
        self, recall_tool: RecallMemoryTool, mock_memory: MagicMock
    ) -> None:
        recall_tool._run(queries="single query")
        mock_memory.recall.assert_called_once_with("single query", limit=20)

    def test_valid_queries_calls_memory(
        self, recall_tool: RecallMemoryTool, mock_memory: MagicMock
    ) -> None:
        recall_tool._run(queries=["query1", "query2"])
        assert mock_memory.recall.call_count == 2

    def test_no_matches_returns_message(
        self, recall_tool: RecallMemoryTool, mock_memory: MagicMock
    ) -> None:
        mock_memory.recall.return_value = []
        result = recall_tool._run(queries=["test"])
        assert "No relevant memories found" in result

    def test_matches_formatted(
        self, recall_tool: RecallMemoryTool, mock_memory: MagicMock
    ) -> None:
        record = MemoryRecord(content="important fact")
        match = MemoryMatch(record=record, score=0.9)
        mock_memory.recall.return_value = [match]
        result = recall_tool._run(queries=["test"])
        assert "important fact" in result

    def test_deduplicates_across_queries(
        self, recall_tool: RecallMemoryTool, mock_memory: MagicMock
    ) -> None:
        record = MemoryRecord(id="same-id", content="fact")
        match = MemoryMatch(record=record, score=0.9)
        mock_memory.recall.return_value = [match]
        result = recall_tool._run(queries=["q1", "q2"])
        # Should only appear once despite two queries returning same record
        assert result.count("fact") == 1


# --- RememberTool ---


class TestRememberToolValidation:
    """Tests for RememberTool input validation."""

    def test_none_contents_returns_error(self, remember_tool: RememberTool) -> None:
        result = remember_tool._run(contents=None)
        assert "Error" in result

    def test_empty_list_returns_error(self, remember_tool: RememberTool) -> None:
        result = remember_tool._run(contents=[])
        assert "Error" in result

    def test_list_of_empty_strings_returns_error(
        self, remember_tool: RememberTool,
    ) -> None:
        result = remember_tool._run(contents=["", "  "])
        assert "Error" in result

    def test_string_input_converted_to_list(
        self, remember_tool: RememberTool, mock_memory: MagicMock
    ) -> None:
        remember_tool._run(contents="single fact")
        mock_memory.remember.assert_called_once_with("single fact")

    def test_single_item_calls_remember(
        self, remember_tool: RememberTool, mock_memory: MagicMock
    ) -> None:
        result = remember_tool._run(contents=["a fact"])
        mock_memory.remember.assert_called_once_with("a fact")
        assert "Saved to memory" in result

    def test_multiple_items_calls_remember_many(
        self, remember_tool: RememberTool, mock_memory: MagicMock
    ) -> None:
        result = remember_tool._run(contents=["fact1", "fact2"])
        mock_memory.remember_many.assert_called_once_with(["fact1", "fact2"])
        assert "2 items" in result

    def test_filters_empty_strings_from_list(
        self, remember_tool: RememberTool, mock_memory: MagicMock
    ) -> None:
        result = remember_tool._run(contents=["real fact", "", "  "])
        mock_memory.remember.assert_called_once_with("real fact")
        assert "Saved to memory" in result


# --- Non-string item handling ---


class TestNonStringItemHandling:
    """Tests that non-string items in lists are silently filtered out."""

    def test_recall_filters_non_string_queries(
        self, recall_tool: RecallMemoryTool, mock_memory: MagicMock
    ) -> None:
        result = recall_tool._run(queries=[123, True, "valid query"])
        mock_memory.recall.assert_called_once_with("valid query", limit=20)

    def test_recall_all_non_string_returns_error(
        self, recall_tool: RecallMemoryTool,
    ) -> None:
        result = recall_tool._run(queries=[123, None, True])
        assert "Error" in result

    def test_remember_filters_non_string_contents(
        self, remember_tool: RememberTool, mock_memory: MagicMock
    ) -> None:
        result = remember_tool._run(contents=[42, "real fact", False])
        mock_memory.remember.assert_called_once_with("real fact")
        assert "Saved to memory" in result

    def test_remember_all_non_string_returns_error(
        self, remember_tool: RememberTool,
    ) -> None:
        result = remember_tool._run(contents=[123, None])
        assert "Error" in result
