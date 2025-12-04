"""Tests for async memory operations."""

import threading
from collections import defaultdict
from unittest.mock import ANY, AsyncMock, MagicMock, patch

import pytest

from crewai.agent import Agent
from crewai.crew import Crew
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.memory_events import (
    MemoryQueryCompletedEvent,
    MemoryQueryStartedEvent,
    MemorySaveCompletedEvent,
    MemorySaveStartedEvent,
)
from crewai.memory.contextual.contextual_memory import ContextualMemory
from crewai.memory.entity.entity_memory import EntityMemory
from crewai.memory.entity.entity_memory_item import EntityMemoryItem
from crewai.memory.external.external_memory import ExternalMemory
from crewai.memory.long_term.long_term_memory import LongTermMemory
from crewai.memory.long_term.long_term_memory_item import LongTermMemoryItem
from crewai.memory.short_term.short_term_memory import ShortTermMemory
from crewai.task import Task


@pytest.fixture
def mock_agent():
    """Fixture to create a mock agent."""
    return Agent(
        role="Researcher",
        goal="Search relevant data and provide results",
        backstory="You are a researcher at a leading tech think tank.",
        tools=[],
        verbose=True,
    )


@pytest.fixture
def mock_task(mock_agent):
    """Fixture to create a mock task."""
    return Task(
        description="Perform a search on specific topics.",
        expected_output="A list of relevant URLs based on the search query.",
        agent=mock_agent,
    )


@pytest.fixture
def short_term_memory(mock_agent, mock_task):
    """Fixture to create a ShortTermMemory instance."""
    return ShortTermMemory(crew=Crew(agents=[mock_agent], tasks=[mock_task]))


@pytest.fixture
def long_term_memory(tmp_path):
    """Fixture to create a LongTermMemory instance."""
    db_path = str(tmp_path / "test_ltm.db")
    return LongTermMemory(path=db_path)


@pytest.fixture
def entity_memory(tmp_path, mock_agent, mock_task):
    """Fixture to create an EntityMemory instance."""
    return EntityMemory(
        crew=Crew(agents=[mock_agent], tasks=[mock_task]),
        path=str(tmp_path / "test_entities"),
    )


class TestAsyncShortTermMemory:
    """Tests for async ShortTermMemory operations."""

    @pytest.mark.asyncio
    async def test_asave_emits_events(self, short_term_memory):
        """Test that asave emits the correct events."""
        events: dict[str, list] = defaultdict(list)
        condition = threading.Condition()

        @crewai_event_bus.on(MemorySaveStartedEvent)
        def on_save_started(source, event):
            with condition:
                events["MemorySaveStartedEvent"].append(event)
                condition.notify()

        @crewai_event_bus.on(MemorySaveCompletedEvent)
        def on_save_completed(source, event):
            with condition:
                events["MemorySaveCompletedEvent"].append(event)
                condition.notify()

        await short_term_memory.asave(
            value="async test value",
            metadata={"task": "async_test_task"},
        )

        with condition:
            success = condition.wait_for(
                lambda: len(events["MemorySaveStartedEvent"]) >= 1
                and len(events["MemorySaveCompletedEvent"]) >= 1,
                timeout=5,
            )
        assert success, "Timeout waiting for async save events"

        assert len(events["MemorySaveStartedEvent"]) >= 1
        assert len(events["MemorySaveCompletedEvent"]) >= 1
        assert events["MemorySaveStartedEvent"][-1].value == "async test value"
        assert events["MemorySaveStartedEvent"][-1].source_type == "short_term_memory"

    @pytest.mark.asyncio
    async def test_asearch_emits_events(self, short_term_memory):
        """Test that asearch emits the correct events."""
        events: dict[str, list] = defaultdict(list)
        search_started = threading.Event()
        search_completed = threading.Event()

        with patch.object(short_term_memory.storage, "asearch", new_callable=AsyncMock, return_value=[]):

            @crewai_event_bus.on(MemoryQueryStartedEvent)
            def on_search_started(source, event):
                events["MemoryQueryStartedEvent"].append(event)
                search_started.set()

            @crewai_event_bus.on(MemoryQueryCompletedEvent)
            def on_search_completed(source, event):
                events["MemoryQueryCompletedEvent"].append(event)
                search_completed.set()

            await short_term_memory.asearch(
                query="async test query",
                limit=3,
                score_threshold=0.35,
            )

            assert search_started.wait(timeout=2), "Timeout waiting for search started event"
            assert search_completed.wait(timeout=2), "Timeout waiting for search completed event"

        assert len(events["MemoryQueryStartedEvent"]) >= 1
        assert len(events["MemoryQueryCompletedEvent"]) >= 1
        assert events["MemoryQueryStartedEvent"][-1].query == "async test query"
        assert events["MemoryQueryStartedEvent"][-1].source_type == "short_term_memory"


class TestAsyncLongTermMemory:
    """Tests for async LongTermMemory operations."""

    @pytest.mark.asyncio
    async def test_asave_emits_events(self, long_term_memory):
        """Test that asave emits the correct events."""
        events: dict[str, list] = defaultdict(list)
        condition = threading.Condition()

        @crewai_event_bus.on(MemorySaveStartedEvent)
        def on_save_started(source, event):
            with condition:
                events["MemorySaveStartedEvent"].append(event)
                condition.notify()

        @crewai_event_bus.on(MemorySaveCompletedEvent)
        def on_save_completed(source, event):
            with condition:
                events["MemorySaveCompletedEvent"].append(event)
                condition.notify()

        item = LongTermMemoryItem(
            task="async test task",
            agent="test_agent",
            expected_output="test output",
            datetime="2024-01-01T00:00:00",
            quality=0.9,
            metadata={"task": "async test task", "quality": 0.9},
        )

        await long_term_memory.asave(item)

        with condition:
            success = condition.wait_for(
                lambda: len(events["MemorySaveStartedEvent"]) >= 1
                and len(events["MemorySaveCompletedEvent"]) >= 1,
                timeout=5,
            )
        assert success, "Timeout waiting for async save events"

        assert len(events["MemorySaveStartedEvent"]) >= 1
        assert len(events["MemorySaveCompletedEvent"]) >= 1
        assert events["MemorySaveStartedEvent"][-1].source_type == "long_term_memory"

    @pytest.mark.asyncio
    async def test_asearch_emits_events(self, long_term_memory):
        """Test that asearch emits the correct events."""
        events: dict[str, list] = defaultdict(list)
        search_started = threading.Event()
        search_completed = threading.Event()

        @crewai_event_bus.on(MemoryQueryStartedEvent)
        def on_search_started(source, event):
            events["MemoryQueryStartedEvent"].append(event)
            search_started.set()

        @crewai_event_bus.on(MemoryQueryCompletedEvent)
        def on_search_completed(source, event):
            events["MemoryQueryCompletedEvent"].append(event)
            search_completed.set()

        await long_term_memory.asearch(task="async test task", latest_n=3)

        assert search_started.wait(timeout=2), "Timeout waiting for search started event"
        assert search_completed.wait(timeout=2), "Timeout waiting for search completed event"

        assert len(events["MemoryQueryStartedEvent"]) >= 1
        assert len(events["MemoryQueryCompletedEvent"]) >= 1
        assert events["MemoryQueryStartedEvent"][-1].source_type == "long_term_memory"

    @pytest.mark.asyncio
    async def test_asave_and_asearch_integration(self, long_term_memory):
        """Test that asave followed by asearch works correctly."""
        item = LongTermMemoryItem(
            task="integration test task",
            agent="test_agent",
            expected_output="test output",
            datetime="2024-01-01T00:00:00",
            quality=0.9,
            metadata={"task": "integration test task", "quality": 0.9},
        )

        await long_term_memory.asave(item)
        results = await long_term_memory.asearch(task="integration test task", latest_n=1)

        assert results is not None
        assert len(results) == 1
        assert results[0]["metadata"]["agent"] == "test_agent"


class TestAsyncEntityMemory:
    """Tests for async EntityMemory operations."""

    @pytest.mark.asyncio
    async def test_asave_single_item_emits_events(self, entity_memory):
        """Test that asave with a single item emits the correct events."""
        events: dict[str, list] = defaultdict(list)
        condition = threading.Condition()

        @crewai_event_bus.on(MemorySaveStartedEvent)
        def on_save_started(source, event):
            with condition:
                events["MemorySaveStartedEvent"].append(event)
                condition.notify()

        @crewai_event_bus.on(MemorySaveCompletedEvent)
        def on_save_completed(source, event):
            with condition:
                events["MemorySaveCompletedEvent"].append(event)
                condition.notify()

        item = EntityMemoryItem(
            name="TestEntity",
            type="Person",
            description="A test entity for async operations",
            relationships="Related to other test entities",
        )

        await entity_memory.asave(item)

        with condition:
            success = condition.wait_for(
                lambda: len(events["MemorySaveStartedEvent"]) >= 1
                and len(events["MemorySaveCompletedEvent"]) >= 1,
                timeout=5,
            )
        assert success, "Timeout waiting for async save events"

        assert len(events["MemorySaveStartedEvent"]) >= 1
        assert len(events["MemorySaveCompletedEvent"]) >= 1
        assert events["MemorySaveStartedEvent"][-1].source_type == "entity_memory"

    @pytest.mark.asyncio
    async def test_asearch_emits_events(self, entity_memory):
        """Test that asearch emits the correct events."""
        events: dict[str, list] = defaultdict(list)
        search_started = threading.Event()
        search_completed = threading.Event()

        @crewai_event_bus.on(MemoryQueryStartedEvent)
        def on_search_started(source, event):
            events["MemoryQueryStartedEvent"].append(event)
            search_started.set()

        @crewai_event_bus.on(MemoryQueryCompletedEvent)
        def on_search_completed(source, event):
            events["MemoryQueryCompletedEvent"].append(event)
            search_completed.set()

        await entity_memory.asearch(query="TestEntity", limit=5, score_threshold=0.6)

        assert search_started.wait(timeout=2), "Timeout waiting for search started event"
        assert search_completed.wait(timeout=2), "Timeout waiting for search completed event"

        assert len(events["MemoryQueryStartedEvent"]) >= 1
        assert len(events["MemoryQueryCompletedEvent"]) >= 1
        assert events["MemoryQueryStartedEvent"][-1].source_type == "entity_memory"


class TestAsyncContextualMemory:
    """Tests for async ContextualMemory operations."""

    @pytest.mark.asyncio
    async def test_abuild_context_for_task_with_empty_query(self, mock_task):
        """Test that abuild_context_for_task returns empty string for empty query."""
        mock_task.description = ""
        contextual_memory = ContextualMemory(
            stm=None,
            ltm=None,
            em=None,
            exm=None,
        )

        result = await contextual_memory.abuild_context_for_task(mock_task, "")
        assert result == ""

    @pytest.mark.asyncio
    async def test_abuild_context_for_task_with_none_memories(self, mock_task):
        """Test that abuild_context_for_task handles None memory sources."""
        contextual_memory = ContextualMemory(
            stm=None,
            ltm=None,
            em=None,
            exm=None,
        )

        result = await contextual_memory.abuild_context_for_task(mock_task, "some context")
        assert result == ""

    @pytest.mark.asyncio
    async def test_abuild_context_for_task_aggregates_results(self, mock_agent, mock_task):
        """Test that abuild_context_for_task aggregates results from all memory sources."""
        mock_stm = MagicMock(spec=ShortTermMemory)
        mock_stm.asearch = AsyncMock(return_value=[{"content": "STM insight"}])

        mock_ltm = MagicMock(spec=LongTermMemory)
        mock_ltm.asearch = AsyncMock(
            return_value=[{"metadata": {"suggestions": ["LTM suggestion"]}}]
        )

        mock_em = MagicMock(spec=EntityMemory)
        mock_em.asearch = AsyncMock(return_value=[{"content": "Entity info"}])

        mock_exm = MagicMock(spec=ExternalMemory)
        mock_exm.asearch = AsyncMock(return_value=[{"content": "External memory"}])

        contextual_memory = ContextualMemory(
            stm=mock_stm,
            ltm=mock_ltm,
            em=mock_em,
            exm=mock_exm,
            agent=mock_agent,
            task=mock_task,
        )

        result = await contextual_memory.abuild_context_for_task(mock_task, "additional context")

        assert "Recent Insights:" in result
        assert "STM insight" in result
        assert "Historical Data:" in result
        assert "LTM suggestion" in result
        assert "Entities:" in result
        assert "Entity info" in result
        assert "External memories:" in result
        assert "External memory" in result

    @pytest.mark.asyncio
    async def test_afetch_stm_context_returns_formatted_results(self, mock_agent, mock_task):
        """Test that _afetch_stm_context returns properly formatted results."""
        mock_stm = MagicMock(spec=ShortTermMemory)
        mock_stm.asearch = AsyncMock(
            return_value=[
                {"content": "First insight"},
                {"content": "Second insight"},
            ]
        )

        contextual_memory = ContextualMemory(
            stm=mock_stm,
            ltm=None,
            em=None,
            exm=None,
        )

        result = await contextual_memory._afetch_stm_context("test query")

        assert "Recent Insights:" in result
        assert "- First insight" in result
        assert "- Second insight" in result

    @pytest.mark.asyncio
    async def test_afetch_ltm_context_returns_formatted_results(self, mock_agent, mock_task):
        """Test that _afetch_ltm_context returns properly formatted results."""
        mock_ltm = MagicMock(spec=LongTermMemory)
        mock_ltm.asearch = AsyncMock(
            return_value=[
                {"metadata": {"suggestions": ["Suggestion 1", "Suggestion 2"]}},
            ]
        )

        contextual_memory = ContextualMemory(
            stm=None,
            ltm=mock_ltm,
            em=None,
            exm=None,
        )

        result = await contextual_memory._afetch_ltm_context("test task")

        assert "Historical Data:" in result
        assert "- Suggestion 1" in result
        assert "- Suggestion 2" in result

    @pytest.mark.asyncio
    async def test_afetch_entity_context_returns_formatted_results(self, mock_agent, mock_task):
        """Test that _afetch_entity_context returns properly formatted results."""
        mock_em = MagicMock(spec=EntityMemory)
        mock_em.asearch = AsyncMock(
            return_value=[
                {"content": "Entity A details"},
                {"content": "Entity B details"},
            ]
        )

        contextual_memory = ContextualMemory(
            stm=None,
            ltm=None,
            em=mock_em,
            exm=None,
        )

        result = await contextual_memory._afetch_entity_context("test query")

        assert "Entities:" in result
        assert "- Entity A details" in result
        assert "- Entity B details" in result

    @pytest.mark.asyncio
    async def test_afetch_external_context_returns_formatted_results(self):
        """Test that _afetch_external_context returns properly formatted results."""
        mock_exm = MagicMock(spec=ExternalMemory)
        mock_exm.asearch = AsyncMock(
            return_value=[
                {"content": "External data 1"},
                {"content": "External data 2"},
            ]
        )

        contextual_memory = ContextualMemory(
            stm=None,
            ltm=None,
            em=None,
            exm=mock_exm,
        )

        result = await contextual_memory._afetch_external_context("test query")

        assert "External memories:" in result
        assert "- External data 1" in result
        assert "- External data 2" in result

    @pytest.mark.asyncio
    async def test_afetch_methods_return_empty_for_empty_results(self):
        """Test that async fetch methods return empty string for no results."""
        mock_stm = MagicMock(spec=ShortTermMemory)
        mock_stm.asearch = AsyncMock(return_value=[])

        mock_ltm = MagicMock(spec=LongTermMemory)
        mock_ltm.asearch = AsyncMock(return_value=[])

        mock_em = MagicMock(spec=EntityMemory)
        mock_em.asearch = AsyncMock(return_value=[])

        mock_exm = MagicMock(spec=ExternalMemory)
        mock_exm.asearch = AsyncMock(return_value=[])

        contextual_memory = ContextualMemory(
            stm=mock_stm,
            ltm=mock_ltm,
            em=mock_em,
            exm=mock_exm,
        )

        stm_result = await contextual_memory._afetch_stm_context("query")
        ltm_result = await contextual_memory._afetch_ltm_context("task")
        em_result = await contextual_memory._afetch_entity_context("query")
        exm_result = await contextual_memory._afetch_external_context("query")

        assert stm_result == ""
        assert ltm_result is None
        assert em_result == ""
        assert exm_result == ""