import threading
from collections import defaultdict
from unittest.mock import ANY, MagicMock, patch

import pytest
from mem0.memory.main import Memory

from crewai.agent import Agent
from crewai.crew import Crew, Process
from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.memory_events import (
    MemoryQueryCompletedEvent,
    MemoryQueryStartedEvent,
    MemorySaveCompletedEvent,
    MemorySaveStartedEvent,
)
from crewai.memory.external.external_memory import ExternalMemory
from crewai.memory.external.external_memory_item import ExternalMemoryItem
from crewai.memory.storage.interface import Storage
from crewai.task import Task


@pytest.fixture(autouse=True)
def cleanup_event_handlers():
    """Cleanup event handlers before and after each test"""
    # Cleanup before test
    with crewai_event_bus._rwlock.w_locked():
        crewai_event_bus._sync_handlers = {}
        crewai_event_bus._async_handlers = {}
        crewai_event_bus._handler_dependencies = {}
        crewai_event_bus._execution_plan_cache = {}

    yield

    # Cleanup after test
    with crewai_event_bus._rwlock.w_locked():
        crewai_event_bus._sync_handlers = {}
        crewai_event_bus._async_handlers = {}
        crewai_event_bus._handler_dependencies = {}
        crewai_event_bus._execution_plan_cache = {}


@pytest.fixture
def mock_mem0_memory():
    mock_memory = MagicMock(spec=Memory)
    return mock_memory


@pytest.fixture
def patch_configure_mem0(mock_mem0_memory):
    with patch(
        "crewai.memory.external.external_memory.ExternalMemory._configure_mem0",
        return_value=mock_mem0_memory,
    ) as mocked:
        yield mocked


@pytest.fixture
def external_memory_with_mocked_config(patch_configure_mem0):
    embedder_config = {"provider": "mem0"}
    external_memory = ExternalMemory(embedder_config=embedder_config)
    return external_memory


@pytest.fixture
def crew_with_external_memory(external_memory_with_mocked_config, patch_configure_mem0):
    agent = Agent(
        role="Researcher",
        goal="Search relevant data and provide results",
        backstory="You are a researcher at a leading tech think tank.",
        tools=[],
        verbose=True,
    )

    task = Task(
        description="Perform a search on specific topics.",
        expected_output="A list of relevant URLs based on the search query.",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True,
        process=Process.sequential,
        memory=True,
        external_memory=external_memory_with_mocked_config,
    )

    return crew


@pytest.fixture
def crew_with_external_memory_without_memory_flag(
    external_memory_with_mocked_config, patch_configure_mem0
):
    agent = Agent(
        role="Researcher",
        goal="Search relevant data and provide results",
        backstory="You are a researcher at a leading tech think tank.",
        tools=[],
        verbose=True,
    )

    task = Task(
        description="Perform a search on specific topics.",
        expected_output="A list of relevant URLs based on the search query.",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True,
        process=Process.sequential,
        external_memory=external_memory_with_mocked_config,
    )

    return crew


def test_external_memory_initialization(external_memory_with_mocked_config):
    assert external_memory_with_mocked_config is not None
    assert isinstance(external_memory_with_mocked_config, ExternalMemory)


def test_external_memory_save(external_memory_with_mocked_config):
    memory_item = ExternalMemoryItem(
        value="test value", metadata={"task": "test_task"}, agent="test_agent"
    )

    with patch.object(ExternalMemory, "save") as mock_save:
        external_memory_with_mocked_config.save(
            value=memory_item.value,
            metadata=memory_item.metadata,
            agent=memory_item.agent,
        )

        mock_save.assert_called_once_with(
            value=memory_item.value,
            metadata=memory_item.metadata,
            agent=memory_item.agent,
        )


def test_external_memory_reset(external_memory_with_mocked_config):
    with patch(
        "crewai.memory.external.external_memory.ExternalMemory.reset"
    ) as mock_reset:
        external_memory_with_mocked_config.reset()
        mock_reset.assert_called_once()


def test_external_memory_supported_storages():
    supported_storages = ExternalMemory.external_supported_storages()
    assert "mem0" in supported_storages
    assert callable(supported_storages["mem0"])


def test_external_memory_create_storage_invalid_provider():
    embedder_config = {"provider": "invalid_provider", "config": {}}

    with pytest.raises(ValueError, match="Provider invalid_provider not supported"):
        ExternalMemory.create_storage(None, embedder_config)


def test_external_memory_create_storage_missing_provider():
    embedder_config = {"config": {}}

    with pytest.raises(
        ValueError, match="embedder_config must include a 'provider' key"
    ):
        ExternalMemory.create_storage(None, embedder_config)


def test_external_memory_create_storage_missing_config():
    with pytest.raises(ValueError, match="embedder_config is required"):
        ExternalMemory.create_storage(None, None)


def test_crew_with_external_memory_initialization(crew_with_external_memory):
    assert crew_with_external_memory._external_memory is not None
    assert isinstance(crew_with_external_memory._external_memory, ExternalMemory)
    assert crew_with_external_memory._external_memory.crew == crew_with_external_memory


@pytest.mark.parametrize("mem_type", ["external", "all"])
def test_crew_external_memory_reset(mem_type, crew_with_external_memory):
    with patch(
        "crewai.memory.external.external_memory.ExternalMemory.reset"
    ) as mock_reset:
        crew_with_external_memory.reset_memories(mem_type)
        mock_reset.assert_called_once()


@pytest.mark.parametrize("mem_method", ["search", "save"])
@pytest.mark.vcr()
def test_crew_external_memory_save_with_memory_flag(
    mem_method, crew_with_external_memory
):
    with patch(
        f"crewai.memory.external.external_memory.ExternalMemory.{mem_method}"
    ) as mock_method:
        crew_with_external_memory.kickoff()
        assert mock_method.call_count > 0


@pytest.mark.parametrize("mem_method", ["search", "save"])
@pytest.mark.vcr()
def test_crew_external_memory_save_using_crew_without_memory_flag(
    mem_method, crew_with_external_memory_without_memory_flag
):
    with patch(
        f"crewai.memory.external.external_memory.ExternalMemory.{mem_method}"
    ) as mock_method:
        crew_with_external_memory_without_memory_flag.kickoff()
        assert mock_method.call_count > 0


@pytest.fixture
def custom_storage():
    class CustomStorage(Storage):
        def __init__(self):
            self.memories = []

        def save(self, value, metadata=None, agent=None):
            self.memories.append({"value": value, "metadata": metadata, "agent": agent})

        def search(self, query, limit=10, score_threshold=0.5):
            return self.memories

        def reset(self):
            self.memories = []

    custom_storage = CustomStorage()
    return custom_storage


def test_external_memory_custom_storage(custom_storage, crew_with_external_memory):
    external_memory = ExternalMemory(storage=custom_storage)

    # by ensuring the crew is set, we can test that the storage is used
    external_memory.set_crew(crew_with_external_memory)

    test_value = "test value"
    test_metadata = {"source": "test"}
    external_memory.save(value=test_value, metadata=test_metadata)

    results = external_memory.search("test")
    assert len(results) == 1
    assert results[0]["value"] == test_value
    assert results[0]["metadata"] == test_metadata

    external_memory.reset()
    results = external_memory.search("test")
    assert len(results) == 0


def test_external_memory_search_events(
    custom_storage, external_memory_with_mocked_config
):
    events: dict[str, list] = defaultdict(list)
    condition = threading.Condition()

    external_memory_with_mocked_config.storage = custom_storage

    @crewai_event_bus.on(MemoryQueryStartedEvent)
    def on_search_started(source, event):
        with condition:
            events["MemoryQueryStartedEvent"].append(event)
            condition.notify()

    @crewai_event_bus.on(MemoryQueryCompletedEvent)
    def on_search_completed(source, event):
        with condition:
            events["MemoryQueryCompletedEvent"].append(event)
            condition.notify()

    external_memory_with_mocked_config.search(
        query="test value",
        limit=3,
        score_threshold=0.35,
    )

    with condition:
        success = condition.wait_for(
            lambda: len(events["MemoryQueryStartedEvent"]) >= 1
            and len(events["MemoryQueryCompletedEvent"]) >= 1,
            timeout=10,
        )
    assert success, "Timeout waiting for search events"
    assert len(events["MemoryQueryStartedEvent"]) == 1
    assert len(events["MemoryQueryCompletedEvent"]) == 1

    assert dict(events["MemoryQueryStartedEvent"][0]) == {
        "timestamp": ANY,
        "type": "memory_query_started",
        "source_fingerprint": None,
        "source_type": "external_memory",
        "fingerprint_metadata": None,
        "task_id": None,
        "task_name": None,
        "from_task": None,
        "from_agent": None,
        "agent_role": None,
        "agent_id": None,
        "query": "test value",
        "limit": 3,
        "score_threshold": 0.35,
    }

    assert dict(events["MemoryQueryCompletedEvent"][0]) == {
        "timestamp": ANY,
        "type": "memory_query_completed",
        "source_fingerprint": None,
        "source_type": "external_memory",
        "fingerprint_metadata": None,
        "task_id": None,
        "task_name": None,
        "from_task": None,
        "from_agent": None,
        "agent_role": None,
        "agent_id": None,
        "query": "test value",
        "results": [],
        "limit": 3,
        "score_threshold": 0.35,
        "query_time_ms": ANY,
    }


def test_external_memory_save_events(
    custom_storage, external_memory_with_mocked_config
):
    events: dict[str, list] = defaultdict(list)
    condition = threading.Condition()

    external_memory_with_mocked_config.storage = custom_storage

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

    external_memory_with_mocked_config.save(
        value="saving value",
        metadata={"task": "test_task"},
    )

    with condition:
        success = condition.wait_for(
            lambda: len(events["MemorySaveStartedEvent"]) >= 1
            and len(events["MemorySaveCompletedEvent"]) >= 1,
            timeout=10,
        )
    assert success, "Timeout waiting for save events"
    assert len(events["MemorySaveStartedEvent"]) == 1
    assert len(events["MemorySaveCompletedEvent"]) == 1

    assert dict(events["MemorySaveStartedEvent"][0]) == {
        "timestamp": ANY,
        "type": "memory_save_started",
        "source_fingerprint": None,
        "source_type": "external_memory",
        "fingerprint_metadata": None,
        "task_id": None,
        "task_name": None,
        "from_task": None,
        "from_agent": None,
        "agent_role": None,
        "agent_id": None,
        "value": "saving value",
        "metadata": {"task": "test_task"},
    }

    assert dict(events["MemorySaveCompletedEvent"][0]) == {
        "timestamp": ANY,
        "type": "memory_save_completed",
        "source_fingerprint": None,
        "source_type": "external_memory",
        "fingerprint_metadata": None,
        "task_id": None,
        "task_name": None,
        "from_task": None,
        "from_agent": None,
        "agent_role": None,
        "agent_id": None,
        "value": "saving value",
        "metadata": {"task": "test_task"},
        "save_time_ms": ANY,
    }
