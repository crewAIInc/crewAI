from collections import defaultdict
from unittest.mock import ANY, MagicMock, patch

import pytest

from crewai.events.event_bus import crewai_event_bus
from crewai.events.types.memory_events import (
    MemoryQueryCompletedEvent,
    MemoryQueryStartedEvent,
    MemorySaveCompletedEvent,
    MemorySaveStartedEvent,
)

try:
    from mem0.memory.main import Memory
except ImportError:
    Memory = None

from crewai.agent import Agent
from crewai.crew import Crew, Process
from crewai.memory.external.external_memory import ExternalMemory
from crewai.memory.external.external_memory_item import ExternalMemoryItem
from crewai.memory.storage.bedrock_agentcore_storage import (
    BedrockAgentCoreConfig,
    BedrockAgentCoreStorage,
)
from crewai.memory.storage.interface import Storage
from crewai.task import Task


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
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
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
    with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"}):
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
    assert "agentcore" in supported_storages
    assert callable(supported_storages["mem0"])
    assert callable(supported_storages["agentcore"])


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
@pytest.mark.vcr(filter_headers=["authorization"])
def test_crew_external_memory_save_with_memory_flag(
    mem_method, crew_with_external_memory
):
    with patch(
        f"crewai.memory.external.external_memory.ExternalMemory.{mem_method}"
    ) as mock_method:
        crew_with_external_memory.kickoff()
        assert mock_method.call_count > 0


@pytest.mark.parametrize("mem_method", ["search", "save"])
@pytest.mark.vcr(filter_headers=["authorization"])
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
    events = defaultdict(list)

    external_memory_with_mocked_config.storage = custom_storage
    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(MemoryQueryStartedEvent)
        def on_search_started(source, event):
            events["MemoryQueryStartedEvent"].append(event)

        @crewai_event_bus.on(MemoryQueryCompletedEvent)
        def on_search_completed(source, event):
            events["MemoryQueryCompletedEvent"].append(event)

        external_memory_with_mocked_config.search(
            query="test value",
            limit=3,
            score_threshold=0.35,
        )

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
    events = defaultdict(list)

    external_memory_with_mocked_config.storage = custom_storage

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(MemorySaveStartedEvent)
        def on_save_started(source, event):
            events["MemorySaveStartedEvent"].append(event)

        @crewai_event_bus.on(MemorySaveCompletedEvent)
        def on_save_completed(source, event):
            events["MemorySaveCompletedEvent"].append(event)

        external_memory_with_mocked_config.save(
            value="saving value",
            metadata={"task": "test_task"},
        )

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


# AWS Bedrock AgentCore Tests
@pytest.fixture
def agentcore_config():
    """Fixture for basic AgentCore configuration."""
    return BedrockAgentCoreConfig(
        memory_id="memory-123",
        actor_id="actor-456",
        session_id="session-789",
        region_name="us-west-2",
    )


@pytest.fixture
def agentcore_config_with_namespaces():
    """Fixture for AgentCore configuration with namespaces."""
    return BedrockAgentCoreConfig(
        memory_id="memory-123",
        actor_id="actor-456",
        session_id="session-789",
        region_name="us-west-2",
        namespaces=["/preferences/actor-456", "/facts/actor-456/session-789"],
    )


@pytest.fixture
def mock_agentcore_storage():
    """Fixture to create a mock AgentCore storage."""
    return MagicMock(spec=BedrockAgentCoreStorage)


@pytest.fixture
def patch_configure_agentcore(mock_agentcore_storage):
    """Fixture to patch AgentCore configuration."""
    with patch(
        "crewai.memory.external.external_memory.ExternalMemory._configure_agentcore",
        return_value=mock_agentcore_storage,
    ) as mocked:
        yield mocked


def test_external_memory_agentcore_create_storage_success(agentcore_config):
    """Test successful creation of AgentCore storage."""
    embedder_config = {"provider": "agentcore", "config": agentcore_config}

    with patch("boto3.client"):
        storage = ExternalMemory.create_storage(None, embedder_config)
        assert isinstance(storage, BedrockAgentCoreStorage)


def test_external_memory_agentcore_create_storage_with_dict_config():
    """Test successful creation of AgentCore storage with dict configuration."""
    # Create a dict configuration instead of BedrockAgentCoreConfig instance
    config_dict = {
        "memory_id": "test-memory-id",
        "actor_id": "test-actor",
        "session_id": "test-session",
        "region_name": "us-west-2",
        "namespaces": ["/test/namespace"],
    }

    embedder_config = {"provider": "agentcore", "config": config_dict}

    with patch("boto3.client"):
        storage = ExternalMemory.create_storage(None, embedder_config)
        assert isinstance(storage, BedrockAgentCoreStorage)
        # Verify the dict was converted to BedrockAgentCoreConfig
        assert isinstance(storage.config, BedrockAgentCoreConfig)
        assert storage.config.memory_id == "test-memory-id"
        assert storage.config.actor_id == "test-actor"
        assert storage.config.session_id == "test-session"
        assert storage.config.region_name == "us-west-2"
        assert storage.config.namespaces == ["/test/namespace"]


def test_external_memory_agentcore_create_storage_with_minimal_dict_config():
    """Test AgentCore storage creation with minimal dict configuration (only required fields)."""
    # Create a minimal dict configuration with only required fields
    config_dict = {
        "memory_id": "minimal-memory",
        "actor_id": "minimal-actor",
        "session_id": "minimal-session",
    }

    embedder_config = {"provider": "agentcore", "config": config_dict}

    with patch("boto3.client"):
        storage = ExternalMemory.create_storage(None, embedder_config)
        assert isinstance(storage, BedrockAgentCoreStorage)
        assert isinstance(storage.config, BedrockAgentCoreConfig)
        assert storage.config.memory_id == "minimal-memory"
        assert storage.config.actor_id == "minimal-actor"
        assert storage.config.session_id == "minimal-session"
        # Check defaults are applied
        assert storage.config.region_name == "us-east-1"  # default region
        assert storage.config.namespaces == []  # default empty list


def test_external_memory_agentcore_create_storage_missing_config():
    """Test AgentCore storage creation with missing config."""
    embedder_config = {"provider": "agentcore", "config": None}

    with pytest.raises(
        ValueError, match="AgentCore storage requires explicit configuration"
    ):
        ExternalMemory.create_storage(None, embedder_config)


def test_external_memory_agentcore_create_storage_invalid_config():
    """Test AgentCore storage creation with invalid config."""
    embedder_config = {"provider": "agentcore", "config": {"invalid": "config"}}

    with pytest.raises(ValueError, match="Invalid AgentCore configuration"):
        ExternalMemory.create_storage(None, embedder_config)


def test_external_memory_agentcore_initialization(
    agentcore_config, patch_configure_agentcore
):
    """Test AgentCore external memory initialization."""
    embedder_config = {"provider": "agentcore", "config": agentcore_config}
    external_memory = ExternalMemory(embedder_config=embedder_config)

    assert external_memory is not None
    assert isinstance(external_memory, ExternalMemory)


def test_external_memory_agentcore_with_crew(
    agentcore_config, patch_configure_agentcore
):
    """Test AgentCore external memory integration with crew."""
    embedder_config = {"provider": "agentcore", "config": agentcore_config}
    external_memory = ExternalMemory(embedder_config=embedder_config)

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
        external_memory=external_memory,
    )

    assert crew._external_memory is not None
    assert isinstance(crew._external_memory, ExternalMemory)


def test_external_memory_agentcore_save_operation(
    agentcore_config, patch_configure_agentcore
):
    """Test AgentCore external memory save operation."""
    embedder_config = {"provider": "agentcore", "config": agentcore_config}
    external_memory = ExternalMemory(embedder_config=embedder_config)

    with patch.object(ExternalMemory, "save") as mock_save:
        external_memory.save(
            value="test agentcore value",
            metadata={"task": "agentcore_test"},
        )

        mock_save.assert_called_once_with(
            value="test agentcore value",
            metadata={"task": "agentcore_test"},
        )


def test_external_memory_agentcore_search_operation(
    agentcore_config, patch_configure_agentcore
):
    """Test AgentCore external memory search operation."""
    embedder_config = {"provider": "agentcore", "config": agentcore_config}
    external_memory = ExternalMemory(embedder_config=embedder_config)

    with patch.object(ExternalMemory, "search", return_value=[]) as mock_search:
        results = external_memory.search("test query", limit=5, score_threshold=0.7)

        mock_search.assert_called_once_with("test query", limit=5, score_threshold=0.7)
        assert results == []


def test_external_memory_agentcore_reset_operation(
    agentcore_config, patch_configure_agentcore
):
    """Test AgentCore external memory reset operation."""
    embedder_config = {"provider": "agentcore", "config": agentcore_config}
    external_memory = ExternalMemory(embedder_config=embedder_config)

    with patch.object(ExternalMemory, "reset") as mock_reset:
        external_memory.reset()
        mock_reset.assert_called_once()


def test_external_memory_agentcore_with_namespaces(
    agentcore_config_with_namespaces, patch_configure_agentcore
):
    """Test AgentCore external memory with namespaces."""
    embedder_config = {
        "provider": "agentcore",
        "config": agentcore_config_with_namespaces,
    }
    external_memory = ExternalMemory(embedder_config=embedder_config)

    assert external_memory is not None
    assert isinstance(external_memory, ExternalMemory)

    # Verify the configuration has namespaces
    assert len(agentcore_config_with_namespaces.namespaces) == 2
    assert "/preferences/actor-456" in agentcore_config_with_namespaces.namespaces
    assert "/facts/actor-456/session-789" in agentcore_config_with_namespaces.namespaces


def test_external_memory_agentcore_error_handling(agentcore_config):
    """Test AgentCore external memory error handling."""
    embedder_config = {"provider": "agentcore", "config": agentcore_config}

    # Test with invalid AgentCore client initialization
    with patch(
        "boto3.client",
        side_effect=Exception("Client error"),
    ):
        with pytest.raises(
            ValueError, match="Failed to initialize Bedrock AgentCore clients"
        ):
            ExternalMemory.create_storage(None, embedder_config)


def test_external_memory_agentcore_events_integration(
    agentcore_config, patch_configure_agentcore
):
    """Test AgentCore external memory events integration."""
    embedder_config = {"provider": "agentcore", "config": agentcore_config}
    external_memory = ExternalMemory(embedder_config=embedder_config)

    # Create a custom storage for testing events
    custom_storage = MagicMock()
    custom_storage.save = MagicMock()
    custom_storage.search = MagicMock(return_value=[])
    external_memory.storage = custom_storage

    events = defaultdict(list)

    with crewai_event_bus.scoped_handlers():

        @crewai_event_bus.on(MemorySaveStartedEvent)
        def on_save_started(source, event):
            events["MemorySaveStartedEvent"].append(event)

        @crewai_event_bus.on(MemorySaveCompletedEvent)
        def on_save_completed(source, event):
            events["MemorySaveCompletedEvent"].append(event)

        @crewai_event_bus.on(MemoryQueryStartedEvent)
        def on_search_started(source, event):
            events["MemoryQueryStartedEvent"].append(event)

        @crewai_event_bus.on(MemoryQueryCompletedEvent)
        def on_search_completed(source, event):
            events["MemoryQueryCompletedEvent"].append(event)

        # Test save events
        external_memory.save(
            value="agentcore test value",
            metadata={"strategy": "user_preferences"},
        )

        # Test search events
        external_memory.search("agentcore query", limit=10, score_threshold=0.5)

    # Verify events were emitted
    assert len(events["MemorySaveStartedEvent"]) == 1
    assert len(events["MemorySaveCompletedEvent"]) == 1
    assert len(events["MemoryQueryStartedEvent"]) == 1
    assert len(events["MemoryQueryCompletedEvent"]) == 1

    # Verify event content
    save_started_event = events["MemorySaveStartedEvent"][0]
    assert save_started_event.value == "agentcore test value"
    assert save_started_event.metadata == {"strategy": "user_preferences"}
    assert save_started_event.source_type == "external_memory"

    search_started_event = events["MemoryQueryStartedEvent"][0]
    assert search_started_event.query == "agentcore query"
    assert search_started_event.limit == 10
    assert search_started_event.score_threshold == 0.5
    assert search_started_event.source_type == "external_memory"
