from unittest.mock import MagicMock, patch

import pytest
from mem0.memory.main import Memory

from crewai.agent import Agent
from crewai.crew import Crew, Process
from crewai.memory.external.external_memory import ExternalMemory
from crewai.memory.external.external_memory_item import ExternalMemoryItem
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
@pytest.mark.vcr(filter_headers=["authorization"])
def test_crew_external_memory_save(mem_method, crew_with_external_memory):
    with patch(
        f"crewai.memory.external.external_memory.ExternalMemory.{mem_method}"
    ) as mock_method:
        crew_with_external_memory.kickoff()
        assert mock_method.call_count > 0
