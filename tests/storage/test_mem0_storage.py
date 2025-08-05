from unittest.mock import MagicMock, patch

import pytest
from mem0.client.main import MemoryClient
from mem0.memory.main import Memory

from crewai.memory.storage.mem0_storage import Mem0Storage


# Define the class (if not already defined)
class MockCrew:
    def __init__(self, memory_config):
        self.memory_config = memory_config
        self.agents = [MagicMock(role="Test Agent")]


@pytest.fixture
def mock_mem0_memory():
    """Fixture to create a mock Memory instance"""
    mock_memory = MagicMock(spec=Memory)
    return mock_memory


@pytest.fixture
def mem0_storage_with_mocked_config(mock_mem0_memory):
    """Fixture to create a Mem0Storage instance with mocked dependencies"""

    # Patch the Memory class to return our mock
    with patch("mem0.memory.main.Memory.from_config", return_value=mock_mem0_memory) as mock_from_config:
        config = {
            "vector_store": {
                "provider": "mock_vector_store",
                "config": {"host": "localhost", "port": 6333},
            },
            "llm": {
                "provider": "mock_llm",
                "config": {"api_key": "mock-api-key", "model": "mock-model"},
            },
            "embedder": {
                "provider": "mock_embedder",
                "config": {"api_key": "mock-api-key", "model": "mock-model"},
            },
            "graph_store": {
                "provider": "mock_graph_store",
                "config": {
                    "url": "mock-url",
                    "username": "mock-user",
                    "password": "mock-password",
                },
            },
            "history_db_path": "/mock/path",
            "version": "test-version",
            "custom_fact_extraction_prompt": "mock prompt 1",
            "custom_update_memory_prompt": "mock prompt 2",
        }

        # Instantiate the class with memory_config
        # Parameters like run_id, includes, and excludes doesn't matter in Memory OSS
        crew = MockCrew(
            memory_config={
                "provider": "mem0",
                "config": {"user_id": "test_user", "local_mem0_config": config, "run_id": "my_run_id", "includes": "include1","excludes": "exclude1", "infer" : True},
            }
        )

        mem0_storage = Mem0Storage(type="short_term", crew=crew)
        return mem0_storage, mock_from_config, config


def test_mem0_storage_initialization(mem0_storage_with_mocked_config, mock_mem0_memory):
    """Test that Mem0Storage initializes correctly with the mocked config"""
    mem0_storage, mock_from_config, config = mem0_storage_with_mocked_config
    assert mem0_storage.memory_type == "short_term"
    assert mem0_storage.memory is mock_mem0_memory
    mock_from_config.assert_called_once_with(config)


@pytest.fixture
def mock_mem0_memory_client():
    """Fixture to create a mock MemoryClient instance"""
    mock_memory = MagicMock(spec=MemoryClient)
    return mock_memory


@pytest.fixture
def mem0_storage_with_memory_client_using_config_from_crew(mock_mem0_memory_client):
    """Fixture to create a Mem0Storage instance with mocked dependencies"""

    # We need to patch the MemoryClient before it's instantiated
    with patch.object(MemoryClient, "__new__", return_value=mock_mem0_memory_client):
        crew = MockCrew(
            memory_config={
                "provider": "mem0",
                "config": {
                    "user_id": "test_user",
                    "api_key": "ABCDEFGH",
                    "org_id": "my_org_id",
                    "project_id": "my_project_id",
                    "run_id": "my_run_id",
                    "includes": "include1",
                    "excludes": "exclude1",
                    "infer": True
                },
            }
        )

        mem0_storage = Mem0Storage(type="short_term", crew=crew)
        return mem0_storage


@pytest.fixture
def mem0_storage_with_memory_client_using_explictly_config(mock_mem0_memory_client, mock_mem0_memory):
    """Fixture to create a Mem0Storage instance with mocked dependencies"""

    # We need to patch both MemoryClient and Memory to prevent actual initialization
    with patch.object(MemoryClient, "__new__", return_value=mock_mem0_memory_client), \
         patch.object(Memory, "__new__", return_value=mock_mem0_memory):

        crew = MockCrew(
            memory_config={
                "provider": "mem0",
                "config": {
                    "user_id": "test_user",
                    "api_key": "ABCDEFGH",
                    "org_id": "my_org_id",
                    "project_id": "my_project_id",
                },
            }
        )

        new_config = {"provider": "mem0", "config": {"api_key": "new-api-key"}}

        mem0_storage = Mem0Storage(type="short_term", crew=crew, config=new_config)
        return mem0_storage


def test_mem0_storage_with_memory_client_initialization(
    mem0_storage_with_memory_client_using_config_from_crew, mock_mem0_memory_client
):
    """Test Mem0Storage initialization with MemoryClient"""
    assert (
        mem0_storage_with_memory_client_using_config_from_crew.memory_type
        == "short_term"
    )
    assert (
        mem0_storage_with_memory_client_using_config_from_crew.memory
        is mock_mem0_memory_client
    )


def test_mem0_storage_with_explict_config(
    mem0_storage_with_memory_client_using_explictly_config,
):
    expected_config = {"provider": "mem0", "config": {"api_key": "new-api-key"}}
    assert (
        mem0_storage_with_memory_client_using_explictly_config.config == expected_config
    )


def test_mem0_storage_updates_project_with_custom_categories(mock_mem0_memory_client):
    mock_mem0_memory_client.update_project = MagicMock()

    new_categories = [
    {"lifestyle_management_concerns": "Tracks daily routines, habits, hobbies and interests including cooking, time management and work-life balance"},
    ]

    crew = MockCrew(
        memory_config={
            "provider": "mem0",
            "config": {
                "user_id": "test_user",
                "api_key": "ABCDEFGH",
                "org_id": "my_org_id",
                "project_id": "my_project_id",
                "custom_categories": new_categories,
            },
        }
    )

    with patch.object(MemoryClient, "__new__", return_value=mock_mem0_memory_client):
        _ = Mem0Storage(type="short_term", crew=crew)

    mock_mem0_memory_client.update_project.assert_called_once_with(
        custom_categories=new_categories
    )




def test_save_method_with_memory_oss(mem0_storage_with_mocked_config):
    """Test save method for different memory types"""
    mem0_storage, _, _ = mem0_storage_with_mocked_config
    mem0_storage.memory.add = MagicMock()

    # Test short_term memory type (already set in fixture)
    test_value = "This is a test memory"
    test_metadata = {"key": "value"}

    mem0_storage.save(test_value, test_metadata)

    mem0_storage.memory.add.assert_called_once_with(
        [{"role": "assistant" , "content": test_value}],
        infer=True,
        metadata={"type": "short_term", "key": "value"},
        run_id="my_run_id",
        user_id="test_user",
        agent_id='Test_Agent'
    )

def test_save_method_with_multiple_agents(mem0_storage_with_mocked_config):
    mem0_storage, _, _ = mem0_storage_with_mocked_config
    mem0_storage.crew.agents = [MagicMock(role="Test Agent"), MagicMock(role="Test Agent 2"), MagicMock(role="Test Agent 3")]
    mem0_storage.memory.add = MagicMock()

    test_value = "This is a test memory"
    test_metadata = {"key": "value"}

    mem0_storage.save(test_value, test_metadata)

    mem0_storage.memory.add.assert_called_once_with(
        [{"role": "assistant" , "content": test_value}],
        infer=True,
        metadata={"type": "short_term", "key": "value"},
        run_id="my_run_id",
        user_id="test_user",
        agent_id='Test_Agent_Test_Agent_2_Test_Agent_3'
    )


def test_save_method_with_memory_client(mem0_storage_with_memory_client_using_config_from_crew):
    """Test save method for different memory types"""
    mem0_storage = mem0_storage_with_memory_client_using_config_from_crew
    mem0_storage.memory.add = MagicMock()

    # Test short_term memory type (already set in fixture)
    test_value = "This is a test memory"
    test_metadata = {"key": "value"}

    mem0_storage.save(test_value, test_metadata)

    mem0_storage.memory.add.assert_called_once_with(
        [{'role': 'assistant' , 'content': test_value}],
        infer=True,
        metadata={"type": "short_term", "key": "value"},
        version="v2",
        run_id="my_run_id",
        includes="include1",
        excludes="exclude1",
        output_format='v1.1',
        user_id='test_user',
         agent_id='Test_Agent'
    )


def test_search_method_with_memory_oss(mem0_storage_with_mocked_config):
    """Test search method for different memory types"""
    mem0_storage, _, _ = mem0_storage_with_mocked_config
    mock_results = {"results": [{"score": 0.9, "content": "Result 1"}, {"score": 0.4, "content": "Result 2"}]}
    mem0_storage.memory.search = MagicMock(return_value=mock_results)

    results = mem0_storage.search("test query", limit=5, score_threshold=0.5)

    mem0_storage.memory.search.assert_called_once_with(
        query="test query",
        limit=5,
        user_id="test_user",
        filters={'AND': [{'run_id': 'my_run_id'}]},
        threshold=0.5
    )

    assert len(results) == 2
    assert results[0]["content"] == "Result 1"


def test_search_method_with_memory_client(mem0_storage_with_memory_client_using_config_from_crew):
    """Test search method for different memory types"""
    mem0_storage = mem0_storage_with_memory_client_using_config_from_crew
    mock_results = {"results": [{"score": 0.9, "content": "Result 1"}, {"score": 0.4, "content": "Result 2"}]}
    mem0_storage.memory.search = MagicMock(return_value=mock_results)

    results = mem0_storage.search("test query", limit=5, score_threshold=0.5)

    mem0_storage.memory.search.assert_called_once_with(
        query="test query",
        limit=5,
        metadata={"type": "short_term"},
        user_id="test_user",
        version='v2',
        run_id="my_run_id",
        output_format='v1.1',
        filters={'AND': [{'run_id': 'my_run_id'}]},
        threshold=0.5
    )

    assert len(results) == 2
    assert results[0]["content"] == "Result 1"


def test_mem0_storage_default_infer_value(mock_mem0_memory_client):
    """Test that Mem0Storage sets infer=True by default for short_term memory."""
    with patch.object(MemoryClient, "__new__", return_value=mock_mem0_memory_client):
        crew = MockCrew(
            memory_config={
                "provider": "mem0",
                "config": {
                    "user_id": "test_user",
                    "api_key": "ABCDEFGH"
                },
            }
        )

        mem0_storage = Mem0Storage(type="short_term", crew=crew)
        assert mem0_storage.infer is True

def test_save_memory_using_agent_entity(mock_mem0_memory_client):
    config = {
        "agent_id": "agent-123",
    }

    mock_memory = MagicMock(spec=Memory)
    with patch.object(Memory, "__new__", return_value=mock_memory):
        mem0_storage = Mem0Storage(type="external", config=config)
        mem0_storage.save("test memory", {"key": "value"})
        mem0_storage.memory.add.assert_called_once_with(
            [{'role': 'assistant' , 'content': 'test memory'}],
            infer=True,
            metadata={"type": "external", "key": "value"},
            agent_id="agent-123",
        )

def test_search_method_with_agent_entity():
    mem0_storage = Mem0Storage(type="external", config={"agent_id": "agent-123"})
    mock_results = {"results": [{"score": 0.9, "content": "Result 1"}, {"score": 0.4, "content": "Result 2"}]}
    mem0_storage.memory.search = MagicMock(return_value=mock_results)

    results = mem0_storage.search("test query", limit=5, score_threshold=0.5)

    mem0_storage.memory.search.assert_called_once_with(
        query="test query",
        limit=5,
        filters={"AND": [{"agent_id": "agent-123"}]},
        threshold=0.5,
    )

    assert len(results) == 2
    assert results[0]["content"] == "Result 1"


def test_search_method_with_agent_id_and_user_id():
    mem0_storage = Mem0Storage(type="external", config={"agent_id": "agent-123", "user_id": "user-123"})
    mock_results = {"results": [{"score": 0.9, "content": "Result 1"}, {"score": 0.4, "content": "Result 2"}]}
    mem0_storage.memory.search = MagicMock(return_value=mock_results)

    results = mem0_storage.search("test query", limit=5, score_threshold=0.5)

    mem0_storage.memory.search.assert_called_once_with(
        query="test query",
        limit=5,
        user_id='user-123',
        filters={"OR": [{"user_id": "user-123"}, {"agent_id": "agent-123"}]},
        threshold=0.5,
    )

    assert len(results) == 2
    assert results[0]["content"] == "Result 1"


def test_search_method_with_memory_oss_missing_params():
    """Test search method handles missing parameters gracefully when using Memory (OSS)"""
    config = {
        "user_id": "test_user",
        "local_mem0_config": {
            "vector_store": {"provider": "mock_vector_store"},
            "llm": {"provider": "mock_llm"},
        }
    }
    
    mock_memory = MagicMock(spec=Memory)
    mock_results = {"results": [{"score": 0.9, "content": "Result 1"}]}
    mock_memory.search = MagicMock(return_value=mock_results)
    
    with patch("mem0.memory.main.Memory.from_config", return_value=mock_memory):
        mem0_storage = Mem0Storage(type="external", config=config)
        
        results = mem0_storage.search("test query", limit=5, score_threshold=0.5)
        
        mock_memory.search.assert_called_once_with(
            query="test query",
            limit=5,
            user_id="test_user",
            filters={"AND": [{"user_id": "test_user"}]},
            threshold=0.5
        )
        
        assert len(results) == 1
        assert results[0]["content"] == "Result 1"


def test_search_method_with_memory_oss_no_optional_params():
    """Test search method works when no optional parameters are present"""
    mock_memory = MagicMock(spec=Memory)
    mock_results = {"results": []}
    mock_memory.search = MagicMock(return_value=mock_results)
    
    with patch("mem0.memory.main.Memory", return_value=mock_memory):
        mem0_storage = Mem0Storage(type="external", config={})
        
        results = mem0_storage.search("test query")
        
        mock_memory.search.assert_called_once_with(
            query="test query",
            limit=3,
            filters={},
            threshold=0.35
        )
        
        assert len(results) == 0
