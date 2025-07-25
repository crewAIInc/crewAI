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
        [{'role': 'assistant' , 'content': test_value}],
        infer=True,
        metadata={"type": "short_term", "key": "value"},
        user_id="test_user"
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
        user_id="test_user",
        version="v2",
        run_id="my_run_id",
        includes="include1",
        excludes="exclude1",
        output_format='v1.1'
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
        filters={'AND': [{'user_id': 'test_user'}, {'run_id': 'my_run_id'}]},
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
        filters={'AND': [{'user_id': 'test_user'}, {'run_id': 'my_run_id'}]},
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


def test_save_with_agent_id_from_metadata(mem0_storage_with_memory_client_using_config_from_crew):
    """Test that agent_id is extracted from metadata and used in save operation"""
    mem0_storage = mem0_storage_with_memory_client_using_config_from_crew
    mem0_storage.memory.add = MagicMock()
    
    test_value = "This is a test memory from agent"
    test_metadata = {"agent": "test_agent_123", "key": "value"}
    
    mem0_storage.save(test_value, test_metadata)
    
    call_args = mem0_storage.memory.add.call_args
    assert "agent_id" in call_args[1]
    assert call_args[1]["agent_id"] == "test_agent_123"


def test_search_with_agent_id_in_config(mem0_storage_with_memory_client_using_config_from_crew):
    """Test search method includes agent_id when present in config"""
    mem0_storage = mem0_storage_with_memory_client_using_config_from_crew
    mem0_storage.config["agent_id"] = "test_agent_456"
    
    mock_results = {"results": [{"score": 0.9, "content": "Result 1"}]}
    mem0_storage.memory.search = MagicMock(return_value=mock_results)

    mem0_storage.search("test query")

    call_args = mem0_storage.memory.search.call_args[1]
    assert "agent_id" in call_args
    assert call_args["agent_id"] == "test_agent_456"


def test_filter_uses_or_logic_with_both_user_and_agent_id(mock_mem0_memory_client):
    """Test that filter uses OR logic when both user_id and agent_id are present"""
    crew = MockCrew(memory_config={"provider": "mem0", "config": {"user_id": "test_user", "api_key": "ABCDEFGH"}})
    
    with patch.object(MemoryClient, "__new__", return_value=mock_mem0_memory_client):
        mem0_storage = Mem0Storage(type="external", crew=crew)
        mem0_storage.config["user_id"] = "test_user"
        mem0_storage.config["agent_id"] = "test_agent"
        
        filter_result = mem0_storage._create_filter_for_search()
        
        assert "AND" in filter_result
        assert len(filter_result["AND"]) == 1
        assert "OR" in filter_result["AND"][0]
        or_conditions = filter_result["AND"][0]["OR"]
        assert {"user_id": "test_user"} in or_conditions
        assert {"agent_id": "test_agent"} in or_conditions


def test_filter_uses_single_condition_with_only_agent_id(mock_mem0_memory_client):
    """Test that filter uses single condition when only agent_id is present"""
    crew = MockCrew(memory_config={"provider": "mem0", "config": {"api_key": "ABCDEFGH"}})
    
    with patch.object(MemoryClient, "__new__", return_value=mock_mem0_memory_client):
        mem0_storage = Mem0Storage(type="external", crew=crew)
        mem0_storage.config["agent_id"] = "test_agent"
        
        filter_result = mem0_storage._create_filter_for_search()
        
        assert "AND" in filter_result
        assert {"agent_id": "test_agent"} in filter_result["AND"]


def test_set_agent_id_method(mock_mem0_memory_client):
    """Test the set_agent_id method"""
    crew = MockCrew(memory_config={"provider": "mem0", "config": {"api_key": "ABCDEFGH"}})
    
    with patch.object(MemoryClient, "__new__", return_value=mock_mem0_memory_client):
        mem0_storage = Mem0Storage(type="external", crew=crew)
        mem0_storage.set_agent_id("new_agent_123")
        
        assert mem0_storage.config["agent_id"] == "new_agent_123"


def test_save_with_both_user_id_and_agent_id(mem0_storage_with_memory_client_using_config_from_crew):
    """Test save method with both user_id and agent_id"""
    mem0_storage = mem0_storage_with_memory_client_using_config_from_crew
    mem0_storage.memory.add = MagicMock()
    
    test_value = "This is a test memory"
    test_metadata = {"agent": "test_agent_789", "key": "value"}
    
    mem0_storage.save(test_value, test_metadata)
    
    call_args = mem0_storage.memory.add.call_args[1]
    assert "user_id" in call_args
    assert call_args["user_id"] == "test_user"
    assert "agent_id" in call_args
    assert call_args["agent_id"] == "test_agent_789"


def test_save_without_agent_id_in_metadata(mem0_storage_with_memory_client_using_config_from_crew):
    """Test save method when no agent_id is in metadata"""
    mem0_storage = mem0_storage_with_memory_client_using_config_from_crew
    mem0_storage.memory.add = MagicMock()
    
    test_value = "This is a test memory"
    test_metadata = {"key": "value"}
    
    mem0_storage.save(test_value, test_metadata)
    
    call_args = mem0_storage.memory.add.call_args[1]
    assert "user_id" in call_args
    assert call_args["user_id"] == "test_user"
    assert "agent_id" not in call_args
