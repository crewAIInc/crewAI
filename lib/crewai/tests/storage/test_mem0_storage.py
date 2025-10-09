from unittest.mock import MagicMock, patch

import pytest
from crewai.memory.storage.mem0_storage import Mem0Storage
from mem0 import Memory, MemoryClient


# Define the class (if not already defined)
class MockCrew:
    def __init__(self):
        self.agents = [MagicMock(role="Test Agent")]


# Test data constants
SYSTEM_CONTENT = (
    "You are Friendly chatbot assistant. You are a kind and "
    "knowledgeable chatbot assistant. You excel at understanding user needs, "
    "providing helpful responses, and maintaining engaging conversations. "
    "You remember previous interactions to provide a personalized experience.\n"
    "Your personal goal is: Engage in useful and interesting conversations "
    "with users while remembering context.\n"
    "To give my best complete final answer to the task respond using the exact "
    "following format:\n\n"
    "Thought: I now can give a great answer\n"
    "Final Answer: Your final answer must be the great and the most complete "
    "as possible, it must be outcome described.\n\n"
    "I MUST use these formats, my job depends on it!"
)

USER_CONTENT = (
    "\nCurrent Task: Respond to user conversation. User message: "
    "What do you know about me?\n\n"
    "This is the expected criteria for your final answer: Contextually "
    "appropriate, helpful, and friendly response.\n"
    "you MUST return the actual complete content as the final answer, "
    "not a summary.\n\n"
    "# Useful context: \nExternal memories:\n"
    "- User is from India\n"
    "- User is interested in the solar system\n"
    "- User name is Vidit Ostwal\n"
    "- User is interested in French cuisine\n\n"
    "Begin! This is VERY important to you, use the tools available and give "
    "your best Final Answer, your job depends on it!\n\n"
    "Thought:"
)

ASSISTANT_CONTENT = (
    "I now can give a great answer  \n"
    "Final Answer: Hi Vidit! From our previous conversations, I know you're "
    "from India and have a great interest in the solar system. It's fascinating "
    "to explore the wonders of space, isn't it? Also, I remember you have a "
    "passion for French cuisine, which has so many delightful dishes to explore. "
    "If there's anything specific you'd like to discuss or learn about—whether "
    "it's about the solar system or some great French recipes—feel free to let "
    "me know! I'm here to help."
)

TEST_DESCRIPTION = (
    "Respond to user conversation. User message: What do you know about me?"
)

# Extracted content (after processing by _get_user_message and _get_assistant_message)
EXTRACTED_USER_CONTENT = "What do you know about me?"
EXTRACTED_ASSISTANT_CONTENT = (
    "Hi Vidit! From our previous conversations, I know you're "
    "from India and have a great interest in the solar system. It's fascinating "
    "to explore the wonders of space, isn't it? Also, I remember you have a "
    "passion for French cuisine, which has so many delightful dishes to explore. "
    "If there's anything specific you'd like to discuss or learn about—whether "
    "it's about the solar system or some great French recipes—feel free to let "
    "me know! I'm here to help."
)


@pytest.fixture
def mock_mem0_memory():
    """Fixture to create a mock Memory instance"""
    return MagicMock(spec=Memory)


@pytest.fixture
def mem0_storage_with_mocked_config(mock_mem0_memory):
    """Fixture to create a Mem0Storage instance with mocked dependencies"""

    # Patch the Memory class to return our mock
    with patch(
        "mem0.Memory.from_config", return_value=mock_mem0_memory
    ) as mock_from_config:
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

        # Parameters like run_id, includes, and excludes doesn't matter in Memory OSS
        crew = MockCrew()

        embedder_config = {
            "user_id": "test_user",
            "local_mem0_config": config,
            "run_id": "my_run_id",
            "includes": "include1",
            "excludes": "exclude1",
            "infer": True,
        }

        mem0_storage = Mem0Storage(type="short_term", crew=crew, config=embedder_config)
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
    return MagicMock(spec=MemoryClient)


@pytest.fixture
def mem0_storage_with_memory_client_using_config_from_crew(mock_mem0_memory_client):
    """Fixture to create a Mem0Storage instance with mocked dependencies"""

    # We need to patch the MemoryClient before it's instantiated
    with patch.object(MemoryClient, "__new__", return_value=mock_mem0_memory_client):
        crew = MockCrew()

        embedder_config = {
            "user_id": "test_user",
            "api_key": "ABCDEFGH",
            "org_id": "my_org_id",
            "project_id": "my_project_id",
            "run_id": "my_run_id",
            "includes": "include1",
            "excludes": "exclude1",
            "infer": True,
        }

        return Mem0Storage(type="short_term", crew=crew, config=embedder_config)


@pytest.fixture
def mem0_storage_with_memory_client_using_explictly_config(
    mock_mem0_memory_client, mock_mem0_memory
):
    """Fixture to create a Mem0Storage instance with mocked dependencies"""

    # We need to patch both MemoryClient and Memory to prevent actual initialization
    with (
        patch.object(MemoryClient, "__new__", return_value=mock_mem0_memory_client),
        patch.object(Memory, "__new__", return_value=mock_mem0_memory),
    ):
        crew = MockCrew()
        new_config = {"provider": "mem0", "config": {"api_key": "new-api-key"}}

        return Mem0Storage(type="short_term", crew=crew, config=new_config)


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
        {
            "lifestyle_management_concerns": (
                "Tracks daily routines, habits, hobbies and interests "
                "including cooking, time management and work-life balance"
            )
        },
    ]

    crew = MockCrew()

    config = {
        "user_id": "test_user",
        "api_key": "ABCDEFGH",
        "org_id": "my_org_id",
        "project_id": "my_project_id",
        "custom_categories": new_categories,
    }

    with patch.object(MemoryClient, "__new__", return_value=mock_mem0_memory_client):
        _ = Mem0Storage(type="short_term", crew=crew, config=config)

    mock_mem0_memory_client.update_project.assert_called_once_with(
        custom_categories=new_categories
    )


def test_save_method_with_memory_oss(mem0_storage_with_mocked_config):
    """Test save method for different memory types"""
    mem0_storage, _, _ = mem0_storage_with_mocked_config
    mem0_storage.memory.add = MagicMock()

    # Test short_term memory type (already set in fixture)
    test_value = "This is a test memory"
    test_metadata = {
        "description": TEST_DESCRIPTION,
        "messages": [
            {"role": "system", "content": SYSTEM_CONTENT},
            {"role": "user", "content": USER_CONTENT},
            {"role": "assistant", "content": ASSISTANT_CONTENT},
        ],
        "agent": "Friendly chatbot assistant",
    }

    mem0_storage.save(test_value, test_metadata)

    mem0_storage.memory.add.assert_called_once_with(
        [
            {"role": "user", "content": EXTRACTED_USER_CONTENT},
            {
                "role": "assistant",
                "content": EXTRACTED_ASSISTANT_CONTENT,
            },
        ],
        infer=True,
        metadata={
            "type": "short_term",
            "description": TEST_DESCRIPTION,
            "agent": "Friendly chatbot assistant",
        },
        run_id="my_run_id",
        user_id="test_user",
        agent_id="Test_Agent",
    )


def test_save_method_with_multiple_agents(mem0_storage_with_mocked_config):
    mem0_storage, _, _ = mem0_storage_with_mocked_config
    mem0_storage.crew.agents = [
        MagicMock(role="Test Agent"),
        MagicMock(role="Test Agent 2"),
        MagicMock(role="Test Agent 3"),
    ]
    mem0_storage.memory.add = MagicMock()

    test_value = "This is a test memory"
    test_metadata = {
        "description": TEST_DESCRIPTION,
        "messages": [
            {"role": "system", "content": SYSTEM_CONTENT},
            {"role": "user", "content": USER_CONTENT},
            {"role": "assistant", "content": ASSISTANT_CONTENT},
        ],
        "agent": "Friendly chatbot assistant",
    }

    mem0_storage.save(test_value, test_metadata)

    mem0_storage.memory.add.assert_called_once_with(
        [
            {"role": "user", "content": EXTRACTED_USER_CONTENT},
            {
                "role": "assistant",
                "content": EXTRACTED_ASSISTANT_CONTENT,
            },
        ],
        infer=True,
        metadata={
            "type": "short_term",
            "description": TEST_DESCRIPTION,
            "agent": "Friendly chatbot assistant",
        },
        run_id="my_run_id",
        user_id="test_user",
        agent_id="Test_Agent_Test_Agent_2_Test_Agent_3",
    )


def test_save_method_with_memory_client(
    mem0_storage_with_memory_client_using_config_from_crew,
):
    """Test save method for different memory types"""
    mem0_storage = mem0_storage_with_memory_client_using_config_from_crew
    mem0_storage.memory.add = MagicMock()

    # Test short_term memory type (already set in fixture)
    test_value = "This is a test memory"
    test_metadata = {
        "description": TEST_DESCRIPTION,
        "messages": [
            {"role": "system", "content": SYSTEM_CONTENT},
            {"role": "user", "content": USER_CONTENT},
            {"role": "assistant", "content": ASSISTANT_CONTENT},
        ],
        "agent": "Friendly chatbot assistant",
    }

    mem0_storage.save(test_value, test_metadata)

    mem0_storage.memory.add.assert_called_once_with(
        [
            {"role": "user", "content": EXTRACTED_USER_CONTENT},
            {
                "role": "assistant",
                "content": EXTRACTED_ASSISTANT_CONTENT,
            },
        ],
        infer=True,
        metadata={
            "type": "short_term",
            "description": TEST_DESCRIPTION,
            "agent": "Friendly chatbot assistant",
        },
        version="v2",
        run_id="my_run_id",
        includes="include1",
        excludes="exclude1",
        output_format="v1.1",
        user_id="test_user",
        agent_id="Test_Agent",
    )


def test_search_method_with_memory_oss(mem0_storage_with_mocked_config):
    """Test search method for different memory types"""
    mem0_storage, _, _ = mem0_storage_with_mocked_config
    mock_results = {
        "results": [
            {"score": 0.9, "memory": "Result 1"},
            {"score": 0.4, "memory": "Result 2"},
        ]
    }
    mem0_storage.memory.search = MagicMock(return_value=mock_results)

    results = mem0_storage.search("test query", limit=5, score_threshold=0.5)

    mem0_storage.memory.search.assert_called_once_with(
        query="test query",
        limit=5,
        user_id="test_user",
        filters={"AND": [{"run_id": "my_run_id"}]},
        threshold=0.5,
    )

    assert len(results) == 2
    assert results[0]["content"] == "Result 1"


def test_search_method_with_memory_client(
    mem0_storage_with_memory_client_using_config_from_crew,
):
    """Test search method for different memory types"""
    mem0_storage = mem0_storage_with_memory_client_using_config_from_crew
    mock_results = {
        "results": [
            {"score": 0.9, "memory": "Result 1"},
            {"score": 0.4, "memory": "Result 2"},
        ]
    }
    mem0_storage.memory.search = MagicMock(return_value=mock_results)

    results = mem0_storage.search("test query", limit=5, score_threshold=0.5)

    mem0_storage.memory.search.assert_called_once_with(
        query="test query",
        limit=5,
        metadata={"type": "short_term"},
        user_id="test_user",
        version="v2",
        run_id="my_run_id",
        output_format="v1.1",
        filters={"AND": [{"run_id": "my_run_id"}]},
        threshold=0.5,
    )

    assert len(results) == 2
    assert results[0]["content"] == "Result 1"


def test_mem0_storage_default_infer_value(mock_mem0_memory_client):
    """Test that Mem0Storage sets infer=True by default for short_term memory."""
    with patch.object(MemoryClient, "__new__", return_value=mock_mem0_memory_client):
        crew = MockCrew()

        config = {"user_id": "test_user", "api_key": "ABCDEFGH"}

        mem0_storage = Mem0Storage(type="short_term", crew=crew, config=config)
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
            [{"role": "assistant", "content": "test memory"}],
            infer=True,
            metadata={"type": "external", "key": "value"},
            agent_id="agent-123",
        )


def test_search_method_with_agent_entity():
    config = {
        "agent_id": "agent-123",
    }

    mock_memory = MagicMock(spec=Memory)
    mock_results = {
        "results": [
            {"score": 0.9, "memory": "Result 1"},
            {"score": 0.4, "memory": "Result 2"},
        ]
    }

    with patch.object(Memory, "__new__", return_value=mock_memory):
        mem0_storage = Mem0Storage(type="external", config=config)

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
    mock_memory = MagicMock(spec=Memory)
    mock_results = {
        "results": [
            {"score": 0.9, "memory": "Result 1"},
            {"score": 0.4, "memory": "Result 2"},
        ]
    }

    with patch.object(Memory, "__new__", return_value=mock_memory):
        mem0_storage = Mem0Storage(
            type="external", config={"agent_id": "agent-123", "user_id": "user-123"}
        )

        mem0_storage.memory.search = MagicMock(return_value=mock_results)
        results = mem0_storage.search("test query", limit=5, score_threshold=0.5)

        mem0_storage.memory.search.assert_called_once_with(
            query="test query",
            limit=5,
            user_id="user-123",
            filters={"OR": [{"user_id": "user-123"}, {"agent_id": "agent-123"}]},
            threshold=0.5,
        )

        assert len(results) == 2
        assert results[0]["content"] == "Result 1"
