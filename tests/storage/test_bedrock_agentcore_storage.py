import json
import pytest
from unittest.mock import MagicMock, patch

from crewai.memory.storage.bedrock_agentcore_storage import (
    BedrockAgentCoreStorage,
    BedrockAgentCoreConfig,
    BedrockAgentCoreStrategyConfig,
)


class TestBedrockAgentCoreStrategyConfig:
    """Test cases for BedrockAgentCoreStrategyConfig validation."""

    def test_valid_strategy_config(self):
        """Test creating a valid strategy configuration."""
        config = BedrockAgentCoreStrategyConfig(
            name="test_strategy",
            namespaces=["/test/namespace1", "/test/namespace2"],
            strategy_id="strategy-123",
        )
        assert config.name == "test_strategy"
        assert config.namespaces == ["/test/namespace1", "/test/namespace2"]
        assert config.strategy_id == "strategy-123"

    def test_empty_name_validation(self):
        """Test that empty strategy name raises validation error."""
        with pytest.raises(ValueError, match="Strategy name cannot be empty"):
            BedrockAgentCoreStrategyConfig(
                name="", namespaces=["/test/namespace"], strategy_id="strategy-123"
            )

    def test_whitespace_name_validation(self):
        """Test that whitespace-only strategy name raises validation error."""
        with pytest.raises(ValueError, match="Strategy name cannot be empty"):
            BedrockAgentCoreStrategyConfig(
                name="   ", namespaces=["/test/namespace"], strategy_id="strategy-123"
            )

    def test_empty_namespaces_validation(self):
        """Test that empty namespaces list raises validation error."""
        with pytest.raises(
            ValueError, match="Strategy namespaces must be a non-empty list"
        ):
            BedrockAgentCoreStrategyConfig(
                name="test_strategy", namespaces=[], strategy_id="strategy-123"
            )

    def test_none_namespaces_validation(self):
        """Test that None namespaces raises validation error."""
        with pytest.raises(ValueError):
            BedrockAgentCoreStrategyConfig(
                name="test_strategy",
                namespaces=None,  # type: ignore
                strategy_id="strategy-123",
            )

    def test_empty_namespace_item_validation(self):
        """Test that empty namespace items raise validation error."""
        with pytest.raises(ValueError, match="Namespace cannot be empty"):
            BedrockAgentCoreStrategyConfig(
                name="test_strategy",
                namespaces=["/valid/namespace", ""],
                strategy_id="strategy-123",
            )

    def test_name_trimming(self):
        """Test that strategy name is trimmed of whitespace."""
        config = BedrockAgentCoreStrategyConfig(
            name="  test_strategy  ",
            namespaces=["/test/namespace"],
            strategy_id="strategy-123",
        )
        assert config.name == "test_strategy"

    def test_namespace_trimming(self):
        """Test that namespaces are trimmed of whitespace."""
        config = BedrockAgentCoreStrategyConfig(
            name="test_strategy",
            namespaces=["  /test/namespace1  ", "  /test/namespace2  "],
            strategy_id="strategy-123",
        )
        assert config.namespaces == ["/test/namespace1", "/test/namespace2"]


class TestBedrockAgentCoreConfig:
    """Test cases for BedrockAgentCoreConfig validation."""

    def test_valid_config(self):
        """Test creating a valid AgentCore configuration."""
        config = BedrockAgentCoreConfig(
            memory_id="memory-123",
            actor_id="actor-456",
            session_id="session-789",
            region_name="us-west-2",
        )
        assert config.memory_id == "memory-123"
        assert config.actor_id == "actor-456"
        assert config.session_id == "session-789"
        assert config.region_name == "us-west-2"
        assert config.run_id is None
        assert config.strategies == []

    def test_default_region(self):
        """Test that default region is us-east-1."""
        config = BedrockAgentCoreConfig(
            memory_id="memory-123", actor_id="actor-456", session_id="session-789"
        )
        assert config.region_name == "us-east-1"

    def test_empty_required_fields_validation(self):
        """Test that empty required fields raise validation errors."""
        with pytest.raises(ValueError, match="Required ID fields cannot be empty"):
            BedrockAgentCoreConfig(
                memory_id="", actor_id="actor-456", session_id="session-789"
            )

        with pytest.raises(ValueError, match="Required ID fields cannot be empty"):
            BedrockAgentCoreConfig(
                memory_id="memory-123", actor_id="", session_id="session-789"
            )

        with pytest.raises(ValueError, match="Required ID fields cannot be empty"):
            BedrockAgentCoreConfig(
                memory_id="memory-123", actor_id="actor-456", session_id=""
            )

    def test_whitespace_required_fields_validation(self):
        """Test that whitespace-only required fields raise validation errors."""
        with pytest.raises(ValueError, match="Required ID fields cannot be empty"):
            BedrockAgentCoreConfig(
                memory_id="   ", actor_id="actor-456", session_id="session-789"
            )

    def test_empty_region_validation(self):
        """Test that empty region raises validation error."""
        with pytest.raises(ValueError, match="Region name cannot be empty"):
            BedrockAgentCoreConfig(
                memory_id="memory-123",
                actor_id="actor-456",
                session_id="session-789",
                region_name="",
            )

    def test_duplicate_strategy_names_validation(self):
        """Test that duplicate strategy names raise validation error."""
        strategy1 = BedrockAgentCoreStrategyConfig(
            name="duplicate_name", namespaces=["/namespace1"], strategy_id="strategy-1"
        )
        strategy2 = BedrockAgentCoreStrategyConfig(
            name="duplicate_name", namespaces=["/namespace2"], strategy_id="strategy-2"
        )

        with pytest.raises(ValueError, match="Strategy names must be unique"):
            BedrockAgentCoreConfig(
                memory_id="memory-123",
                actor_id="actor-456",
                session_id="session-789",
                strategies=[strategy1, strategy2],
            )

    def test_field_trimming(self):
        """Test that fields are trimmed of whitespace."""
        config = BedrockAgentCoreConfig(
            memory_id="  memory-123  ",
            actor_id="  actor-456  ",
            session_id="  session-789  ",
            region_name="  us-west-2  ",
        )
        assert config.memory_id == "memory-123"
        assert config.actor_id == "actor-456"
        assert config.session_id == "session-789"
        assert config.region_name == "us-west-2"

    def test_config_with_strategies(self):
        """Test configuration with strategies."""
        strategy = BedrockAgentCoreStrategyConfig(
            name="test_strategy",
            namespaces=["/test/namespace"],
            strategy_id="strategy-123",
        )
        config = BedrockAgentCoreConfig(
            memory_id="memory-123",
            actor_id="actor-456",
            session_id="session-789",
            strategies=[strategy],
        )
        assert len(config.strategies) == 1
        assert config.strategies[0].name == "test_strategy"

    def test_extra_fields_forbidden(self):
        """Test that extra fields are forbidden."""
        with pytest.raises(ValueError):
            BedrockAgentCoreConfig(
                memory_id="memory-123",
                actor_id="actor-456",
                session_id="session-789",
                extra_field="not_allowed",  # type: ignore
            )


class TestBedrockAgentCoreStorage:
    """Test cases for BedrockAgentCoreStorage."""

    @pytest.fixture
    def mock_memory_client(self):
        """Fixture to create a mock MemoryClient."""
        return MagicMock()

    @pytest.fixture
    def basic_config(self):
        """Fixture for basic AgentCore configuration."""
        return BedrockAgentCoreConfig(
            memory_id="memory-123",
            actor_id="actor-456",
            session_id="session-789",
            region_name="us-west-2",
        )

    @pytest.fixture
    def config_with_strategies(self):
        """Fixture for AgentCore configuration with strategies."""
        strategy1 = BedrockAgentCoreStrategyConfig(
            name="user_preferences",
            namespaces=["/preferences/actor-456"],
            strategy_id="strategy-pref-123",
        )
        strategy2 = BedrockAgentCoreStrategyConfig(
            name="semantic_facts",
            namespaces=["/facts/actor-456/session-789"],
            strategy_id="strategy-facts-456",
        )
        return BedrockAgentCoreConfig(
            memory_id="memory-123",
            actor_id="actor-456",
            session_id="session-789",
            region_name="us-west-2",
            strategies=[strategy1, strategy2],
        )

    def test_invalid_type_validation(self, basic_config):
        """Test that invalid memory type raises validation error."""
        with pytest.raises(
            ValueError, match="Invalid type 'invalid' for AgentCoreStorage"
        ):
            BedrockAgentCoreStorage(type="invalid", config=basic_config)

    @patch("crewai.memory.storage.bedrock_agentcore_storage.MemoryClient")
    def test_initialization_success(self, mock_memory_client_class, basic_config):
        """Test successful initialization of BedrockAgentCoreStorage."""
        mock_client = MagicMock()
        mock_memory_client_class.return_value = mock_client

        storage = BedrockAgentCoreStorage(type="external", config=basic_config)

        assert storage.memory_type == "external"
        assert storage.config == basic_config
        assert storage.memory_client == mock_client
        mock_memory_client_class.assert_called_once_with(region_name="us-west-2")

    @patch("crewai.memory.storage.bedrock_agentcore_storage.MemoryClient", None)
    def test_initialization_missing_client(self, basic_config):
        """Test initialization failure when MemoryClient is not available."""
        with pytest.raises(
            RuntimeError, match="bedrock-agentcore-sdk-python is not available"
        ):
            BedrockAgentCoreStorage(type="external", config=basic_config)

    @patch("crewai.memory.storage.bedrock_agentcore_storage.MemoryClient")
    def test_initialization_client_error(self, mock_memory_client_class, basic_config):
        """Test initialization failure when MemoryClient raises an error."""
        mock_memory_client_class.side_effect = Exception("Client initialization failed")

        with pytest.raises(
            ValueError, match="Failed to initialize Bedrock AgentCore Memory client"
        ):
            BedrockAgentCoreStorage(type="external", config=basic_config)

    @patch("crewai.memory.storage.bedrock_agentcore_storage.MemoryClient")
    def test_strategy_namespace_setup(
        self, mock_memory_client_class, config_with_strategies
    ):
        """Test that strategy namespaces are set up correctly."""
        mock_memory_client_class.return_value = MagicMock()

        storage = BedrockAgentCoreStorage(
            type="external", config=config_with_strategies
        )

        assert len(storage.strategy_namespace_map) == 2
        assert storage.strategy_namespace_map["user_preferences"] == [
            "/preferences/actor-456"
        ]
        assert storage.strategy_namespace_map["semantic_facts"] == [
            "/facts/actor-456/session-789"
        ]

    @patch("crewai.memory.storage.bedrock_agentcore_storage.MemoryClient")
    def test_get_namespaces_for_strategy(
        self, mock_memory_client_class, config_with_strategies
    ):
        """Test getting namespaces for a specific strategy."""
        mock_memory_client_class.return_value = MagicMock()

        storage = BedrockAgentCoreStorage(
            type="external", config=config_with_strategies
        )

        namespaces = storage.get_namespaces_for_strategy("user_preferences")
        assert namespaces == ["/preferences/actor-456"]

        namespaces = storage.get_namespaces_for_strategy("nonexistent")
        assert namespaces is None

    @patch("crewai.memory.storage.bedrock_agentcore_storage.MemoryClient")
    def test_get_all_unique_namespaces(
        self, mock_memory_client_class, config_with_strategies
    ):
        """Test getting all unique namespaces."""
        mock_memory_client_class.return_value = MagicMock()

        storage = BedrockAgentCoreStorage(
            type="external", config=config_with_strategies
        )

        all_namespaces = storage.get_all_unique_namespaces()
        assert set(all_namespaces) == {
            "/preferences/actor-456",
            "/facts/actor-456/session-789",
        }

    def test_resolve_namespace_template(self):
        """Test namespace template resolution utility."""
        template = (
            "/strategies/{memoryStrategyId}/actors/{actorId}/sessions/{sessionId}"
        )
        resolved = BedrockAgentCoreStorage.resolve_namespace_template(
            template, "actor-456", "session-789", "strategy-123"
        )
        expected = "/strategies/strategy-123/actors/actor-456/sessions/session-789"
        assert resolved == expected

    @patch("crewai.memory.storage.bedrock_agentcore_storage.MemoryClient")
    def test_save_single_message(self, mock_memory_client_class, basic_config):
        """Test saving a single message."""
        mock_client = MagicMock()
        mock_memory_client_class.return_value = mock_client
        mock_client.create_event.return_value = {"eventId": "event-123"}

        storage = BedrockAgentCoreStorage(type="external", config=basic_config)

        test_value = "This is a test message"
        test_metadata = {"type": "test", "agent": "test_agent"}

        storage.save(test_value, test_metadata)

        mock_client.create_event.assert_called_once_with(
            memory_id="memory-123",
            actor_id="actor-456",
            session_id="session-789",
            messages=[("This is a test message", "ASSISTANT")],
        )

    @patch("crewai.memory.storage.bedrock_agentcore_storage.MemoryClient")
    def test_save_batch_messages(self, mock_memory_client_class, basic_config):
        """Test saving a batch of messages."""
        mock_client = MagicMock()
        mock_memory_client_class.return_value = mock_client
        mock_client.create_event.return_value = {"eventId": "event-123"}

        storage = BedrockAgentCoreStorage(type="external", config=basic_config)

        test_messages = [
            {"content": "User message", "metadata": {"role": "user"}},
            {"content": "Assistant response", "metadata": {"role": "assistant"}},
            {"content": "Tool output", "metadata": {"role": "tool"}},
        ]

        storage.save(test_messages)

        expected_messages = [
            ("User message", "USER"),
            ("Assistant response", "ASSISTANT"),
            ("Tool output", "TOOL"),
        ]

        mock_client.create_event.assert_called_once_with(
            memory_id="memory-123",
            actor_id="actor-456",
            session_id="session-789",
            messages=expected_messages,
        )

    @patch("crewai.memory.storage.bedrock_agentcore_storage.MemoryClient")
    def test_save_batch_size_limit(self, mock_memory_client_class, basic_config):
        """Test that batch size limit is enforced."""
        mock_memory_client_class.return_value = MagicMock()

        storage = BedrockAgentCoreStorage(type="external", config=basic_config)

        # Create a batch with more than 100 messages
        large_batch = [{"content": f"Message {i}", "metadata": {}} for i in range(101)]

        with pytest.raises(
            ValueError, match="Cannot process more than 100 messages in a batch"
        ):
            storage.save(large_batch)

    @patch("crewai.memory.storage.bedrock_agentcore_storage.MemoryClient")
    def test_save_client_error(self, mock_memory_client_class, basic_config):
        """Test save method when client raises an error."""
        mock_client = MagicMock()
        mock_memory_client_class.return_value = mock_client
        mock_client.create_event.side_effect = Exception("Client error")

        storage = BedrockAgentCoreStorage(type="external", config=basic_config)

        with pytest.raises(Exception, match="Client error"):
            storage.save("test message")

    @patch("crewai.memory.storage.bedrock_agentcore_storage.MemoryClient")
    def test_save_uninitialized_client(self, mock_memory_client_class, basic_config):
        """Test save method when client is not initialized."""
        mock_memory_client_class.return_value = MagicMock()

        storage = BedrockAgentCoreStorage(type="external", config=basic_config)
        storage.memory_client = None  # Simulate uninitialized client

        with pytest.raises(
            RuntimeError, match="AgentCore memory client not initialized"
        ):
            storage.save("test message")

    @patch("crewai.memory.storage.bedrock_agentcore_storage.MemoryClient")
    def test_search_with_strategies(
        self, mock_memory_client_class, config_with_strategies
    ):
        """Test search method with configured strategies."""
        mock_client = MagicMock()
        mock_memory_client_class.return_value = mock_client

        # Mock search results for different namespaces
        mock_client.retrieve_memories.side_effect = [
            [  # Results for first namespace
                {
                    "memoryRecordId": "record-1",
                    "content": {"text": "User preference data"},
                    "score": 0.9,
                    "namespaces": ["/preferences/actor-456"],
                    "createdAt": "2024-01-01T00:00:00Z",
                    "memoryStrategyId": "strategy-pref-123",
                }
            ],
            [  # Results for second namespace
                {
                    "memoryRecordId": "record-2",
                    "content": {"text": "Semantic fact data"},
                    "score": 0.8,
                    "namespaces": ["/facts/actor-456/session-789"],
                    "createdAt": "2024-01-01T01:00:00Z",
                    "memoryStrategyId": "strategy-facts-456",
                }
            ],
        ]

        storage = BedrockAgentCoreStorage(
            type="external", config=config_with_strategies
        )

        results = storage.search("test query", limit=5, score_threshold=0.5)

        assert len(results) == 2
        assert results[0]["id"] == "record-1"
        assert results[0]["context"] == "User preference data"
        assert results[0]["score"] == 0.9
        assert results[1]["id"] == "record-2"
        assert results[1]["context"] == "Semantic fact data"
        assert results[1]["score"] == 0.8

        # Verify that retrieve_memories was called for each namespace
        assert mock_client.retrieve_memories.call_count == 2

    @patch("crewai.memory.storage.bedrock_agentcore_storage.MemoryClient")
    def test_search_no_strategies(self, mock_memory_client_class, basic_config):
        """Test search method when no strategies are configured."""
        mock_memory_client_class.return_value = MagicMock()

        storage = BedrockAgentCoreStorage(type="external", config=basic_config)

        results = storage.search("test query")

        assert results == []

    @patch("crewai.memory.storage.bedrock_agentcore_storage.MemoryClient")
    def test_search_score_threshold_filtering(
        self, mock_memory_client_class, config_with_strategies
    ):
        """Test that search results are filtered by score threshold."""
        mock_client = MagicMock()
        mock_memory_client_class.return_value = mock_client

        # Mock results with different scores - need to return different results for each namespace
        mock_client.retrieve_memories.side_effect = [
            [  # Results for first namespace (/preferences/actor-456)
                {
                    "memoryRecordId": "record-1",
                    "content": {"text": "High score result"},
                    "score": 0.9,
                    "namespaces": ["/preferences/actor-456"],
                    "createdAt": "2024-01-01T00:00:00Z",
                },
                {
                    "memoryRecordId": "record-2",
                    "content": {"text": "Low score result"},
                    "score": 0.2,  # Below threshold
                    "namespaces": ["/preferences/actor-456"],
                    "createdAt": "2024-01-01T01:00:00Z",
                },
            ],
            [],  # No results for second namespace (/facts/actor-456/session-789)
        ]

        storage = BedrockAgentCoreStorage(
            type="external", config=config_with_strategies
        )

        results = storage.search("test query", score_threshold=0.5)

        # Only the high score result should be returned
        assert len(results) == 1
        assert results[0]["context"] == "High score result"

    @patch("crewai.memory.storage.bedrock_agentcore_storage.MemoryClient")
    def test_search_uninitialized_client(
        self, mock_memory_client_class, config_with_strategies
    ):
        """Test search method when client is not initialized."""
        mock_memory_client_class.return_value = MagicMock()

        storage = BedrockAgentCoreStorage(
            type="external", config=config_with_strategies
        )
        storage.memory_client = None  # Simulate uninitialized client

        with pytest.raises(
            RuntimeError, match="AgentCore memory client not initialized"
        ):
            storage.search("test query")

    @patch("crewai.memory.storage.bedrock_agentcore_storage.MemoryClient")
    def test_search_client_error(
        self, mock_memory_client_class, config_with_strategies
    ):
        """Test search method when client raises an error."""
        mock_client = MagicMock()
        mock_memory_client_class.return_value = mock_client
        mock_client.retrieve_memories.side_effect = Exception("Search error")

        storage = BedrockAgentCoreStorage(
            type="external", config=config_with_strategies
        )

        # Should return empty list on error, not raise exception
        results = storage.search("test query")
        assert results == []

    @patch("crewai.memory.storage.bedrock_agentcore_storage.MemoryClient")
    def test_search_namespace_error_handling(
        self, mock_memory_client_class, config_with_strategies
    ):
        """Test that search continues when one namespace fails."""
        mock_client = MagicMock()
        mock_memory_client_class.return_value = mock_client

        # First namespace fails, second succeeds
        mock_client.retrieve_memories.side_effect = [
            Exception("Namespace error"),
            [
                {
                    "memoryRecordId": "record-1",
                    "content": {"text": "Success result"},
                    "score": 0.9,
                    "namespaces": ["/facts/actor-456/session-789"],
                    "createdAt": "2024-01-01T00:00:00Z",
                }
            ],
        ]

        storage = BedrockAgentCoreStorage(
            type="external", config=config_with_strategies
        )

        results = storage.search("test query")

        # Should return results from successful namespace
        assert len(results) == 1
        assert results[0]["context"] == "Success result"

    @patch("crewai.memory.storage.bedrock_agentcore_storage.MemoryClient")
    def test_determine_role_from_metadata(self, mock_memory_client_class, basic_config):
        """Test role determination from metadata."""
        mock_memory_client_class.return_value = MagicMock()
        storage = BedrockAgentCoreStorage(type="external", config=basic_config)

        # Test explicit role
        assert storage._determine_role_from_metadata({"role": "user"}) == "USER"
        assert (
            storage._determine_role_from_metadata({"role": "assistant"}) == "ASSISTANT"
        )
        assert storage._determine_role_from_metadata({"role": "tool"}) == "TOOL"

        # Test agent metadata
        assert (
            storage._determine_role_from_metadata({"agent": "test_agent"})
            == "ASSISTANT"
        )

        # Test user metadata
        assert storage._determine_role_from_metadata({"user": "test_user"}) == "USER"

        # Test tool metadata
        assert storage._determine_role_from_metadata({"tool": "test_tool"}) == "TOOL"

        # Test default
        assert storage._determine_role_from_metadata({}) == "ASSISTANT"
        assert storage._determine_role_from_metadata(None) == "ASSISTANT"  # type: ignore

    @patch("crewai.memory.storage.bedrock_agentcore_storage.MemoryClient")
    def test_convert_to_message_tuples_single_string(
        self, mock_memory_client_class, basic_config
    ):
        """Test converting single string to message tuples."""
        mock_memory_client_class.return_value = MagicMock()
        storage = BedrockAgentCoreStorage(type="external", config=basic_config)

        result = storage._convert_to_message_tuples("test message", {"role": "user"})
        assert result == [("test message", "USER")]

    @patch("crewai.memory.storage.bedrock_agentcore_storage.MemoryClient")
    def test_convert_to_message_tuples_dict(
        self, mock_memory_client_class, basic_config
    ):
        """Test converting dictionary to message tuples."""
        mock_memory_client_class.return_value = MagicMock()
        storage = BedrockAgentCoreStorage(type="external", config=basic_config)

        test_dict = {"key": "value", "number": 123}
        result = storage._convert_to_message_tuples(test_dict, {"role": "assistant"})

        expected_json = json.dumps(test_dict, default=str)
        assert result == [(expected_json, "ASSISTANT")]

    @patch("crewai.memory.storage.bedrock_agentcore_storage.MemoryClient")
    def test_convert_to_message_tuples_object_with_raw(
        self, mock_memory_client_class, basic_config
    ):
        """Test converting object with raw attribute to message tuples."""
        mock_memory_client_class.return_value = MagicMock()
        storage = BedrockAgentCoreStorage(type="external", config=basic_config)

        mock_obj = MagicMock()
        mock_obj.raw = "raw content"

        result = storage._convert_to_message_tuples(mock_obj, {"role": "tool"})
        assert result == [("raw content", "TOOL")]

    @patch("crewai.memory.storage.bedrock_agentcore_storage.MemoryClient")
    def test_convert_to_message_tuples_list(
        self, mock_memory_client_class, basic_config
    ):
        """Test converting list of messages to message tuples."""
        mock_memory_client_class.return_value = MagicMock()
        storage = BedrockAgentCoreStorage(type="external", config=basic_config)

        messages = [
            {"content": "User message", "metadata": {"role": "user"}},
            {"content": "Assistant response", "metadata": {"role": "assistant"}},
            "Simple string message",
        ]

        result = storage._convert_to_message_tuples(messages, {"role": "default"})

        # For string messages without metadata, the default role is ASSISTANT (not DEFAULT)
        expected = [
            ("User message", "USER"),
            ("Assistant response", "ASSISTANT"),
            ("Simple string message", "ASSISTANT"),  # Default role is ASSISTANT
        ]
        assert result == expected

    @patch("crewai.memory.storage.bedrock_agentcore_storage.MemoryClient")
    def test_reset_method(self, mock_memory_client_class, basic_config):
        """Test reset method (should log warning)."""
        mock_memory_client_class.return_value = MagicMock()

        storage = BedrockAgentCoreStorage(type="external", config=basic_config)

        # Should not raise an exception, just log a warning
        storage.reset()
