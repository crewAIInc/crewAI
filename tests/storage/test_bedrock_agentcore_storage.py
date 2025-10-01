"""Tests for BedrockAgentCoreStorage implementation."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from crewai.memory.storage.bedrock_agentcore_storage import (
    BedrockAgentCoreConfig,
    BedrockAgentCoreStorage,
)


class TestBedrockAgentCoreConfig:
    """Test BedrockAgentCoreConfig validation and behavior."""

    def test_config_with_required_fields_only(self):
        """Test config creation with only required fields."""
        config = BedrockAgentCoreConfig(
            memory_id="mem-123",
            actor_id="actor-456",
            session_id="session-789",
        )

        assert config.memory_id == "mem-123"
        assert config.actor_id == "actor-456"
        assert config.session_id == "session-789"
        assert config.region_name == "us-east-1"  # default
        assert config.namespaces == []

    def test_config_with_all_fields(self):
        """Test config creation with all fields."""
        config = BedrockAgentCoreConfig(
            memory_id="mem-123",
            actor_id="actor-456",
            session_id="session-789",
            region_name="us-west-2",
            namespaces=["/preferences/actor-456", "/facts/session-789"],
        )

        assert config.memory_id == "mem-123"
        assert config.actor_id == "actor-456"
        assert config.session_id == "session-789"
        assert config.region_name == "us-west-2"
        assert len(config.namespaces) == 2
        assert "/preferences/actor-456" in config.namespaces

    def test_config_validation_empty_required_fields(self):
        """Test validation fails for empty required fields."""
        with pytest.raises(ValueError, match="Required ID fields cannot be empty"):
            BedrockAgentCoreConfig(
                memory_id="",
                actor_id="actor-456",
                session_id="session-789",
            )

        with pytest.raises(ValueError, match="Required ID fields cannot be empty"):
            BedrockAgentCoreConfig(
                memory_id="mem-123",
                actor_id="   ",  # whitespace only
                session_id="session-789",
            )

    def test_config_validation_empty_region(self):
        """Test validation fails for empty region."""
        with pytest.raises(ValueError, match="Region name cannot be empty"):
            BedrockAgentCoreConfig(
                memory_id="mem-123",
                actor_id="actor-456",
                session_id="session-789",
                region_name="",
            )

    def test_config_validation_invalid_namespaces(self):
        """Test validation fails for invalid namespaces."""
        # Test with empty strings in namespaces
        with pytest.raises(ValueError, match="Namespace cannot be empty"):
            BedrockAgentCoreConfig(
                memory_id="mem-123",
                actor_id="actor-456",
                session_id="session-789",
                namespaces=["", "  "],  # Invalid namespaces
            )

        with pytest.raises(ValueError, match="Namespace cannot be empty"):
            BedrockAgentCoreConfig(
                memory_id="mem-123",
                actor_id="actor-456",
                session_id="session-789",
                namespaces=["valid-namespace", ""],
            )

    def test_config_strips_whitespace(self):
        """Test that config strips whitespace from values."""
        config = BedrockAgentCoreConfig(
            memory_id="  mem-123  ",
            actor_id="  actor-456  ",
            session_id="  session-789  ",
            region_name="  us-west-2  ",
            namespaces=["  /preferences/actor  ", "  /facts/session  "],
        )

        assert config.memory_id == "mem-123"
        assert config.actor_id == "actor-456"
        assert config.session_id == "session-789"
        assert config.region_name == "us-west-2"
        assert config.namespaces == ["/preferences/actor", "/facts/session"]

    def test_config_forbids_extra_fields(self):
        """Test that config forbids extra fields."""
        with pytest.raises(ValueError, match="Extra inputs are not permitted"):
            BedrockAgentCoreConfig(
                memory_id="mem-123",
                actor_id="actor-456",
                session_id="session-789",
                extra_field="not-allowed",  # type: ignore
            )


class TestBedrockAgentCoreStorage:
    """Test BedrockAgentCoreStorage functionality."""

    @pytest.fixture
    def basic_config(self):
        """Basic configuration for testing."""
        return BedrockAgentCoreConfig(
            memory_id="mem-123",
            actor_id="actor-456",
            session_id="session-789",
            region_name="us-west-2",
        )

    @pytest.fixture
    def config_with_namespaces(self):
        """Configuration with namespaces for testing."""
        return BedrockAgentCoreConfig(
            memory_id="mem-123",
            actor_id="actor-456",
            session_id="session-789",
            region_name="us-west-2",
            namespaces=[
                "/strategies/{memoryStrategyId}/actors/{actorId}",
                "/facts/{actorId}/sessions/{sessionId}",
            ],
        )

    @pytest.fixture
    def mock_boto3_client(self):
        """Mock boto3 client."""
        with patch("boto3.client") as mock_client:
            mock_bedrock_client = MagicMock()
            mock_client.return_value = mock_bedrock_client
            yield mock_bedrock_client

    def test_storage_initialization_success(self, basic_config, mock_boto3_client):
        """Test successful storage initialization."""
        storage = BedrockAgentCoreStorage(type="external", config=basic_config)

        assert storage.memory_type == "external"
        assert storage.config == basic_config
        assert storage.memory_client is not None

    def test_storage_initialization_invalid_type(self, basic_config):
        """Test storage initialization with invalid type."""
        with pytest.raises(
            ValueError, match="Invalid type 'invalid' for BedrockAgentCoreStorage"
        ):
            BedrockAgentCoreStorage(type="invalid", config=basic_config)

    def test_storage_initialization_boto3_not_installed(self, basic_config):
        """Test storage initialization when boto3 is not installed."""
        # Mock the import by patching sys.modules to remove boto3
        import sys

        original_modules = sys.modules.copy()

        # Remove boto3 from sys.modules if it exists
        if "boto3" in sys.modules:
            del sys.modules["boto3"]

        # Mock import to raise ImportError
        def mock_import(name, *args, **kwargs):
            if name == "boto3":
                raise ImportError("No module named 'boto3'")
            return original_modules.get(name)

        try:
            with patch("builtins.__import__", side_effect=mock_import):
                with pytest.raises(
                    ModuleNotFoundError, match="boto3 is required for Bedrock AgentCore"
                ):
                    BedrockAgentCoreStorage(type="external", config=basic_config)
        finally:
            # Restore original sys.modules
            sys.modules.clear()
            sys.modules.update(original_modules)

    def test_storage_initialization_boto3_error(self, basic_config):
        """Test storage initialization with boto3 client error."""
        with patch("boto3.client", side_effect=Exception("AWS error")):
            with pytest.raises(
                ValueError,
                match="Failed to initialize Bedrock AgentCore clients: AWS error",
            ):
                BedrockAgentCoreStorage(type="external", config=basic_config)

    def test_save_basic(self, basic_config, mock_boto3_client):
        """Test basic save operation."""
        storage = BedrockAgentCoreStorage(type="external", config=basic_config)

        # Mock the create_event response
        mock_boto3_client.create_event.return_value = {
            "event": {"eventId": "event-123"}
        }

        # Test save
        value = "Test task output"
        metadata = {
            "messages": [
                {"role": "user", "content": "What is AI?"},
                {"role": "assistant", "content": "AI is artificial intelligence."},
            ]
        }

        storage.save(value=value, metadata=metadata)

        # Verify create_event was called
        mock_boto3_client.create_event.assert_called_once()
        call_args = mock_boto3_client.create_event.call_args

        assert call_args.kwargs["memoryId"] == "mem-123"
        assert call_args.kwargs["actorId"] == "actor-456"
        assert call_args.kwargs["sessionId"] == "session-789"
        assert isinstance(call_args.kwargs["eventTimestamp"], datetime)

        # Verify payload structure
        payload = call_args.kwargs["payload"]
        assert len(payload) == 3  # 2 messages + 1 task output
        assert payload[0]["conversational"]["content"]["text"] == "What is AI?"
        assert payload[0]["conversational"]["role"] == "USER"
        assert (
            payload[1]["conversational"]["content"]["text"]
            == "AI is artificial intelligence."
        )
        assert payload[1]["conversational"]["role"] == "ASSISTANT"
        assert payload[2]["conversational"]["content"]["text"] == value
        assert payload[2]["conversational"]["role"] == "ASSISTANT"

    def test_save_with_role_mapping(self, basic_config, mock_boto3_client):
        """Test save with various role mappings."""
        storage = BedrockAgentCoreStorage(type="external", config=basic_config)

        mock_boto3_client.create_event.return_value = {
            "event": {"eventId": "event-123"}
        }

        metadata = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "tool", "content": "Tool output"},
                {"role": "unknown", "content": "Unknown role"},
            ]
        }

        storage.save(value="Final output", metadata=metadata)

        payload = mock_boto3_client.create_event.call_args.kwargs["payload"]

        # Verify role mappings
        assert payload[0]["conversational"]["role"] == "OTHER"  # system -> OTHER
        assert payload[1]["conversational"]["role"] == "USER"
        assert payload[2]["conversational"]["role"] == "ASSISTANT"
        assert payload[3]["conversational"]["role"] == "TOOL"
        assert payload[4]["conversational"]["role"] == "OTHER"  # unknown -> OTHER

    def test_save_with_error(self, basic_config, mock_boto3_client):
        """Test save with AWS error."""
        storage = BedrockAgentCoreStorage(type="external", config=basic_config)

        mock_boto3_client.create_event.side_effect = Exception("AWS error")

        with pytest.raises(Exception, match="AWS error"):
            storage.save(value="test", metadata={"messages": []})

    def test_search_with_namespaces(self, config_with_namespaces, mock_boto3_client):
        """Test search operation with namespaces."""
        storage = BedrockAgentCoreStorage(
            type="external", config=config_with_namespaces
        )

        # Mock retrieve_memory_records response - return different results for each namespace
        mock_boto3_client.retrieve_memory_records.side_effect = [
            {
                "memoryRecordSummaries": [
                    {
                        "memoryRecordId": "record-1",
                        "content": {"text": "User prefers dark mode"},
                        "score": 0.9,
                        "namespaces": ["/preferences/actor-456"],
                        "createdAt": "2024-01-01T00:00:00Z",
                        "memoryStrategyId": "strat-123",
                    }
                ]
            },
            {
                "memoryRecordSummaries": [
                    {
                        "memoryRecordId": "record-2",
                        "content": {"text": "User likes Python"},
                        "score": 0.8,
                        "namespaces": ["/facts/actor-456/session-789"],
                        "createdAt": "2024-01-02T00:00:00Z",
                        "memoryStrategyId": "strat-123",
                    }
                ]
            },
        ]

        # Mock list_events for short-term memory fallback
        mock_boto3_client.list_events.return_value = {"events": []}

        results = storage.search("user preferences", limit=5, score_threshold=0.7)

        # Verify search was performed on both namespaces
        assert mock_boto3_client.retrieve_memory_records.call_count == 2

        # Verify results - should have 2 unique results
        assert len(results) == 2
        assert results[0]["content"] == "User prefers dark mode"
        assert results[0]["score"] == 0.9
        assert results[1]["content"] == "User likes Python"
        assert results[1]["score"] == 0.8

    def test_search_with_score_threshold(
        self, config_with_namespaces, mock_boto3_client
    ):
        """Test search filters by score threshold."""
        storage = BedrockAgentCoreStorage(
            type="external", config=config_with_namespaces
        )

        # Use side_effect to return different results for each namespace call
        mock_boto3_client.retrieve_memory_records.side_effect = [
            {
                "memoryRecordSummaries": [
                    {
                        "memoryRecordId": "record-1",
                        "content": {"text": "High relevance"},
                        "score": 0.9,
                    },
                    {
                        "memoryRecordId": "record-2",
                        "content": {"text": "Low relevance"},
                        "score": 0.3,  # Below threshold
                    },
                ]
            },
            {
                "memoryRecordSummaries": []  # Empty results from second namespace
            },
        ]

        mock_boto3_client.list_events.return_value = {"events": []}

        results = storage.search("test", limit=5, score_threshold=0.5)

        # Only high-score result should be returned (low score filtered out)
        assert len(results) == 1
        assert results[0]["content"] == "High relevance"

    def test_search_with_short_term_memory_fallback(
        self, basic_config, mock_boto3_client
    ):
        """Test search falls back to short-term memory when needed."""
        storage = BedrockAgentCoreStorage(type="external", config=basic_config)

        # No namespaces configured, so should return empty results
        results = storage.search("test", limit=3)

        # Should return empty results since no namespaces are configured
        assert results == []

        # Should not have called retrieve_memory_records or list_events
        mock_boto3_client.retrieve_memory_records.assert_not_called()
        mock_boto3_client.list_events.assert_called_once()

    def test_search_with_namespace_error(
        self, config_with_namespaces, mock_boto3_client
    ):
        """Test search raises exception when namespace search fails."""
        storage = BedrockAgentCoreStorage(
            type="external", config=config_with_namespaces
        )

        # First namespace fails - this should now raise the exception
        mock_boto3_client.retrieve_memory_records.side_effect = Exception(
            "Namespace error"
        )

        # The search should raise the exception
        with pytest.raises(Exception, match="Namespace error"):
            storage.search("test", limit=5)

    def test_search_empty_namespaces(self, basic_config, mock_boto3_client):
        """Test search with no configured namespaces."""
        storage = BedrockAgentCoreStorage(type="external", config=basic_config)

        results = storage.search("test", limit=5)

        # Should return empty results without calling retrieve_memory_records
        assert results == []
        mock_boto3_client.retrieve_memory_records.assert_not_called()

    def test_reset(self, basic_config, mock_boto3_client):
        """Test reset operation logs warning."""
        storage = BedrockAgentCoreStorage(type="external", config=basic_config)

        with patch(
            "crewai.memory.storage.bedrock_agentcore_storage.logger"
        ) as mock_logger:
            storage.reset()

            mock_logger.warning.assert_called_once_with(
                "Bedrock AgentCore memory reset not implemented - memories persist in AWS"
            )

    def test_convert_to_event_payload_empty_messages(
        self, basic_config, mock_boto3_client
    ):
        """Test payload conversion with empty messages."""
        storage = BedrockAgentCoreStorage(type="external", config=basic_config)

        value = "Task output only"
        metadata = {"messages": []}

        payload = storage._convert_to_event_payload(value, metadata)

        # Should only have the task output
        assert len(payload) == 1
        assert payload[0]["conversational"]["content"]["text"] == "Task output only"
        assert payload[0]["conversational"]["role"] == "ASSISTANT"

    def test_search_results_sorting(self, config_with_namespaces, mock_boto3_client):
        """Test search results are sorted by score."""
        storage = BedrockAgentCoreStorage(
            type="external", config=config_with_namespaces
        )

        # Return results in non-sorted order
        mock_boto3_client.retrieve_memory_records.side_effect = [
            {
                "memoryRecordSummaries": [
                    {
                        "memoryRecordId": "1",
                        "content": {"text": "Mid score"},
                        "score": 0.5,
                    },
                    {
                        "memoryRecordId": "2",
                        "content": {"text": "High score"},
                        "score": 0.9,
                    },
                ]
            },
            {
                "memoryRecordSummaries": [
                    {
                        "memoryRecordId": "3",
                        "content": {"text": "Low score"},
                        "score": 0.4,
                    },
                    {
                        "memoryRecordId": "4",
                        "content": {"text": "Medium score"},
                        "score": 0.7,
                    },
                ]
            },
        ]

        mock_boto3_client.list_events.return_value = {"events": []}

        results = storage.search("test", limit=10, score_threshold=0.35)

        # Verify results are sorted by score (descending)
        scores = [r["score"] for r in results]
        assert scores == [0.9, 0.7, 0.5, 0.4]

    def test_search_with_short_term_memory_supplement(
        self, config_with_namespaces, mock_boto3_client
    ):
        """Test search supplements long-term results with short-term memory when needed."""
        storage = BedrockAgentCoreStorage(
            type="external", config=config_with_namespaces
        )

        # Mock long-term memory with only 1 result (less than limit of 5)
        mock_boto3_client.retrieve_memory_records.side_effect = [
            {
                "memoryRecordSummaries": [
                    {
                        "memoryRecordId": "long-term-1",
                        "content": {"text": "Long-term memory result"},
                        "score": 0.8,
                        "createdAt": "2024-01-01T00:00:00Z",
                    }
                ]
            },
            {
                "memoryRecordSummaries": []  # Empty from second namespace
            },
        ]

        # Mock short-term memory with events
        mock_boto3_client.list_events.return_value = {
            "events": [
                {
                    "eventId": "event-1",
                    "eventTimestamp": "2024-01-02T00:00:00Z",
                    "payload": [
                        {
                            "conversational": {
                                "content": {"text": "Recent conversation from user"},
                                "role": "USER",
                            }
                        }
                    ],
                },
                {
                    "eventId": "event-2",
                    "eventTimestamp": "2024-01-02T01:00:00Z",
                    "payload": [
                        {
                            "conversational": {
                                "content": {"text": "Assistant response"},
                                "role": "ASSISTANT",
                            }
                        }
                    ],
                },
            ]
        }

        results = storage.search("test", limit=5, score_threshold=0.5)

        # Should have called list_events to supplement results
        mock_boto3_client.list_events.assert_called_once_with(
            memoryId="mem-123",
            actorId="actor-456",
            sessionId="session-789",
            includePayloads=True,
            maxResults=4,  # limit(5) - long_term_results(1) = 4
        )

        # Should return 3 results: 1 long-term + 2 short-term
        assert len(results) == 3

        # Verify long-term result
        assert results[0]["content"] == "Long-term memory result"
        assert results[0]["score"] == 0.8
        assert results[0]["id"] == "long-term-1"

        # Verify short-term results have default score of 1.0
        short_term_results = [r for r in results if r["id"].startswith("event-")]
        assert len(short_term_results) == 2
        assert all(r["score"] == 1.0 for r in short_term_results)
        assert short_term_results[0]["content"] == "Recent conversation from user"
        assert short_term_results[0]["metadata"]["role"] == "USER"
        assert short_term_results[1]["content"] == "Assistant response"
        assert short_term_results[1]["metadata"]["role"] == "ASSISTANT"

    def test_search_short_term_memory_error_handling(
        self, config_with_namespaces, mock_boto3_client
    ):
        """Test search handles short-term memory errors gracefully."""
        storage = BedrockAgentCoreStorage(
            type="external", config=config_with_namespaces
        )

        # Mock long-term memory with partial results
        mock_boto3_client.retrieve_memory_records.side_effect = [
            {
                "memoryRecordSummaries": [
                    {
                        "memoryRecordId": "long-term-1",
                        "content": {"text": "Long-term result"},
                        "score": 0.9,
                    }
                ]
            },
            {"memoryRecordSummaries": []},
        ]

        # Mock short-term memory to raise an error
        mock_boto3_client.list_events.side_effect = Exception("Short-term memory error")

        results = storage.search("test", limit=5)

        # Should still return long-term results despite short-term memory error
        assert len(results) == 1
        assert results[0]["content"] == "Long-term result"
        assert results[0]["score"] == 0.9

        # Should have attempted to call list_events
        mock_boto3_client.list_events.assert_called_once()
