"""Tests for ValkeyStorage error handling."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from crewai.memory.storage.valkey_storage import ValkeyStorage
from crewai.memory.types import MemoryRecord


@pytest.fixture
def mock_glide_client() -> AsyncMock:
    """Create a mock GlideClient for testing."""
    client = AsyncMock()
    client.hset = AsyncMock(return_value=1)
    client.zrange = AsyncMock(return_value=[])
    client.zadd = AsyncMock()
    client.sadd = AsyncMock()
    client.hgetall = AsyncMock(return_value={})
    client.close = AsyncMock()
    return client


@pytest.fixture
def valkey_storage(mock_glide_client: AsyncMock) -> ValkeyStorage:
    """Create a ValkeyStorage instance with mocked client."""
    storage = ValkeyStorage(host="localhost", port=6379, db=0)

    # Mock the client creation to return our mock
    async def mock_create_client() -> AsyncMock:
        storage._client = mock_glide_client
        return mock_glide_client

    storage._get_client = mock_create_client  # type: ignore[method-assign]
    return storage


class TestSerializationErrors:
    """Tests for serialization error handling."""

    def test_serialization_error_raises_descriptive_exception(
        self, valkey_storage: ValkeyStorage
    ) -> None:
        """Test that serialization errors raise descriptive ValueError."""
        # Create a record with non-serializable metadata
        record = MemoryRecord(
            id="test-id",
            content="test content",
            scope="/test",
            categories=["test"],
            metadata={"bad_key": object()},  # Non-serializable object
            importance=0.5,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            embedding=[0.1, 0.2, 0.3],
        )

        # Should raise ValueError with descriptive message
        with pytest.raises(ValueError, match="Failed to serialize record test-id"):
            valkey_storage._record_to_dict(record)

    def test_serialization_error_includes_cause(
        self, valkey_storage: ValkeyStorage
    ) -> None:
        """Test that serialization error includes the original exception as cause."""
        # Create a mock record that will fail during JSON serialization
        # We need to bypass Pydantic validation, so we'll patch json.dumps
        record = MemoryRecord(
            id="test-id-2",
            content="test content",
            scope="/test",
            categories=["valid"],
            metadata={"key": "value"},
            importance=0.5,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            embedding=[0.1, 0.2, 0.3],
        )

        # Patch json.dumps to raise an error
        with patch("json.dumps", side_effect=TypeError("Cannot serialize")):
            with pytest.raises(ValueError) as exc_info:
                valkey_storage._record_to_dict(record)

            # Verify the exception has a cause
            assert exc_info.value.__cause__ is not None
            assert isinstance(exc_info.value.__cause__, TypeError)


class TestDeserializationErrors:
    """Tests for deserialization error handling."""

    def test_deserialization_error_logs_and_returns_none(
        self, valkey_storage: ValkeyStorage, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that deserialization errors log error and return None."""
        # Create malformed data (missing required fields)
        malformed_data = {
            "id": "test-id",
            "content": "test content",
            # Missing scope, categories, metadata, etc.
        }

        # Should return None and log error
        result = valkey_storage._dict_to_record(malformed_data)

        assert result is None
        assert "Failed to deserialize record test-id" in caplog.text

    def test_deserialization_with_invalid_json_categories_uses_tag_fallback(
        self, valkey_storage: ValkeyStorage
    ) -> None:
        """Test that non-JSON categories fall back to TAG (comma-separated) parsing."""
        # Create data with non-JSON categories string
        data = {
            "id": "test-id-json",
            "content": "test content",
            "scope": "/test",
            "categories": "not valid json [",  # Not JSON, treated as TAG format
            "metadata": "{}",
            "importance": "0.5",
            "created_at": "2024-01-01T12:00:00",
            "last_accessed": "2024-01-01T12:00:00",
            "source": "",
            "private": "false",
        }

        result = valkey_storage._dict_to_record(data)

        # TAG fallback: comma-split produces the raw string as a single category
        assert result is not None
        assert result.id == "test-id-json"
        assert result.categories == ["not valid json ["]

    def test_deserialization_with_invalid_datetime_returns_none(
        self, valkey_storage: ValkeyStorage, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid datetime format returns None."""
        # Create data with invalid datetime
        invalid_data = {
            "id": "test-id-datetime",
            "content": "test content",
            "scope": "/test",
            "categories": '["test"]',
            "metadata": "{}",
            "importance": "0.5",
            "created_at": "not a datetime",  # Invalid datetime
            "last_accessed": "2024-01-01T12:00:00",
            "source": "",
            "private": "false",
        }

        result = valkey_storage._dict_to_record(invalid_data)

        assert result is None
        assert "Failed to deserialize record test-id-datetime" in caplog.text

    def test_deserialization_with_invalid_float_returns_none(
        self, valkey_storage: ValkeyStorage, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Test that invalid float importance returns None."""
        # Create data with invalid float
        invalid_data = {
            "id": "test-id-float",
            "content": "test content",
            "scope": "/test",
            "categories": '["test"]',
            "metadata": "{}",
            "importance": "not a float",  # Invalid float
            "created_at": "2024-01-01T12:00:00",
            "last_accessed": "2024-01-01T12:00:00",
            "source": "",
            "private": "false",
        }

        result = valkey_storage._dict_to_record(invalid_data)

        assert result is None
        assert "Failed to deserialize record test-id-float" in caplog.text

    def test_deserialization_with_bytes_keys_uses_tag_fallback(
        self, valkey_storage: ValkeyStorage
    ) -> None:
        """Test that deserialization handles bytes keys with non-JSON categories via TAG fallback."""
        # Create data with bytes keys (as returned by Valkey)
        bytes_data = {
            b"id": b"test-id-bytes",
            b"content": b"test content",
            b"scope": b"/test",
            b"categories": b"invalid json [",  # Not JSON, treated as TAG format
            b"metadata": b"{}",
            b"importance": b"0.5",
            b"created_at": b"2024-01-01T12:00:00",
            b"last_accessed": b"2024-01-01T12:00:00",
        }

        result = valkey_storage._dict_to_record(bytes_data)

        # TAG fallback: comma-split produces the raw string as a single category
        assert result is not None
        assert result.id == "test-id-bytes"
        assert result.categories == ["invalid json ["]


class TestRetryBehaviorIntegration:
    """Integration tests demonstrating retry behavior patterns."""

    @pytest.mark.asyncio
    async def test_mock_client_operation_with_retry_pattern(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test demonstrating how retry would work with client operations."""
        from glide import ClosingError

        # Mock a client operation that fails once
        mock_glide_client.hgetall.side_effect = [
            ClosingError("Connection lost"),
            {
                b"id": b"test-id",
                b"content": b"test content",
                b"scope": b"/test",
                b"categories": b'["test"]',
                b"metadata": b"{}",
                b"importance": b"0.5",
                b"created_at": b"2024-01-01T12:00:00",
                b"last_accessed": b"2024-01-01T12:00:00",
                b"source": b"",
                b"private": b"false",
                b"embedding": b"",
            },
        ]

        # First call fails, second succeeds
        with pytest.raises(ClosingError):
            await mock_glide_client.hgetall("record:test-id")

        # Second call succeeds
        result = await mock_glide_client.hgetall("record:test-id")
        assert result is not None

    @pytest.mark.asyncio
    async def test_serialization_error_not_retried(
        self, valkey_storage: ValkeyStorage
    ) -> None:
        """Test that serialization errors are not retried (they're not connection errors)."""
        # Create a record with non-serializable data
        record = MemoryRecord(
            id="test-id",
            content="test content",
            scope="/test",
            categories=["test"],
            metadata={"bad": object()},
            importance=0.5,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            embedding=[0.1, 0.2, 0.3],
        )

        # Serialization error should not be retried
        with pytest.raises(ValueError, match="Failed to serialize"):
            valkey_storage._record_to_dict(record)
