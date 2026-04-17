"""Tests for ValkeyStorage save operation."""

from __future__ import annotations

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


class TestValkeyStorageSave:
    """Tests for ValkeyStorage save operation."""

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_save_single_record_with_all_fields(
        self, mock_ft_list, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test saving a single record with all fields populated."""
        # Create a record with all fields
        record = MemoryRecord(
            id="test-id-123",
            content="Test memory content",
            scope="/agent/task",
            categories=["planning", "execution"],
            metadata={"agent_id": "agent-1", "priority": "high"},
            importance=0.8,
            created_at=datetime(2024, 1, 1, 12, 0, 0),
            last_accessed=datetime(2024, 1, 1, 12, 0, 0),
            embedding=[0.1, 0.2, 0.3, 0.4],
            source="test-source",
            private=True,
        )

        # Mock ft.list to simulate index exists
        mock_ft_list.return_value = [b"memory_index"]

        # Save the record
        await valkey_storage.asave([record])

        # Verify ft.list was called to check index
        mock_ft_list.assert_called_once()

        # Verify HSET was called with correct record data
        mock_glide_client.hset.assert_called()
        hset_call = mock_glide_client.hset.call_args
        assert hset_call[0][0] == "record:test-id-123"  # key
        hset_dict = hset_call[0][1]  # field_value_map dict
        
        assert hset_dict["id"] == "test-id-123"
        assert hset_dict["content"] == "Test memory content"
        assert hset_dict["scope"] == "/agent/task"
        assert hset_dict["source"] == "test-source"
        assert hset_dict["private"] == "true"
        assert hset_dict["importance"] == "0.8"
        assert "embedding" in hset_dict
        assert isinstance(hset_dict["embedding"], bytes)

        # Verify scope index was updated
        mock_glide_client.zadd.assert_called_once()
        zadd_call = mock_glide_client.zadd.call_args
        assert zadd_call[0][0] == "scope:/agent/task"
        assert "test-id-123" in zadd_call[0][1]

        # Verify category indexes were updated
        assert mock_glide_client.sadd.call_count >= 2
        sadd_calls = [call[0] for call in mock_glide_client.sadd.call_args_list]
        category_calls = [call for call in sadd_calls if call[0].startswith("category:")]
        assert len(category_calls) == 2
        assert any("category:planning" in str(call) for call in category_calls)
        assert any("category:execution" in str(call) for call in category_calls)

        # Verify metadata indexes were updated
        metadata_calls = [call for call in sadd_calls if call[0].startswith("metadata:")]
        assert len(metadata_calls) == 2
        assert any("metadata:agent_id:agent-1" in str(call) for call in metadata_calls)
        assert any("metadata:priority:high" in str(call) for call in metadata_calls)

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_save_multiple_records_in_batch(
        self, mock_ft_list, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test saving multiple records in a single batch."""
        records = [
            MemoryRecord(
                id=f"record-{i}",
                content=f"Content {i}",
                scope="/test",
                embedding=[0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i],
            )
            for i in range(3)
        ]

        # Mock ft.list to simulate index exists
        mock_ft_list.return_value = [b"memory_index"]

        await valkey_storage.asave(records)

        # Verify HSET was called for each record
        assert mock_glide_client.hset.call_count == 3

        # Verify each record was stored
        hset_calls = mock_glide_client.hset.call_args_list
        for i in range(3):
            record_key = f"record:record-{i}"
            assert any(call[0][0] == record_key for call in hset_calls)

        # Verify scope index was updated for all records
        assert mock_glide_client.zadd.call_count == 3

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_save_record_with_empty_categories_and_metadata(
        self, mock_ft_list, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test saving a record with empty categories and metadata."""
        record = MemoryRecord(
            id="empty-fields-record",
            content="Content with no categories or metadata",
            scope="/test",
            categories=[],
            metadata={},
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        # Mock ft.list to simulate index exists
        mock_ft_list.return_value = [b"memory_index"]

        await valkey_storage.asave([record])

        # Verify record was saved
        mock_glide_client.hset.assert_called_once()

        # Verify no category or metadata index updates
        sadd_calls = mock_glide_client.sadd.call_args_list
        # Should have no calls since categories and metadata are empty
        assert len(sadd_calls) == 0

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_save_record_without_embedding(
        self, mock_ft_list, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test saving a record without an embedding."""
        record = MemoryRecord(
            id="no-embedding-record",
            content="Content without embedding",
            scope="/test",
            embedding=None,
        )

        # Mock ft.list to simulate index exists
        mock_ft_list.return_value = [b"memory_index"]

        await valkey_storage.asave([record])

        # Verify record was saved
        mock_glide_client.hset.assert_called_once()

        # Verify embedding field is empty bytes
        hset_call = mock_glide_client.hset.call_args
        hset_dict = hset_call[0][1]  # field_value_map dict
        
        assert "embedding" in hset_dict
        assert hset_dict["embedding"] == b""

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_save_record_with_none_source(
        self, mock_ft_list, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test saving a record with None source."""
        record = MemoryRecord(
            id="none-source-record",
            content="Content with None source",
            scope="/test",
            source=None,
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        # Mock ft.list to simulate index exists
        mock_ft_list.return_value = [b"memory_index"]

        await valkey_storage.asave([record])

        # Verify record was saved
        mock_glide_client.hset.assert_called_once()

        # Verify source field is empty string
        hset_call = mock_glide_client.hset.call_args
        hset_dict = hset_call[0][1]  # field_value_map dict
        
        assert hset_dict["source"] == ""

    @pytest.mark.asyncio
    async def test_save_empty_list_does_nothing(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test that saving an empty list does nothing."""
        await valkey_storage.asave([])

        # Verify no operations were performed
        mock_glide_client.hset.assert_not_called()
        mock_glide_client.zadd.assert_not_called()
        mock_glide_client.sadd.assert_not_called()

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.FtCreateOptions")
    @patch("crewai.memory.storage.valkey_storage.VectorField")
    @patch("crewai.memory.storage.valkey_storage.VectorFieldAttributesHnsw")
    @patch("crewai.memory.storage.valkey_storage.ft.create")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_save_creates_vector_index_if_not_exists(
        self, mock_ft_list, mock_ft_create, mock_vector_attrs, mock_vector_field, mock_ft_create_options, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test that save creates vector index if it doesn't exist."""
        record = MemoryRecord(
            id="test-record",
            content="Test content",
            scope="/test",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        # Mock ft.info to fail (index doesn't exist), then ft.create succeeds
        mock_ft_list.return_value = []
        mock_ft_create.return_value = "OK"

        await valkey_storage.asave([record])

        # Verify ft.create was called
        mock_ft_create.assert_called_once()
        
        # Verify ft.create was called with correct index name
        create_args = mock_ft_create.call_args
        assert create_args[0][1] == "memory_index"

    @pytest.mark.asyncio
    async def test_save_error_handling_for_serialization_failure(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test error handling when serialization fails."""
        # Create a record with a field that will cause serialization to fail
        record = MemoryRecord(
            id="bad-record",
            content="Test content",
            scope="/test",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        # Mock _record_to_dict to raise an error
        with patch.object(
            valkey_storage,
            "_record_to_dict",
            side_effect=ValueError("Serialization failed"),
        ):
            with pytest.raises(ValueError, match="Serialization failed"):
                await valkey_storage.asave([record])

    @patch("crewai.memory.storage.valkey_storage.ft.list")
    def test_save_sync_wrapper(
        self, mock_ft_list, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test that sync save wrapper calls async implementation."""
        record = MemoryRecord(
            id="sync-test-record",
            content="Test content",
            scope="/test",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        # Mock ft.list to simulate index exists
        mock_ft_list.return_value = [b"memory_index"]

        # Call sync save
        valkey_storage.save([record])

        # Verify async operations were called
        mock_glide_client.hset.assert_called_once()

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_save_with_special_characters_in_metadata(
        self, mock_ft_list, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test saving a record with special characters in metadata values."""
        record = MemoryRecord(
            id="special-chars-record",
            content="Test content",
            scope="/test",
            metadata={
                "key:with:colons": "value:with:colons",
                "key with spaces": "value with spaces",
                "key/with/slashes": "value/with/slashes",
            },
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        # Mock ft.list to simulate index exists
        mock_ft_list.return_value = [b"memory_index"]

        await valkey_storage.asave([record])

        # Verify metadata indexes were created with special characters
        sadd_calls = mock_glide_client.sadd.call_args_list
        metadata_calls = [call[0][0] for call in sadd_calls if call[0][0].startswith("metadata:")]
        
        assert len(metadata_calls) == 3
        assert any("key:with:colons:value:with:colons" in call for call in metadata_calls)
        assert any("key with spaces:value with spaces" in call for call in metadata_calls)
        assert any("key/with/slashes:value/with/slashes" in call for call in metadata_calls)

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_save_with_numeric_metadata_values(
        self, mock_ft_list, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test saving a record with numeric metadata values."""
        record = MemoryRecord(
            id="numeric-metadata-record",
            content="Test content",
            scope="/test",
            metadata={
                "count": 42,
                "score": 3.14,
                "is_active": True,
            },
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        # Mock ft.list to simulate index exists
        mock_ft_list.return_value = [b"memory_index"]

        await valkey_storage.asave([record])

        # Verify metadata indexes were created with string-converted values
        sadd_calls = mock_glide_client.sadd.call_args_list
        metadata_calls = [call[0][0] for call in sadd_calls if call[0][0].startswith("metadata:")]
        
        assert len(metadata_calls) == 3
        assert any("metadata:count:42" in call for call in metadata_calls)
        assert any("metadata:score:3.14" in call for call in metadata_calls)
        assert any("metadata:is_active:True" in call for call in metadata_calls)

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_save_preserves_datetime_precision(
        self, mock_ft_list, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test that datetime fields are serialized with proper precision."""
        created_at = datetime(2024, 1, 15, 10, 30, 45, 123456)
        last_accessed = datetime(2024, 1, 15, 11, 45, 30, 654321)
        
        record = MemoryRecord(
            id="datetime-precision-record",
            content="Test content",
            scope="/test",
            created_at=created_at,
            last_accessed=last_accessed,
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        # Mock ft.list to simulate index exists
        mock_ft_list.return_value = [b"memory_index"]

        await valkey_storage.asave([record])

        # Verify datetime fields are in ISO format
        mock_glide_client.hset.assert_called_once()
        hset_call = mock_glide_client.hset.call_args
        hset_dict = hset_call[0][1]  # field_value_map dict
        
        assert hset_dict["created_at"] == created_at.isoformat()
        assert hset_dict["last_accessed"] == last_accessed.isoformat()



class TestValkeyStorageGetRecord:
    """Tests for ValkeyStorage get_record operation."""

    @pytest.mark.asyncio
    async def test_retrieve_existing_record_with_all_fields(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test retrieving an existing record with all fields populated."""
        # Mock HGETALL to return a complete record
        mock_glide_client.hgetall.return_value = {
            "id": "test-record-123",
            "content": "Test memory content",
            "scope": "/agent/task",
            "categories": '["planning", "execution"]',
            "metadata": '{"agent_id": "agent-1", "priority": "high"}',
            "importance": "0.8",
            "created_at": "2024-01-01T12:00:00",
            "last_accessed": "2024-01-01T13:00:00",
            "embedding": valkey_storage._embedding_to_bytes([0.1, 0.2, 0.3, 0.4]),
            "source": "test-source",
            "private": "true",
        }

        # Retrieve the record
        record = await valkey_storage._aget_record("test-record-123")

        # Verify HGETALL was called with correct key
        mock_glide_client.hgetall.assert_called_once_with("record:test-record-123")

        # Verify all fields are correctly deserialized
        assert record is not None
        assert record.id == "test-record-123"
        assert record.content == "Test memory content"
        assert record.scope == "/agent/task"
        assert record.categories == ["planning", "execution"]
        assert record.metadata == {"agent_id": "agent-1", "priority": "high"}
        assert record.importance == 0.8
        assert record.created_at == datetime(2024, 1, 1, 12, 0, 0)
        assert record.last_accessed == datetime(2024, 1, 1, 13, 0, 0)
        # Check embedding with approximate comparison (float32 precision)
        assert record.embedding is not None
        assert len(record.embedding) == 4
        for i, expected in enumerate([0.1, 0.2, 0.3, 0.4]):
            assert abs(record.embedding[i] - expected) < 1e-6
        assert record.source == "test-source"
        assert record.private is True

    @pytest.mark.asyncio
    async def test_retrieve_non_existent_record_returns_none(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test retrieving a non-existent record returns None."""
        # Mock HGETALL to return empty dict (record doesn't exist)
        mock_glide_client.hgetall.return_value = {}

        # Retrieve non-existent record
        record = await valkey_storage._aget_record("non-existent-id")

        # Verify HGETALL was called
        mock_glide_client.hgetall.assert_called_once_with("record:non-existent-id")

        # Verify None is returned
        assert record is None

    @pytest.mark.asyncio
    async def test_retrieve_record_with_empty_embedding(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test retrieving a record with empty embedding."""
        # Mock HGETALL to return record with empty embedding
        mock_glide_client.hgetall.return_value = {
            "id": "no-embedding-record",
            "content": "Content without embedding",
            "scope": "/test",
            "categories": "[]",
            "metadata": "{}",
            "importance": "0.5",
            "created_at": "2024-01-01T12:00:00",
            "last_accessed": "2024-01-01T12:00:00",
            "embedding": b"",  # Empty bytes
            "source": "",
            "private": "false",
        }

        # Retrieve the record
        record = await valkey_storage._aget_record("no-embedding-record")

        # Verify record is retrieved with None embedding
        assert record is not None
        assert record.id == "no-embedding-record"
        assert record.embedding is None

    @pytest.mark.asyncio
    async def test_retrieve_record_with_none_source(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test retrieving a record with None source."""
        # Mock HGETALL to return record with empty source
        mock_glide_client.hgetall.return_value = {
            "id": "no-source-record",
            "content": "Content without source",
            "scope": "/test",
            "categories": "[]",
            "metadata": "{}",
            "importance": "0.5",
            "created_at": "2024-01-01T12:00:00",
            "last_accessed": "2024-01-01T12:00:00",
            "embedding": b"",
            "source": "",  # Empty string
            "private": "false",
        }

        # Retrieve the record
        record = await valkey_storage._aget_record("no-source-record")

        # Verify record is retrieved with None source
        assert record is not None
        assert record.source is None

    @pytest.mark.asyncio
    async def test_retrieve_record_with_false_private_flag(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test retrieving a record with private=false."""
        # Mock HGETALL to return record with private=false
        mock_glide_client.hgetall.return_value = {
            "id": "public-record",
            "content": "Public content",
            "scope": "/test",
            "categories": "[]",
            "metadata": "{}",
            "importance": "0.5",
            "created_at": "2024-01-01T12:00:00",
            "last_accessed": "2024-01-01T12:00:00",
            "embedding": b"",
            "source": "",
            "private": "false",
        }

        # Retrieve the record
        record = await valkey_storage._aget_record("public-record")

        # Verify private flag is False
        assert record is not None
        assert record.private is False

    @pytest.mark.asyncio
    async def test_retrieve_record_with_empty_categories_and_metadata(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test retrieving a record with empty categories and metadata."""
        # Mock HGETALL to return record with empty lists/dicts
        mock_glide_client.hgetall.return_value = {
            "id": "minimal-record",
            "content": "Minimal content",
            "scope": "/test",
            "categories": "[]",
            "metadata": "{}",
            "importance": "0.5",
            "created_at": "2024-01-01T12:00:00",
            "last_accessed": "2024-01-01T12:00:00",
            "embedding": b"",
            "source": "",
            "private": "false",
        }

        # Retrieve the record
        record = await valkey_storage._aget_record("minimal-record")

        # Verify empty collections are preserved
        assert record is not None
        assert record.categories == []
        assert record.metadata == {}

    @pytest.mark.asyncio
    async def test_deserialization_of_datetime_fields(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test deserialization of datetime fields with microseconds."""
        # Mock HGETALL with datetime including microseconds
        mock_glide_client.hgetall.return_value = {
            "id": "datetime-record",
            "content": "Test content",
            "scope": "/test",
            "categories": "[]",
            "metadata": "{}",
            "importance": "0.5",
            "created_at": "2024-01-15T10:30:45.123456",
            "last_accessed": "2024-01-15T11:45:30.654321",
            "embedding": b"",
            "source": "",
            "private": "false",
        }

        # Retrieve the record
        record = await valkey_storage._aget_record("datetime-record")

        # Verify datetime fields are correctly parsed
        assert record is not None
        assert record.created_at == datetime(2024, 1, 15, 10, 30, 45, 123456)
        assert record.last_accessed == datetime(2024, 1, 15, 11, 45, 30, 654321)

    @pytest.mark.asyncio
    async def test_deserialization_of_float_importance(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test deserialization of float importance value."""
        # Mock HGETALL with various float formats
        mock_glide_client.hgetall.return_value = {
            "id": "float-record",
            "content": "Test content",
            "scope": "/test",
            "categories": "[]",
            "metadata": "{}",
            "importance": "0.123456789",
            "created_at": "2024-01-01T12:00:00",
            "last_accessed": "2024-01-01T12:00:00",
            "embedding": b"",
            "source": "",
            "private": "false",
        }

        # Retrieve the record
        record = await valkey_storage._aget_record("float-record")

        # Verify float is correctly parsed
        assert record is not None
        assert abs(record.importance - 0.123456789) < 1e-9

    @pytest.mark.asyncio
    async def test_deserialization_of_json_categories(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test deserialization of JSON categories array."""
        # Mock HGETALL with multiple categories
        mock_glide_client.hgetall.return_value = {
            "id": "categories-record",
            "content": "Test content",
            "scope": "/test",
            "categories": '["planning", "execution", "review", "analysis"]',
            "metadata": "{}",
            "importance": "0.5",
            "created_at": "2024-01-01T12:00:00",
            "last_accessed": "2024-01-01T12:00:00",
            "embedding": b"",
            "source": "",
            "private": "false",
        }

        # Retrieve the record
        record = await valkey_storage._aget_record("categories-record")

        # Verify categories are correctly parsed
        assert record is not None
        assert record.categories == ["planning", "execution", "review", "analysis"]

    @pytest.mark.asyncio
    async def test_deserialization_of_json_metadata(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test deserialization of JSON metadata object."""
        # Mock HGETALL with complex metadata
        mock_glide_client.hgetall.return_value = {
            "id": "metadata-record",
            "content": "Test content",
            "scope": "/test",
            "categories": "[]",
            "metadata": '{"agent_id": "agent-1", "count": 42, "score": 3.14, "active": true}',
            "importance": "0.5",
            "created_at": "2024-01-01T12:00:00",
            "last_accessed": "2024-01-01T12:00:00",
            "embedding": b"",
            "source": "",
            "private": "false",
        }

        # Retrieve the record
        record = await valkey_storage._aget_record("metadata-record")

        # Verify metadata is correctly parsed
        assert record is not None
        assert record.metadata == {
            "agent_id": "agent-1",
            "count": 42,
            "score": 3.14,
            "active": True,
        }

    @pytest.mark.asyncio
    async def test_deserialization_of_binary_embedding(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test deserialization of binary embedding vector."""
        # Create a test embedding
        test_embedding = [0.1, 0.2, 0.3, 0.4, 0.5]
        embedding_bytes = valkey_storage._embedding_to_bytes(test_embedding)

        # Mock HGETALL with binary embedding
        mock_glide_client.hgetall.return_value = {
            "id": "embedding-record",
            "content": "Test content",
            "scope": "/test",
            "categories": "[]",
            "metadata": "{}",
            "importance": "0.5",
            "created_at": "2024-01-01T12:00:00",
            "last_accessed": "2024-01-01T12:00:00",
            "embedding": embedding_bytes,
            "source": "",
            "private": "false",
        }

        # Retrieve the record
        record = await valkey_storage._aget_record("embedding-record")

        # Verify embedding is correctly deserialized
        assert record is not None
        assert record.embedding is not None
        assert len(record.embedding) == 5
        for i, val in enumerate(test_embedding):
            assert abs(record.embedding[i] - val) < 1e-6

    @pytest.mark.asyncio
    async def test_handling_of_malformed_json_categories(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test handling of non-JSON categories uses TAG fallback."""
        # Mock HGETALL with non-JSON categories (treated as TAG format)
        mock_glide_client.hgetall.return_value = {
            "id": "malformed-categories",
            "content": "Test content",
            "scope": "/test",
            "categories": "not valid json [",
            "metadata": "{}",
            "importance": "0.5",
            "created_at": "2024-01-01T12:00:00",
            "last_accessed": "2024-01-01T12:00:00",
            "embedding": b"",
            "source": "",
            "private": "false",
        }

        # Retrieve the record
        record = await valkey_storage._aget_record("malformed-categories")

        # TAG fallback: comma-split produces the raw string as a single category
        assert record is not None
        assert record.id == "malformed-categories"
        assert record.categories == ["not valid json ["]
        mock_glide_client.hgetall.assert_called_once()

    @pytest.mark.asyncio
    async def test_handling_of_malformed_json_metadata(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test handling of malformed JSON in metadata field."""
        # Mock HGETALL with invalid JSON
        mock_glide_client.hgetall.return_value = {
            "id": "malformed-metadata",
            "content": "Test content",
            "scope": "/test",
            "categories": "[]",
            "metadata": "{invalid json}",
            "importance": "0.5",
            "created_at": "2024-01-01T12:00:00",
            "last_accessed": "2024-01-01T12:00:00",
            "embedding": b"",
            "source": "",
            "private": "false",
        }

        # Retrieve the record
        record = await valkey_storage._aget_record("malformed-metadata")

        # Verify None is returned and error is logged
        assert record is None

    @pytest.mark.asyncio
    async def test_handling_of_invalid_datetime_format(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test handling of invalid datetime format."""
        # Mock HGETALL with invalid datetime
        mock_glide_client.hgetall.return_value = {
            "id": "invalid-datetime",
            "content": "Test content",
            "scope": "/test",
            "categories": "[]",
            "metadata": "{}",
            "importance": "0.5",
            "created_at": "not a valid datetime",
            "last_accessed": "2024-01-01T12:00:00",
            "embedding": b"",
            "source": "",
            "private": "false",
        }

        # Retrieve the record
        record = await valkey_storage._aget_record("invalid-datetime")

        # Verify None is returned and error is logged
        assert record is None

    @pytest.mark.asyncio
    async def test_handling_of_invalid_importance_value(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test handling of invalid importance value."""
        # Mock HGETALL with non-numeric importance
        mock_glide_client.hgetall.return_value = {
            "id": "invalid-importance",
            "content": "Test content",
            "scope": "/test",
            "categories": "[]",
            "metadata": "{}",
            "importance": "not a number",
            "created_at": "2024-01-01T12:00:00",
            "last_accessed": "2024-01-01T12:00:00",
            "embedding": b"",
            "source": "",
            "private": "false",
        }

        # Retrieve the record
        record = await valkey_storage._aget_record("invalid-importance")

        # Verify None is returned and error is logged
        assert record is None

    @pytest.mark.asyncio
    async def test_handling_of_missing_required_fields(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test handling of missing required fields."""
        # Mock HGETALL with missing fields
        mock_glide_client.hgetall.return_value = {
            "id": "incomplete-record",
            "content": "Test content",
            # Missing scope, categories, metadata, etc.
        }

        # Retrieve the record
        record = await valkey_storage._aget_record("incomplete-record")

        # Verify None is returned and error is logged
        assert record is None

    @pytest.mark.asyncio
    async def test_handling_of_connection_error(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test handling of connection error during retrieval."""
        # Mock HGETALL to raise connection error
        mock_glide_client.hgetall.side_effect = Exception("Connection failed")

        # Retrieve the record
        record = await valkey_storage._aget_record("test-record")

        # Verify None is returned and error is logged
        assert record is None

    def test_get_record_sync_wrapper(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test that sync get_record wrapper calls async implementation."""
        # Mock HGETALL to return a record
        mock_glide_client.hgetall.return_value = {
            "id": "sync-test-record",
            "content": "Test content",
            "scope": "/test",
            "categories": "[]",
            "metadata": "{}",
            "importance": "0.5",
            "created_at": "2024-01-01T12:00:00",
            "last_accessed": "2024-01-01T12:00:00",
            "embedding": b"",
            "source": "",
            "private": "false",
        }

        # Call sync get_record
        record = valkey_storage.get_record("sync-test-record")

        # Verify async operation was called
        mock_glide_client.hgetall.assert_called_once_with("record:sync-test-record")
        assert record is not None
        assert record.id == "sync-test-record"

    @pytest.mark.asyncio
    async def test_retrieve_record_with_special_characters_in_content(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test retrieving a record with special characters in content."""
        # Mock HGETALL with special characters
        mock_glide_client.hgetall.return_value = {
            "id": "special-chars-record",
            "content": "Content with special chars: \n\t\"quotes\" 'apostrophes' <tags> & symbols",
            "scope": "/test",
            "categories": "[]",
            "metadata": "{}",
            "importance": "0.5",
            "created_at": "2024-01-01T12:00:00",
            "last_accessed": "2024-01-01T12:00:00",
            "embedding": b"",
            "source": "",
            "private": "false",
        }

        # Retrieve the record
        record = await valkey_storage._aget_record("special-chars-record")

        # Verify special characters are preserved
        assert record is not None
        assert "\n" in record.content
        assert "\t" in record.content
        assert '"quotes"' in record.content
        assert "'apostrophes'" in record.content

    @pytest.mark.asyncio
    async def test_retrieve_record_with_unicode_content(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test retrieving a record with unicode content."""
        # Mock HGETALL with unicode characters
        mock_glide_client.hgetall.return_value = {
            "id": "unicode-record",
            "content": "Unicode content: 你好 مرحبا שלום 🚀 ñ é ü",
            "scope": "/test",
            "categories": "[]",
            "metadata": "{}",
            "importance": "0.5",
            "created_at": "2024-01-01T12:00:00",
            "last_accessed": "2024-01-01T12:00:00",
            "embedding": b"",
            "source": "",
            "private": "false",
        }

        # Retrieve the record
        record = await valkey_storage._aget_record("unicode-record")

        # Verify unicode is preserved
        assert record is not None
        assert "你好" in record.content
        assert "🚀" in record.content



class TestValkeyStorageUpdate:
    """Tests for ValkeyStorage update operation."""

    @pytest.mark.asyncio
    async def test_update_existing_record_preserves_created_at(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test updating an existing record preserves created_at timestamp."""
        original_created_at = datetime(2024, 1, 1, 10, 0, 0)
        original_last_accessed = datetime(2024, 1, 1, 11, 0, 0)
        
        # Mock HGETALL to return existing record
        mock_glide_client.hgetall.return_value = {
            "id": "existing-record",
            "content": "Original content",
            "scope": "/original/scope",
            "categories": '["old-category"]',
            "metadata": '{"old_key": "old_value"}',
            "importance": "0.5",
            "created_at": original_created_at.isoformat(),
            "last_accessed": original_last_accessed.isoformat(),
            "embedding": b"",
            "source": "",
            "private": "false",
        }

        # Create updated record with different created_at
        updated_record = MemoryRecord(
            id="existing-record",
            content="Updated content",
            scope="/updated/scope",
            categories=["new-category"],
            metadata={"new_key": "new_value"},
            importance=0.8,
            created_at=datetime(2024, 2, 1, 10, 0, 0),  # Different created_at
            last_accessed=datetime(2024, 2, 1, 11, 0, 0),
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        # Update the record
        await valkey_storage._aupdate(updated_record)

        # Verify HGETALL was called to fetch existing record
        mock_glide_client.hgetall.assert_called_once_with("record:existing-record")

        # Verify HSET was called with updated data
        mock_glide_client.hset.assert_called()
        hset_call = mock_glide_client.hset.call_args
        assert hset_call[0][0] == "record:existing-record"  # key
        hset_dict = hset_call[0][1]  # field_value_map dict
        
        # Verify created_at was preserved from original
        assert hset_dict["created_at"] == original_created_at.isoformat()
        
        # Verify other fields were updated
        assert hset_dict["content"] == "Updated content"
        assert hset_dict["scope"] == "/updated/scope"
        assert hset_dict["importance"] == "0.8"
        
        # Verify last_accessed was updated to current time (not the one in updated_record)
        last_accessed_dt = datetime.fromisoformat(hset_dict["last_accessed"])
        assert last_accessed_dt > original_last_accessed

    @pytest.mark.asyncio
    async def test_update_non_existent_record_creates_new_one(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test updating a non-existent record creates a new one."""
        # Mock HGETALL to return empty dict (record doesn't exist)
        mock_glide_client.hgetall.return_value = {}

        # Create new record
        new_record = MemoryRecord(
            id="new-record",
            content="New content",
            scope="/new/scope",
            categories=["new-category"],
            metadata={"key": "value"},
            importance=0.7,
            created_at=datetime(2024, 1, 1, 10, 0, 0),
            last_accessed=datetime(2024, 1, 1, 11, 0, 0),
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        # Update (create) the record
        await valkey_storage._aupdate(new_record)

        # Verify HGETALL was called
        mock_glide_client.hgetall.assert_called_once_with("record:new-record")

        # Verify HSET was called to create the record
        mock_glide_client.hset.assert_called_once()

        # Verify new indexes were created
        mock_glide_client.zadd.assert_called_once()
        assert mock_glide_client.sadd.call_count == 2  # 1 category + 1 metadata

    @pytest.mark.asyncio
    async def test_update_maintains_index_consistency(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test that update maintains index consistency."""
        # Mock HGETALL to return existing record
        mock_glide_client.hgetall.return_value = {
            "id": "indexed-record",
            "content": "Original content",
            "scope": "/original",
            "categories": '["cat1", "cat2"]',
            "metadata": '{"key1": "value1", "key2": "value2"}',
            "importance": "0.5",
            "created_at": "2024-01-01T10:00:00",
            "last_accessed": "2024-01-01T11:00:00",
            "embedding": b"",
            "source": "",
            "private": "false",
        }

        # Create updated record with same categories and metadata
        updated_record = MemoryRecord(
            id="indexed-record",
            content="Updated content",
            scope="/original",
            categories=["cat1", "cat2"],
            metadata={"key1": "value1", "key2": "value2"},
            importance=0.8,
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        # Update the record
        await valkey_storage._aupdate(updated_record)

        # Verify old indexes were removed
        mock_glide_client.zrem.assert_called_once_with("scope:/original", ["indexed-record"])
        
        # Verify old category indexes were removed
        srem_calls = [call for call in mock_glide_client.srem.call_args_list]
        category_srem_calls = [call for call in srem_calls if "category:" in str(call)]
        assert len(category_srem_calls) == 2
        
        # Verify old metadata indexes were removed
        metadata_srem_calls = [call for call in srem_calls if "metadata:" in str(call)]
        assert len(metadata_srem_calls) == 2

        # Verify new indexes were added
        mock_glide_client.zadd.assert_called_once()
        
        # Verify new category indexes were added
        sadd_calls = [call for call in mock_glide_client.sadd.call_args_list]
        category_sadd_calls = [call for call in sadd_calls if "category:" in str(call[0])]
        assert len(category_sadd_calls) == 2
        
        # Verify new metadata indexes were added
        metadata_sadd_calls = [call for call in sadd_calls if "metadata:" in str(call[0])]
        assert len(metadata_sadd_calls) == 2

    @pytest.mark.asyncio
    async def test_update_removes_from_old_scope_index(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test updating scope removes record from old scope index."""
        # Mock HGETALL to return existing record with old scope
        mock_glide_client.hgetall.return_value = {
            "id": "scope-change-record",
            "content": "Content",
            "scope": "/old/scope",
            "categories": "[]",
            "metadata": "{}",
            "importance": "0.5",
            "created_at": "2024-01-01T10:00:00",
            "last_accessed": "2024-01-01T11:00:00",
            "embedding": b"",
            "source": "",
            "private": "false",
        }

        # Create updated record with new scope
        updated_record = MemoryRecord(
            id="scope-change-record",
            content="Content",
            scope="/new/scope",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        # Update the record
        await valkey_storage._aupdate(updated_record)

        # Verify removed from old scope index
        mock_glide_client.zrem.assert_called_once_with(
            "scope:/old/scope", ["scope-change-record"]
        )

        # Verify added to new scope index
        zadd_call = mock_glide_client.zadd.call_args
        assert zadd_call[0][0] == "scope:/new/scope"
        assert "scope-change-record" in zadd_call[0][1]

    @pytest.mark.asyncio
    async def test_update_removes_from_old_category_indexes(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test updating categories removes record from old category indexes."""
        # Mock HGETALL to return existing record with old categories
        mock_glide_client.hgetall.return_value = {
            "id": "category-change-record",
            "content": "Content",
            "scope": "/test",
            "categories": '["old-cat1", "old-cat2", "shared-cat"]',
            "metadata": "{}",
            "importance": "0.5",
            "created_at": "2024-01-01T10:00:00",
            "last_accessed": "2024-01-01T11:00:00",
            "embedding": b"",
            "source": "",
            "private": "false",
        }

        # Create updated record with new categories (one shared, two new)
        updated_record = MemoryRecord(
            id="category-change-record",
            content="Content",
            scope="/test",
            categories=["new-cat1", "new-cat2", "shared-cat"],
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        # Update the record
        await valkey_storage._aupdate(updated_record)

        # Verify removed from all old category indexes
        srem_calls = [call for call in mock_glide_client.srem.call_args_list]
        category_srem_calls = [call for call in srem_calls if "category:" in str(call)]
        assert len(category_srem_calls) == 3
        
        # Verify removed from old-cat1, old-cat2, and shared-cat
        srem_keys = [call[0][0] for call in category_srem_calls]
        assert "category:old-cat1" in srem_keys
        assert "category:old-cat2" in srem_keys
        assert "category:shared-cat" in srem_keys

        # Verify added to all new category indexes
        sadd_calls = [call for call in mock_glide_client.sadd.call_args_list]
        category_sadd_calls = [call for call in sadd_calls if "category:" in str(call[0])]
        assert len(category_sadd_calls) == 3
        
        # Verify added to new-cat1, new-cat2, and shared-cat
        sadd_keys = [call[0][0] for call in category_sadd_calls]
        assert "category:new-cat1" in sadd_keys
        assert "category:new-cat2" in sadd_keys
        assert "category:shared-cat" in sadd_keys

    @pytest.mark.asyncio
    async def test_update_removes_from_old_metadata_indexes(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test updating metadata removes record from old metadata indexes."""
        # Mock HGETALL to return existing record with old metadata
        mock_glide_client.hgetall.return_value = {
            "id": "metadata-change-record",
            "content": "Content",
            "scope": "/test",
            "categories": "[]",
            "metadata": '{"old_key1": "old_value1", "old_key2": "old_value2", "shared_key": "old_value"}',
            "importance": "0.5",
            "created_at": "2024-01-01T10:00:00",
            "last_accessed": "2024-01-01T11:00:00",
            "embedding": b"",
            "source": "",
            "private": "false",
        }

        # Create updated record with new metadata
        updated_record = MemoryRecord(
            id="metadata-change-record",
            content="Content",
            scope="/test",
            metadata={"new_key1": "new_value1", "new_key2": "new_value2", "shared_key": "new_value"},
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        # Update the record
        await valkey_storage._aupdate(updated_record)

        # Verify removed from all old metadata indexes
        srem_calls = [call for call in mock_glide_client.srem.call_args_list]
        metadata_srem_calls = [call for call in srem_calls if "metadata:" in str(call)]
        assert len(metadata_srem_calls) == 3
        
        # Verify removed from old metadata keys
        srem_keys = [call[0][0] for call in metadata_srem_calls]
        assert "metadata:old_key1:old_value1" in srem_keys
        assert "metadata:old_key2:old_value2" in srem_keys
        assert "metadata:shared_key:old_value" in srem_keys

        # Verify added to all new metadata indexes
        sadd_calls = [call for call in mock_glide_client.sadd.call_args_list]
        metadata_sadd_calls = [call for call in sadd_calls if "metadata:" in str(call[0])]
        assert len(metadata_sadd_calls) == 3
        
        # Verify added to new metadata keys
        sadd_keys = [call[0][0] for call in metadata_sadd_calls]
        assert "metadata:new_key1:new_value1" in sadd_keys
        assert "metadata:new_key2:new_value2" in sadd_keys
        assert "metadata:shared_key:new_value" in sadd_keys

    @pytest.mark.asyncio
    async def test_update_with_empty_categories_removes_all_old_categories(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test updating to empty categories removes all old category indexes."""
        # Mock HGETALL to return existing record with categories
        mock_glide_client.hgetall.return_value = {
            "id": "remove-categories-record",
            "content": "Content",
            "scope": "/test",
            "categories": '["cat1", "cat2", "cat3"]',
            "metadata": "{}",
            "importance": "0.5",
            "created_at": "2024-01-01T10:00:00",
            "last_accessed": "2024-01-01T11:00:00",
            "embedding": b"",
            "source": "",
            "private": "false",
        }

        # Create updated record with empty categories
        updated_record = MemoryRecord(
            id="remove-categories-record",
            content="Content",
            scope="/test",
            categories=[],
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        # Update the record
        await valkey_storage._aupdate(updated_record)

        # Verify removed from all old category indexes
        srem_calls = [call for call in mock_glide_client.srem.call_args_list]
        category_srem_calls = [call for call in srem_calls if "category:" in str(call)]
        assert len(category_srem_calls) == 3

        # Verify no new category indexes were added
        sadd_calls = [call for call in mock_glide_client.sadd.call_args_list]
        category_sadd_calls = [call for call in sadd_calls if "category:" in str(call[0])]
        assert len(category_sadd_calls) == 0

    @pytest.mark.asyncio
    async def test_update_with_empty_metadata_removes_all_old_metadata(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test updating to empty metadata removes all old metadata indexes."""
        # Mock HGETALL to return existing record with metadata
        mock_glide_client.hgetall.return_value = {
            "id": "remove-metadata-record",
            "content": "Content",
            "scope": "/test",
            "categories": "[]",
            "metadata": '{"key1": "value1", "key2": "value2"}',
            "importance": "0.5",
            "created_at": "2024-01-01T10:00:00",
            "last_accessed": "2024-01-01T11:00:00",
            "embedding": b"",
            "source": "",
            "private": "false",
        }

        # Create updated record with empty metadata
        updated_record = MemoryRecord(
            id="remove-metadata-record",
            content="Content",
            scope="/test",
            metadata={},
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        # Update the record
        await valkey_storage._aupdate(updated_record)

        # Verify removed from all old metadata indexes
        srem_calls = [call for call in mock_glide_client.srem.call_args_list]
        metadata_srem_calls = [call for call in srem_calls if "metadata:" in str(call)]
        assert len(metadata_srem_calls) == 2

        # Verify no new metadata indexes were added
        sadd_calls = [call for call in mock_glide_client.sadd.call_args_list]
        metadata_sadd_calls = [call for call in sadd_calls if "metadata:" in str(call[0])]
        assert len(metadata_sadd_calls) == 0

    @pytest.mark.asyncio
    async def test_update_handles_malformed_old_data_gracefully(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test update handles malformed old data gracefully."""
        # Mock HGETALL to return record with malformed JSON
        mock_glide_client.hgetall.return_value = {
            "id": "malformed-record",
            "content": "Content",
            "scope": "/test",
            "categories": "not valid json",
            "metadata": "{invalid json}",
            "importance": "0.5",
            "created_at": "2024-01-01T10:00:00",
            "last_accessed": "2024-01-01T11:00:00",
            "embedding": b"",
            "source": "",
            "private": "false",
        }

        # Create updated record
        updated_record = MemoryRecord(
            id="malformed-record",
            content="Updated content",
            scope="/test",
            categories=["new-cat"],
            metadata={"new_key": "new_value"},
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        # Update should not raise an error
        await valkey_storage._aupdate(updated_record)

        # Verify HSET was called (update proceeded despite malformed old data)
        mock_glide_client.hset.assert_called_once()

        # Verify new indexes were added
        mock_glide_client.zadd.assert_called_once()
        assert mock_glide_client.sadd.call_count >= 2

    @pytest.mark.asyncio
    async def test_update_handles_missing_created_at_gracefully(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test update handles missing created_at in old record gracefully."""
        # Mock HGETALL to return record without created_at
        mock_glide_client.hgetall.return_value = {
            "id": "no-created-at-record",
            "content": "Content",
            "scope": "/test",
            "categories": "[]",
            "metadata": "{}",
            "importance": "0.5",
            # Missing created_at
            "last_accessed": "2024-01-01T11:00:00",
            "embedding": b"",
            "source": "",
            "private": "false",
        }

        # Create updated record with created_at
        updated_record = MemoryRecord(
            id="no-created-at-record",
            content="Updated content",
            scope="/test",
            created_at=datetime(2024, 2, 1, 10, 0, 0),
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        # Update should not raise an error
        await valkey_storage._aupdate(updated_record)

        # Verify HSET was called
        mock_glide_client.hset.assert_called_once()
        
        # Verify created_at from updated_record was used (since old one was missing)
        hset_call = mock_glide_client.hset.call_args
        hset_dict = hset_call[0][1]  # field_value_map dict
        
        # Should use the created_at from updated_record since old one was missing
        assert "created_at" in hset_dict

    @pytest.mark.asyncio
    async def test_update_with_numeric_metadata_values(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test update with numeric metadata values converts to strings."""
        # Mock HGETALL to return existing record
        mock_glide_client.hgetall.return_value = {
            "id": "numeric-metadata-record",
            "content": "Content",
            "scope": "/test",
            "categories": "[]",
            "metadata": '{"count": 10, "score": 5.5}',
            "importance": "0.5",
            "created_at": "2024-01-01T10:00:00",
            "last_accessed": "2024-01-01T11:00:00",
            "embedding": b"",
            "source": "",
            "private": "false",
        }

        # Create updated record with different numeric metadata
        updated_record = MemoryRecord(
            id="numeric-metadata-record",
            content="Content",
            scope="/test",
            metadata={"count": 20, "score": 7.5, "active": True},
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        # Update the record
        await valkey_storage._aupdate(updated_record)

        # Verify removed from old metadata indexes with string-converted values
        srem_calls = [call for call in mock_glide_client.srem.call_args_list]
        metadata_srem_calls = [call for call in srem_calls if "metadata:" in str(call)]
        srem_keys = [call[0][0] for call in metadata_srem_calls]
        assert "metadata:count:10" in srem_keys
        assert "metadata:score:5.5" in srem_keys

        # Verify added to new metadata indexes with string-converted values
        sadd_calls = [call for call in mock_glide_client.sadd.call_args_list]
        metadata_sadd_calls = [call for call in sadd_calls if "metadata:" in str(call[0])]
        sadd_keys = [call[0][0] for call in metadata_sadd_calls]
        assert "metadata:count:20" in sadd_keys
        assert "metadata:score:7.5" in sadd_keys
        assert "metadata:active:True" in sadd_keys

    def test_update_sync_wrapper(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test that sync update wrapper calls async implementation."""
        # Mock HGETALL to return empty dict (new record)
        mock_glide_client.hgetall.return_value = {}

        # Create record
        record = MemoryRecord(
            id="sync-update-record",
            content="Content",
            scope="/test",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        # Call sync update
        valkey_storage.update(record)

        # Verify async operations were called
        mock_glide_client.hgetall.assert_called_once_with("record:sync-update-record")

    @pytest.mark.asyncio
    async def test_update_preserves_embedding(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test that update preserves embedding correctly."""
        # Mock HGETALL to return existing record
        mock_glide_client.hgetall.return_value = {
            "id": "embedding-update-record",
            "content": "Original content",
            "scope": "/test",
            "categories": "[]",
            "metadata": "{}",
            "importance": "0.5",
            "created_at": "2024-01-01T10:00:00",
            "last_accessed": "2024-01-01T11:00:00",
            "embedding": valkey_storage._embedding_to_bytes([0.1, 0.2, 0.3, 0.4]),
            "source": "",
            "private": "false",
        }

        # Create updated record with new embedding
        new_embedding = [0.5, 0.6, 0.7, 0.8]
        updated_record = MemoryRecord(
            id="embedding-update-record",
            content="Updated content",
            scope="/test",
            embedding=new_embedding,
        )

        # Update the record
        await valkey_storage._aupdate(updated_record)

        # Verify HSET was called with new embedding
        mock_glide_client.hset.assert_called_once()
        hset_call = mock_glide_client.hset.call_args
        hset_dict = hset_call[0][1]  # field_value_map dict
        
        # Verify embedding was updated
        assert "embedding" in hset_dict
        # Deserialize and check values
        deserialized_embedding = valkey_storage._bytes_to_embedding(hset_dict["embedding"])
        assert len(deserialized_embedding) == 4
        for i, val in enumerate(new_embedding):
            assert abs(deserialized_embedding[i] - val) < 1e-6


class TestValkeyStorageDelete:
    """Tests for ValkeyStorage delete operation."""

    @pytest.mark.asyncio
    async def test_delete_by_record_ids(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test deleting records by specific record IDs."""
        # Mock record data for deletion
        mock_glide_client.hgetall.side_effect = [
            {
                "id": "record-1",
                "content": "Content 1",
                "scope": "/test",
                "categories": '["cat1", "cat2"]',
                "metadata": '{"key1": "value1"}',
                "importance": "0.5",
                "created_at": "2024-01-01T10:00:00",
                "last_accessed": "2024-01-01T11:00:00",
                "embedding": b"",
                "source": "",
                "private": "false",
            },
            {
                "id": "record-2",
                "content": "Content 2",
                "scope": "/test",
                "categories": '["cat1"]',
                "metadata": '{"key1": "value2"}',
                "importance": "0.6",
                "created_at": "2024-01-01T10:00:00",
                "last_accessed": "2024-01-01T11:00:00",
                "embedding": b"",
                "source": "",
                "private": "false",
            },
        ]

        # Delete by record IDs
        count = await valkey_storage.adelete(record_ids=["record-1", "record-2"])

        # Verify correct count returned
        assert count == 2

        # Verify records were deleted
        delete_calls = [call for call in mock_glide_client.delete.call_args_list]
        assert len(delete_calls) == 2

        # Verify records were removed from scope indexes
        zrem_calls = [call for call in mock_glide_client.zrem.call_args_list]
        assert len(zrem_calls) == 2
        assert any("scope:/test" in str(call) for call in zrem_calls)

        # Verify records were removed from category indexes
        srem_calls = [call for call in mock_glide_client.srem.call_args_list]
        category_srem_calls = [call for call in srem_calls if "category:" in str(call)]
        assert len(category_srem_calls) >= 2  # At least cat1 and cat2

        # Verify records were removed from metadata indexes
        metadata_srem_calls = [call for call in srem_calls if "metadata:" in str(call)]
        assert len(metadata_srem_calls) >= 2  # At least key1:value1 and key1:value2

    @pytest.mark.asyncio
    async def test_delete_by_scope_prefix(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test deleting records by scope prefix."""
        # Mock scan to return scope keys
        mock_glide_client.scan.return_value = (
            b"0",  # cursor (as bytes)
            [b"scope:/agent/task1", b"scope:/agent/task2", b"scope:/other"],
        )

        # Mock zrange calls (used by _find_records_by_scope)
        mock_glide_client.zrange.side_effect = [
            ["record-1", "record-2"],  # zrange scope:/agent/task1
            ["record-3"],  # zrange scope:/agent/task2
            [],  # zrange scope:/other (not matched by prefix)
        ]

        # Mock record data (for _fetch_records_for_deletion)
        mock_glide_client.hgetall.side_effect = [
            {
                b"id": b"record-1",
                b"content": b"Content 1",
                b"scope": b"/agent/task1",
                b"categories": b"[]",
                b"metadata": b"{}",
                b"importance": b"0.5",
                b"created_at": b"2024-01-01T10:00:00",
                b"last_accessed": b"2024-01-01T11:00:00",
                b"embedding": b"",
                b"source": b"",
                b"private": b"false",
            },

            {
                b"id": b"record-2",
                b"content": b"Content 2",
                b"scope": b"/agent/task1",
                b"categories": b"[]",
                b"metadata": b"{}",
                b"importance": b"0.5",
                b"created_at": b"2024-01-01T10:00:00",
                b"last_accessed": b"2024-01-01T11:00:00",
                b"embedding": b"",
                b"source": b"",
                b"private": b"false",
            },
            {
                b"id": b"record-3",
                b"content": b"Content 3",
                b"scope": b"/agent/task2",
                b"categories": b"[]",
                b"metadata": b"{}",
                b"importance": b"0.5",
                b"created_at": b"2024-01-01T10:00:00",
                "last_accessed": "2024-01-01T11:00:00",
                "embedding": b"",
                "source": "",
                "private": "false",
            },
        ]

        # Delete by scope prefix
        count = await valkey_storage.adelete(scope_prefix="/agent")

        # Verify correct count returned (3 records in /agent scopes)
        assert count == 3

        # Verify scan was called to find scope keys
        mock_glide_client.scan.assert_called()

        # Verify zrange was called to get record IDs
        assert mock_glide_client.zrange.call_count >= 2

        # Verify records were deleted
        assert mock_glide_client.delete.call_count == 3


    @pytest.mark.asyncio
    async def test_delete_by_categories(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test deleting records by categories."""
        # Mock smembers to return record IDs for categories
        mock_glide_client.smembers.side_effect = [
            {"record-1", "record-2", "record-3"},  # category:planning
            {"record-2", "record-3", "record-4"},  # category:execution
        ]

        # Mock sinter to return intersection (records with ANY category)
        mock_glide_client.sunion.return_value = {"record-1", "record-2", "record-3", "record-4"}

        # Mock record data
        mock_glide_client.hgetall.side_effect = [
            {
                "id": "record-1",
                "content": "Content 1",
                "scope": "/test",
                "categories": '["planning"]',
                "metadata": "{}",
                "importance": "0.5",
                "created_at": "2024-01-01T10:00:00",
                "last_accessed": "2024-01-01T11:00:00",
                "embedding": b"",
                "source": "",
                "private": "false",
            },
            {
                "id": "record-2",
                "content": "Content 2",
                "scope": "/test",
                "categories": '["planning", "execution"]',
                "metadata": "{}",
                "importance": "0.5",
                "created_at": "2024-01-01T10:00:00",
                "last_accessed": "2024-01-01T11:00:00",
                "embedding": b"",
                "source": "",
                "private": "false",
            },

            {
                "id": "record-3",
                "content": "Content 3",
                "scope": "/test",
                "categories": '["planning", "execution"]',
                "metadata": "{}",
                "importance": "0.5",
                "created_at": "2024-01-01T10:00:00",
                "last_accessed": "2024-01-01T11:00:00",
                "embedding": b"",
                "source": "",
                "private": "false",
            },
            {
                "id": "record-4",
                "content": "Content 4",
                "scope": "/test",
                "categories": '["execution"]',
                "metadata": "{}",
                "importance": "0.5",
                "created_at": "2024-01-01T10:00:00",
                "last_accessed": "2024-01-01T11:00:00",
                "embedding": b"",
                "source": "",
                "private": "false",
            },
        ]

        # Delete by categories (OR logic - any record with planning OR execution)
        count = await valkey_storage.adelete(categories=["planning", "execution"])

        # Verify correct count returned
        assert count == 4

        # Verify records were deleted
        assert mock_glide_client.delete.call_count == 4


    @pytest.mark.asyncio
    async def test_delete_by_older_than(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test deleting records older than a specific datetime."""
        cutoff_date = datetime(2024, 1, 15, 0, 0, 0)
        cutoff_timestamp = cutoff_date.timestamp()

        # Mock scan to return all scope keys
        mock_glide_client.scan.return_value = (
            b"0",  # cursor
            [b"scope:/test"],
        )

        # Mock zrange for ZRANGEBYSCORE to return old records
        mock_glide_client.zrange.return_value = ["record-1", "record-2"]

        # Mock record data
        mock_glide_client.hgetall.side_effect = [
            {
                b"id": b"record-1",
                b"content": b"Old content 1",
                b"scope": b"/test",
                b"categories": b"[]",
                b"metadata": b"{}",
                b"importance": b"0.5",
                b"created_at": b"2024-01-01T10:00:00",
                b"last_accessed": b"2024-01-01T11:00:00",
                b"embedding": b"",
                b"source": b"",
                b"private": b"false",
            },
            {
                b"id": b"record-2",
                b"content": b"Old content 2",
                b"scope": b"/test",
                b"categories": b"[]",
                b"metadata": b"{}",
                b"importance": b"0.5",
                b"created_at": b"2024-01-10T10:00:00",
                b"last_accessed": b"2024-01-10T11:00:00",
                b"embedding": b"",
                b"source": b"",
                b"private": b"false",
            },
        ]

        # Delete records older than cutoff
        count = await valkey_storage.adelete(older_than=cutoff_date)

        # Verify correct count returned
        assert count == 2

        # Verify scan was called
        mock_glide_client.scan.assert_called()

        # Verify zrange was called for score-based range query
        mock_glide_client.zrange.assert_called_once()

        # Verify records were deleted
        assert mock_glide_client.delete.call_count == 2

    @pytest.mark.asyncio
    async def test_delete_by_metadata_filter(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test deleting records by metadata filter."""
        # Mock smembers to return records matching each metadata criterion
        mock_glide_client.smembers.side_effect = [
            {"record-1", "record-2", "record-3"},  # metadata:agent_id:agent-1
            {"record-1", "record-2"},  # metadata:priority:high
        ]

        # Mock record data (only record-1 and record-2 match both criteria)
        mock_glide_client.hgetall.side_effect = [
            {
                "id": "record-1",
                "content": "Content 1",
                "scope": "/test",
                "categories": "[]",
                "metadata": '{"agent_id": "agent-1", "priority": "high"}',
                "importance": "0.5",
                "created_at": "2024-01-01T10:00:00",
                "last_accessed": "2024-01-01T11:00:00",
                "embedding": b"",
                "source": "",
                "private": "false",
            },
            {
                "id": "record-2",
                "content": "Content 2",
                "scope": "/test",
                "categories": "[]",
                "metadata": '{"agent_id": "agent-1", "priority": "high"}',
                "importance": "0.5",
                "created_at": "2024-01-01T10:00:00",
                "last_accessed": "2024-01-01T11:00:00",
                "embedding": b"",
                "source": "",
                "private": "false",
            },
        ]

        # Delete by metadata filter (AND logic - both criteria must match)
        count = await valkey_storage.adelete(
            metadata_filter={"agent_id": "agent-1", "priority": "high"}
        )

        # Verify correct count returned (only records matching both criteria)
        assert count == 2

        # Verify smembers was called for each metadata criterion
        assert mock_glide_client.smembers.call_count == 2

        # Verify records were deleted
        assert mock_glide_client.delete.call_count == 2

    @pytest.mark.asyncio
    async def test_delete_with_combined_criteria(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test deleting records with combined criteria (AND logic)."""
        # Mock scan for scope filtering
        mock_glide_client.scan.return_value = (
            b"0",  # cursor (as bytes)
            [b"scope:/agent/task1", b"scope:/agent/task2"],
        )

        # Mock zrange calls (used by _find_records_by_scope)
        mock_glide_client.zrange.side_effect = [
            ["record-1", "record-2", "record-3"],  # zrange scope:/agent/task1
            ["record-4"],  # zrange scope:/agent/task2
        ]

        # Mock smembers for category filtering (returns records with planning category)
        # Only record-1 and record-2 have planning category (not record-4)
        mock_glide_client.smembers.return_value = {"record-1", "record-2"}

        # The AND logic will intersect scope records (1,2,3,4) with category records (1,2)
        # Result: record-1 and record-2 (both in /agent scope AND have planning category)
        # Mock record data for the 2 matching records (for _fetch_records_for_deletion)
        mock_glide_client.hgetall.side_effect = [
            {
                b"id": b"record-1",
                b"content": b"Content 1",
                b"scope": b"/agent/task1",
                b"categories": b'["planning"]',
                b"metadata": b"{}",
                b"importance": b"0.5",
                b"created_at": b"2024-01-01T10:00:00",
                b"last_accessed": b"2024-01-01T11:00:00",
                b"embedding": b"",
                b"source": b"",
                b"private": b"false",
            },
            {
                b"id": b"record-2",
                b"content": b"Content 2",
                b"scope": b"/agent/task1",
                b"categories": b'["planning"]',
                b"metadata": b"{}",
                b"importance": b"0.5",
                b"created_at": b"2024-01-01T10:00:00",
                b"last_accessed": b"2024-01-01T11:00:00",
                b"embedding": b"",
                b"source": b"",
                b"private": b"false",
            },
        ]

        # Mock delete, zrem, srem operations
        mock_glide_client.delete.return_value = 1
        mock_glide_client.zrem.return_value = 1
        mock_glide_client.srem.return_value = 1

        # Delete with combined criteria: scope_prefix AND categories
        count = await valkey_storage.adelete(
            scope_prefix="/agent", categories=["planning"]
        )

        # Verify correct count (only records in /agent scope AND with planning category)
        assert count == 2

        # Verify both scope and category filtering were used
        mock_glide_client.scan.assert_called()
        mock_glide_client.smembers.assert_called()

        # Verify only matching records were deleted
        assert mock_glide_client.delete.call_count == 2

    @pytest.mark.asyncio
    async def test_delete_returns_correct_count(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test that delete returns the correct count of deleted records."""
        # Mock record data
        mock_glide_client.hgetall.side_effect = [
            {
                "id": "record-1",
                "content": "Content 1",
                "scope": "/test",
                "categories": "[]",
                "metadata": "{}",
                "importance": "0.5",
                "created_at": "2024-01-01T10:00:00",
                "last_accessed": "2024-01-01T11:00:00",
                "embedding": b"",
                "source": "",
                "private": "false",
            },

            {
                "id": "record-2",
                "content": "Content 2",
                "scope": "/test",
                "categories": "[]",
                "metadata": "{}",
                "importance": "0.5",
                "created_at": "2024-01-01T10:00:00",
                "last_accessed": "2024-01-01T11:00:00",
                "embedding": b"",
                "source": "",
                "private": "false",
            },
            {
                "id": "record-3",
                "content": "Content 3",
                "scope": "/test",
                "categories": "[]",
                "metadata": "{}",
                "importance": "0.5",
                "created_at": "2024-01-01T10:00:00",
                "last_accessed": "2024-01-01T11:00:00",
                "embedding": b"",
                "source": "",
                "private": "false",
            },
        ]

        # Delete 3 records
        count = await valkey_storage.adelete(
            record_ids=["record-1", "record-2", "record-3"]
        )

        # Verify count is exactly 3
        assert count == 3

    @pytest.mark.asyncio
    async def test_delete_with_no_matching_records_returns_zero(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test that delete returns 0 when no records match criteria."""
        # Mock scan to return no matching scopes
        mock_glide_client.scan.return_value = (b"0", [])

        # Delete with scope that doesn't exist
        count = await valkey_storage.adelete(scope_prefix="/nonexistent")

        # Verify count is 0
        assert count == 0

        # Verify no delete operations were performed
        mock_glide_client.delete.assert_not_called()


    @pytest.mark.asyncio
    async def test_delete_removes_from_all_indexes(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test that delete removes records from all index structures."""
        # Mock record with multiple categories and metadata
        mock_glide_client.hgetall.return_value = {
            "id": "indexed-record",
            "content": "Content",
            "scope": "/agent/task",
            "categories": '["cat1", "cat2", "cat3"]',
            "metadata": '{"key1": "value1", "key2": "value2", "key3": "value3"}',
            "importance": "0.5",
            "created_at": "2024-01-01T10:00:00",
            "last_accessed": "2024-01-01T11:00:00",
            "embedding": b"",
            "source": "",
            "private": "false",
        }

        # Delete the record
        count = await valkey_storage.adelete(record_ids=["indexed-record"])

        # Verify record was deleted
        assert count == 1
        mock_glide_client.delete.assert_called_once_with(["record:indexed-record"])

        # Verify removed from scope index
        mock_glide_client.zrem.assert_called_once_with(
            "scope:/agent/task", ["indexed-record"]
        )

        # Verify removed from all category indexes
        srem_calls = [call for call in mock_glide_client.srem.call_args_list]
        category_srem_calls = [call for call in srem_calls if "category:" in str(call)]
        assert len(category_srem_calls) == 3

        category_keys = [call[0][0] for call in category_srem_calls]
        assert "category:cat1" in category_keys
        assert "category:cat2" in category_keys
        assert "category:cat3" in category_keys

        # Verify removed from all metadata indexes
        metadata_srem_calls = [call for call in srem_calls if "metadata:" in str(call)]
        assert len(metadata_srem_calls) == 3

        metadata_keys = [call[0][0] for call in metadata_srem_calls]
        assert "metadata:key1:value1" in metadata_keys
        assert "metadata:key2:value2" in metadata_keys
        assert "metadata:key3:value3" in metadata_keys


    @pytest.mark.asyncio
    async def test_delete_with_empty_categories_list(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test delete with empty categories list removes no category indexes."""
        # Mock record with no categories
        mock_glide_client.hgetall.return_value = {
            "id": "no-categories-record",
            "content": "Content",
            "scope": "/test",
            "categories": "[]",
            "metadata": "{}",
            "importance": "0.5",
            "created_at": "2024-01-01T10:00:00",
            "last_accessed": "2024-01-01T11:00:00",
            "embedding": b"",
            "source": "",
            "private": "false",
        }

        # Delete the record
        count = await valkey_storage.adelete(record_ids=["no-categories-record"])

        # Verify record was deleted
        assert count == 1

        # Verify no category index removals
        srem_calls = [call for call in mock_glide_client.srem.call_args_list]
        category_srem_calls = [call for call in srem_calls if "category:" in str(call)]
        assert len(category_srem_calls) == 0

    @pytest.mark.asyncio
    async def test_delete_with_empty_metadata_dict(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test delete with empty metadata dict removes no metadata indexes."""
        # Mock record with no metadata
        mock_glide_client.hgetall.return_value = {
            "id": "no-metadata-record",
            "content": "Content",
            "scope": "/test",
            "categories": "[]",
            "metadata": "{}",
            "importance": "0.5",
            "created_at": "2024-01-01T10:00:00",
            "last_accessed": "2024-01-01T11:00:00",
            "embedding": b"",
            "source": "",
            "private": "false",
        }

        # Delete the record
        count = await valkey_storage.adelete(record_ids=["no-metadata-record"])


        # Verify record was deleted
        assert count == 1

        # Verify no metadata index removals
        srem_calls = [call for call in mock_glide_client.srem.call_args_list]
        metadata_srem_calls = [call for call in srem_calls if "metadata:" in str(call)]
        assert len(metadata_srem_calls) == 0

    @pytest.mark.asyncio
    async def test_delete_with_numeric_metadata_values(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test delete with numeric metadata values converts to strings."""
        # Mock record with numeric metadata
        mock_glide_client.hgetall.return_value = {
            "id": "numeric-metadata-record",
            "content": "Content",
            "scope": "/test",
            "categories": "[]",
            "metadata": '{"count": 42, "score": 3.14, "active": true}',
            "importance": "0.5",
            "created_at": "2024-01-01T10:00:00",
            "last_accessed": "2024-01-01T11:00:00",
            "embedding": b"",
            "source": "",
            "private": "false",
        }

        # Delete the record
        count = await valkey_storage.adelete(record_ids=["numeric-metadata-record"])

        # Verify record was deleted
        assert count == 1

        # Verify metadata indexes were removed with string-converted values
        srem_calls = [call for call in mock_glide_client.srem.call_args_list]
        metadata_srem_calls = [call for call in srem_calls if "metadata:" in str(call)]
        metadata_keys = [call[0][0] for call in metadata_srem_calls]

        assert "metadata:count:42" in metadata_keys
        assert "metadata:score:3.14" in metadata_keys
        assert "metadata:active:True" in metadata_keys

    @pytest.mark.asyncio
    async def test_delete_handles_missing_record_data_gracefully(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test delete handles missing record data gracefully."""
        # Mock hgetall to return empty dict (record doesn't exist)
        mock_glide_client.hgetall.return_value = {}

        # Delete non-existent record
        count = await valkey_storage.adelete(record_ids=["non-existent-record"])


        # Verify count is 0 (record not found)
        assert count == 0

        # Verify no delete operations were performed
        mock_glide_client.delete.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_with_no_criteria_returns_zero(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test delete with no criteria specified returns 0."""
        # Delete with no criteria
        count = await valkey_storage.adelete()

        # Verify count is 0
        assert count == 0

        # Verify no operations were performed
        mock_glide_client.delete.assert_not_called()
        mock_glide_client.scan.assert_not_called()

    @pytest.mark.asyncio
    async def test_delete_with_malformed_record_data(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test delete handles malformed record data gracefully."""
        # Mock record with malformed JSON
        mock_glide_client.hgetall.return_value = {
            "id": "malformed-record",
            "content": "Content",
            "scope": "/test",
            "categories": "not valid json",
            "metadata": "{invalid json}",
            "importance": "0.5",
            "created_at": "2024-01-01T10:00:00",
            "last_accessed": "2024-01-01T11:00:00",
            "embedding": b"",
            "source": "",
            "private": "false",
        }

        # Delete should not raise an error
        count = await valkey_storage.adelete(record_ids=["malformed-record"])

        # Verify record was still deleted (best effort)
        assert count == 1
        mock_glide_client.delete.assert_called_once()

    def test_delete_sync_wrapper(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test that sync delete wrapper calls async implementation."""
        # Mock record data
        mock_glide_client.hgetall.return_value = {
            "id": "sync-delete-record",
            "content": "Content",
            "scope": "/test",
            "categories": "[]",
            "metadata": "{}",
            "importance": "0.5",
            "created_at": "2024-01-01T10:00:00",
            "last_accessed": "2024-01-01T11:00:00",
            "embedding": b"",
            "source": "",
            "private": "false",
        }


        # Call sync delete
        count = valkey_storage.delete(record_ids=["sync-delete-record"])

        # Verify async operation was called
        assert count == 1
        mock_glide_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_with_special_characters_in_scope(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test delete with special characters in scope path."""
        # Mock scan to return scope with special characters
        mock_glide_client.scan.return_value = (
            b"0",
            [b"scope:/agent:task/sub-task"],
        )

        # Mock zrange to return record IDs
        mock_glide_client.zrange.return_value = ["record-1"]

        # Mock record data
        mock_glide_client.hgetall.return_value = {
            b"id": b"record-1",
            b"content": b"Content",
            b"scope": b"/agent:task/sub-task",
            b"categories": b"[]",
            b"metadata": b"{}",
            b"importance": b"0.5",
            b"created_at": b"2024-01-01T10:00:00",
            b"last_accessed": b"2024-01-01T11:00:00",
            b"embedding": b"",
            b"source": b"",
            b"private": b"false",
        }

        # Delete by scope with special characters
        count = await valkey_storage.adelete(scope_prefix="/agent:task")

        # Verify record was deleted
        assert count == 1
        mock_glide_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_delete_multiple_records_in_single_scope(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test deleting multiple records in a single scope."""
        # Mock scan to return one scope
        mock_glide_client.scan.return_value = (b"0", [b"scope:/test"])

        # Mock zrange to return multiple record IDs
        mock_glide_client.zrange.return_value = [
            "record-1",
            "record-2",
            "record-3",
            "record-4",
            "record-5",
        ]


        # Mock record data for all records
        mock_glide_client.hgetall.side_effect = [
            {
                b"id": f"record-{i}".encode(),
                b"content": f"Content {i}".encode(),
                b"scope": b"/test",
                b"categories": b"[]",
                b"metadata": b"{}",
                b"importance": b"0.5",
                b"created_at": b"2024-01-01T10:00:00",
                b"last_accessed": b"2024-01-01T11:00:00",
                b"embedding": b"",
                b"source": b"",
                b"private": b"false",
            }
            for i in range(1, 6)
        ]

        # Delete all records in scope
        count = await valkey_storage.adelete(scope_prefix="/test")

        # Verify all 5 records were deleted
        assert count == 5
        assert mock_glide_client.delete.call_count == 5

    @pytest.mark.asyncio
    async def test_delete_with_root_scope(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test deleting records with root scope '/'."""
        # Mock scan to return all scopes
        mock_glide_client.scan.return_value = (
            b"0",
            [b"scope:/", b"scope:/agent", b"scope:/task"],
        )

        # Mock zrange calls
        mock_glide_client.zrange.side_effect = [
            ["record-1"],  # zrange scope:/
            ["record-2"],  # zrange scope:/agent
            ["record-3"],  # zrange scope:/task
        ]

        # Mock record data
        mock_glide_client.hgetall.side_effect = [
            {
                b"id": b"record-1",
                b"content": b"Content 1",
                b"scope": b"/",
                b"categories": b"[]",
                b"metadata": b"{}",
                b"importance": b"0.5",
                b"created_at": b"2024-01-01T10:00:00",
                b"last_accessed": b"2024-01-01T11:00:00",
                b"embedding": b"",
                b"source": b"",
                b"private": b"false",
            },

            {
                b"id": b"record-2",
                b"content": b"Content 2",
                b"scope": b"/agent",
                b"categories": b"[]",
                b"metadata": b"{}",
                b"importance": b"0.5",
                "created_at": "2024-01-01T10:00:00",
                "last_accessed": "2024-01-01T11:00:00",
                "embedding": b"",
                "source": "",
                "private": "false",
            },
            {
                "id": "record-3",
                "content": "Content 3",
                "scope": "/task",
                "categories": "[]",
                "metadata": "{}",
                "importance": "0.5",
                "created_at": "2024-01-01T10:00:00",
                "last_accessed": "2024-01-01T11:00:00",
                "embedding": b"",
                "source": "",
                "private": "false",
            },
        ]

        # Delete all records (root scope matches all)
        count = await valkey_storage.adelete(scope_prefix="/")

        # Verify all records were deleted
        assert count == 3
        assert mock_glide_client.delete.call_count == 3

    @pytest.mark.asyncio
    async def test_delete_preserves_unmatched_records(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test that delete only removes matching records, not all records."""
        # Mock scan to return multiple scopes
        mock_glide_client.scan.return_value = (
            b"0",
            [b"scope:/agent", b"scope:/task"],
        )

        # Mock zrange - only /agent scope matches prefix
        mock_glide_client.zrange.side_effect = [
            ["record-1", "record-2"],  # zrange scope:/agent (matches)
            [],  # zrange scope:/task (doesn't match prefix, but still scanned)
        ]

        # Mock record data only for matching records
        mock_glide_client.hgetall.side_effect = [
            {
                b"id": b"record-1",
                b"content": b"Content 1",
                b"scope": b"/agent",
                b"categories": b"[]",
                b"metadata": b"{}",
                b"importance": b"0.5",
                b"created_at": b"2024-01-01T10:00:00",
                b"last_accessed": b"2024-01-01T11:00:00",
                b"embedding": b"",
                b"source": b"",
                b"private": b"false",
            },

            {
                b"id": b"record-2",
                b"content": b"Content 2",
                b"scope": b"/agent",
                b"categories": b"[]",
                b"metadata": b"{}",
                b"importance": b"0.5",
                b"created_at": b"2024-01-01T10:00:00",
                b"last_accessed": b"2024-01-01T11:00:00",
                b"embedding": b"",
                b"source": b"",
                b"private": b"false",
            },
        ]

        # Delete only records in /agent scope
        count = await valkey_storage.adelete(scope_prefix="/agent")

        # Verify only 2 records were deleted (not records in /task)
        assert count == 2
        assert mock_glide_client.delete.call_count == 2

        # Verify only /agent records were deleted
        delete_calls = [call[0][0][0] for call in mock_glide_client.delete.call_args_list]
        assert "record:record-1" in delete_calls
        assert "record:record-2" in delete_calls



class TestValkeyStorageIndexing:
    """Tests for ValkeyStorage indexing system (_update_indexes and _remove_from_indexes)."""

    @pytest.mark.asyncio
    async def test_update_indexes_with_simple_scope_path(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test scope index updates with a simple scope path."""
        record_id = "test-record-123"
        scope = "/agent/task"
        categories = ["planning"]
        metadata = {"agent_id": "agent-1"}
        timestamp = 1704067200.0

        await valkey_storage._update_indexes(
            record_id, scope, categories, metadata, timestamp
        )

        # Verify scope index was updated
        mock_glide_client.zadd.assert_called_once_with(
            "scope:/agent/task", {record_id: timestamp}
        )

        # Verify category index was updated
        mock_glide_client.sadd.assert_any_call("category:planning", [record_id])

        # Verify metadata index was updated
        mock_glide_client.sadd.assert_any_call("metadata:agent_id:agent-1", [record_id])

    @pytest.mark.asyncio
    async def test_update_indexes_with_root_scope(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test scope index updates with root scope '/'."""
        record_id = "root-record"
        scope = "/"
        categories: list[str] = []
        metadata: dict[str, str] = {}
        timestamp = 1704067200.0

        await valkey_storage._update_indexes(
            record_id, scope, categories, metadata, timestamp
        )

        # Verify root scope index was created correctly
        mock_glide_client.zadd.assert_called_once_with(
            "scope:/", {record_id: timestamp}
        )

        # Verify no category or metadata indexes were created
        assert mock_glide_client.sadd.call_count == 0

    @pytest.mark.asyncio
    async def test_update_indexes_with_nested_scope_path(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test scope index updates with deeply nested scope path."""
        record_id = "nested-record"
        scope = "/agent/task/subtask/step"
        categories: list[str] = []
        metadata: dict[str, str] = {}
        timestamp = 1704067200.0

        await valkey_storage._update_indexes(
            record_id, scope, categories, metadata, timestamp
        )

        # Verify nested scope index was created correctly
        mock_glide_client.zadd.assert_called_once_with(
            "scope:/agent/task/subtask/step", {record_id: timestamp}
        )

    @pytest.mark.asyncio
    async def test_update_indexes_with_scope_containing_special_characters(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test scope index updates with special characters in scope path."""
        record_id = "special-scope-record"
        scope = "/agent:123/task-456/step_789"
        categories: list[str] = []
        metadata: dict[str, str] = {}
        timestamp = 1704067200.0

        await valkey_storage._update_indexes(
            record_id, scope, categories, metadata, timestamp
        )

        # Verify scope with special characters is handled correctly
        mock_glide_client.zadd.assert_called_once_with(
            "scope:/agent:123/task-456/step_789", {record_id: timestamp}
        )

    @pytest.mark.asyncio
    async def test_update_indexes_with_multiple_categories(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test category index updates with multiple categories."""
        record_id = "multi-category-record"
        scope = "/test"
        categories = ["planning", "execution", "review", "analysis"]
        metadata: dict[str, str] = {}
        timestamp = 1704067200.0

        await valkey_storage._update_indexes(
            record_id, scope, categories, metadata, timestamp
        )

        # Verify all category indexes were updated
        assert mock_glide_client.sadd.call_count == 4
        mock_glide_client.sadd.assert_any_call("category:planning", [record_id])
        mock_glide_client.sadd.assert_any_call("category:execution", [record_id])
        mock_glide_client.sadd.assert_any_call("category:review", [record_id])
        mock_glide_client.sadd.assert_any_call("category:analysis", [record_id])

    @pytest.mark.asyncio
    async def test_update_indexes_with_empty_categories(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test category index updates with empty categories list."""
        record_id = "no-categories-record"
        scope = "/test"
        categories: list[str] = []
        metadata: dict[str, str] = {}
        timestamp = 1704067200.0

        await valkey_storage._update_indexes(
            record_id, scope, categories, metadata, timestamp
        )

        # Verify scope index was updated
        mock_glide_client.zadd.assert_called_once()

        # Verify no category indexes were created
        assert mock_glide_client.sadd.call_count == 0

    @pytest.mark.asyncio
    async def test_update_indexes_with_categories_containing_special_characters(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test category index updates with special characters in category names."""
        record_id = "special-category-record"
        scope = "/test"
        categories = ["category:with:colons", "category-with-dashes", "category_with_underscores"]
        metadata: dict[str, str] = {}
        timestamp = 1704067200.0

        await valkey_storage._update_indexes(
            record_id, scope, categories, metadata, timestamp
        )

        # Verify all category indexes were created with special characters preserved
        assert mock_glide_client.sadd.call_count == 3
        mock_glide_client.sadd.assert_any_call("category:category:with:colons", [record_id])
        mock_glide_client.sadd.assert_any_call("category:category-with-dashes", [record_id])
        mock_glide_client.sadd.assert_any_call("category:category_with_underscores", [record_id])

    @pytest.mark.asyncio
    async def test_update_indexes_with_string_metadata_values(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test metadata index updates with string values."""
        record_id = "string-metadata-record"
        scope = "/test"
        categories: list[str] = []
        metadata = {
            "agent_id": "agent-1",
            "task_type": "planning",
            "status": "active",
        }
        timestamp = 1704067200.0

        await valkey_storage._update_indexes(
            record_id, scope, categories, metadata, timestamp
        )

        # Verify all metadata indexes were created
        assert mock_glide_client.sadd.call_count == 3
        mock_glide_client.sadd.assert_any_call("metadata:agent_id:agent-1", [record_id])
        mock_glide_client.sadd.assert_any_call("metadata:task_type:planning", [record_id])
        mock_glide_client.sadd.assert_any_call("metadata:status:active", [record_id])

    @pytest.mark.asyncio
    async def test_update_indexes_with_numeric_metadata_values(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test metadata index updates with numeric values (converted to strings)."""
        record_id = "numeric-metadata-record"
        scope = "/test"
        categories: list[str] = []
        metadata = {
            "count": 42,
            "score": 3.14159,
            "priority": 1,
        }
        timestamp = 1704067200.0

        await valkey_storage._update_indexes(
            record_id, scope, categories, metadata, timestamp
        )

        # Verify metadata values are converted to strings
        assert mock_glide_client.sadd.call_count == 3
        mock_glide_client.sadd.assert_any_call("metadata:count:42", [record_id])
        mock_glide_client.sadd.assert_any_call("metadata:score:3.14159", [record_id])
        mock_glide_client.sadd.assert_any_call("metadata:priority:1", [record_id])

    @pytest.mark.asyncio
    async def test_update_indexes_with_boolean_metadata_values(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test metadata index updates with boolean values (converted to strings)."""
        record_id = "boolean-metadata-record"
        scope = "/test"
        categories: list[str] = []
        metadata = {
            "is_active": True,
            "is_complete": False,
        }
        timestamp = 1704067200.0

        await valkey_storage._update_indexes(
            record_id, scope, categories, metadata, timestamp
        )

        # Verify boolean values are converted to strings
        assert mock_glide_client.sadd.call_count == 2
        mock_glide_client.sadd.assert_any_call("metadata:is_active:True", [record_id])
        mock_glide_client.sadd.assert_any_call("metadata:is_complete:False", [record_id])

    @pytest.mark.asyncio
    async def test_update_indexes_with_empty_metadata(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test metadata index updates with empty metadata dict."""
        record_id = "no-metadata-record"
        scope = "/test"
        categories: list[str] = []
        metadata: dict[str, str] = {}
        timestamp = 1704067200.0

        await valkey_storage._update_indexes(
            record_id, scope, categories, metadata, timestamp
        )

        # Verify scope index was updated
        mock_glide_client.zadd.assert_called_once()

        # Verify no metadata indexes were created
        assert mock_glide_client.sadd.call_count == 0

    @pytest.mark.asyncio
    async def test_update_indexes_with_metadata_containing_special_characters(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test metadata index updates with special characters in keys and values."""
        record_id = "special-metadata-record"
        scope = "/test"
        categories: list[str] = []
        metadata = {
            "key:with:colons": "value:with:colons",
            "key-with-dashes": "value-with-dashes",
            "key_with_underscores": "value_with_underscores",
            "key with spaces": "value with spaces",
        }
        timestamp = 1704067200.0

        await valkey_storage._update_indexes(
            record_id, scope, categories, metadata, timestamp
        )

        # Verify all metadata indexes were created with special characters preserved
        assert mock_glide_client.sadd.call_count == 4
        mock_glide_client.sadd.assert_any_call(
            "metadata:key:with:colons:value:with:colons", [record_id]
        )
        mock_glide_client.sadd.assert_any_call(
            "metadata:key-with-dashes:value-with-dashes", [record_id]
        )
        mock_glide_client.sadd.assert_any_call(
            "metadata:key_with_underscores:value_with_underscores", [record_id]
        )
        mock_glide_client.sadd.assert_any_call(
            "metadata:key with spaces:value with spaces", [record_id]
        )

    @pytest.mark.asyncio
    async def test_update_indexes_with_mixed_data_types_in_metadata(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test metadata index updates with mixed data types."""
        record_id = "mixed-metadata-record"
        scope = "/test"
        categories: list[str] = []
        metadata = {
            "string_key": "string_value",
            "int_key": 123,
            "float_key": 45.67,
            "bool_key": True,
        }
        timestamp = 1704067200.0

        await valkey_storage._update_indexes(
            record_id, scope, categories, metadata, timestamp
        )

        # Verify all metadata indexes were created with proper type conversion
        assert mock_glide_client.sadd.call_count == 4
        mock_glide_client.sadd.assert_any_call("metadata:string_key:string_value", [record_id])
        mock_glide_client.sadd.assert_any_call("metadata:int_key:123", [record_id])
        mock_glide_client.sadd.assert_any_call("metadata:float_key:45.67", [record_id])
        mock_glide_client.sadd.assert_any_call("metadata:bool_key:True", [record_id])

    @pytest.mark.asyncio
    async def test_update_indexes_with_all_fields_populated(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test index updates with scope, categories, and metadata all populated."""
        record_id = "full-record"
        scope = "/agent/task"
        categories = ["planning", "execution"]
        metadata = {"agent_id": "agent-1", "priority": "high"}
        timestamp = 1704067200.0

        await valkey_storage._update_indexes(
            record_id, scope, categories, metadata, timestamp
        )

        # Verify scope index was updated
        mock_glide_client.zadd.assert_called_once_with(
            "scope:/agent/task", {record_id: timestamp}
        )

        # Verify all indexes were updated (2 categories + 2 metadata = 4 sadd calls)
        assert mock_glide_client.sadd.call_count == 4

    @pytest.mark.asyncio
    async def test_remove_from_indexes_with_simple_scope(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test removing record from indexes with simple scope."""
        record_id = "test-record-123"
        scope = "/agent/task"
        categories = ["planning"]
        metadata = {"agent_id": "agent-1"}

        await valkey_storage._remove_from_indexes(
            record_id, scope, categories, metadata
        )

        # Verify record was removed from scope index
        mock_glide_client.zrem.assert_called_once_with("scope:/agent/task", [record_id])

        # Verify record was removed from category index
        mock_glide_client.srem.assert_any_call("category:planning", [record_id])

        # Verify record was removed from metadata index
        mock_glide_client.srem.assert_any_call("metadata:agent_id:agent-1", [record_id])

    @pytest.mark.asyncio
    async def test_remove_from_indexes_with_root_scope(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test removing record from indexes with root scope '/'."""
        record_id = "root-record"
        scope = "/"
        categories: list[str] = []
        metadata: dict[str, str] = {}

        await valkey_storage._remove_from_indexes(
            record_id, scope, categories, metadata
        )

        # Verify record was removed from root scope index
        mock_glide_client.zrem.assert_called_once_with("scope:/", [record_id])

        # Verify no category or metadata removals
        assert mock_glide_client.srem.call_count == 0

    @pytest.mark.asyncio
    async def test_remove_from_indexes_with_multiple_categories(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test removing record from multiple category indexes."""
        record_id = "multi-category-record"
        scope = "/test"
        categories = ["planning", "execution", "review"]
        metadata: dict[str, str] = {}

        await valkey_storage._remove_from_indexes(
            record_id, scope, categories, metadata
        )

        # Verify record was removed from all category indexes
        assert mock_glide_client.srem.call_count == 3
        mock_glide_client.srem.assert_any_call("category:planning", [record_id])
        mock_glide_client.srem.assert_any_call("category:execution", [record_id])
        mock_glide_client.srem.assert_any_call("category:review", [record_id])

    @pytest.mark.asyncio
    async def test_remove_from_indexes_with_empty_categories(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test removing record with empty categories list."""
        record_id = "no-categories-record"
        scope = "/test"
        categories: list[str] = []
        metadata: dict[str, str] = {}

        await valkey_storage._remove_from_indexes(
            record_id, scope, categories, metadata
        )

        # Verify scope removal
        mock_glide_client.zrem.assert_called_once()

        # Verify no category removals
        assert mock_glide_client.srem.call_count == 0

    @pytest.mark.asyncio
    async def test_remove_from_indexes_with_multiple_metadata_entries(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test removing record from multiple metadata indexes."""
        record_id = "multi-metadata-record"
        scope = "/test"
        categories: list[str] = []
        metadata = {
            "agent_id": "agent-1",
            "task_type": "planning",
            "priority": "high",
        }

        await valkey_storage._remove_from_indexes(
            record_id, scope, categories, metadata
        )

        # Verify record was removed from all metadata indexes
        assert mock_glide_client.srem.call_count == 3
        mock_glide_client.srem.assert_any_call("metadata:agent_id:agent-1", [record_id])
        mock_glide_client.srem.assert_any_call("metadata:task_type:planning", [record_id])
        mock_glide_client.srem.assert_any_call("metadata:priority:high", [record_id])

    @pytest.mark.asyncio
    async def test_remove_from_indexes_with_empty_metadata(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test removing record with empty metadata dict."""
        record_id = "no-metadata-record"
        scope = "/test"
        categories: list[str] = []
        metadata: dict[str, str] = {}

        await valkey_storage._remove_from_indexes(
            record_id, scope, categories, metadata
        )

        # Verify scope removal
        mock_glide_client.zrem.assert_called_once()

        # Verify no metadata removals
        assert mock_glide_client.srem.call_count == 0

    @pytest.mark.asyncio
    async def test_remove_from_indexes_with_numeric_metadata_values(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test removing record with numeric metadata values (converted to strings)."""
        record_id = "numeric-metadata-record"
        scope = "/test"
        categories: list[str] = []
        metadata = {
            "count": 42,
            "score": 3.14,
        }

        await valkey_storage._remove_from_indexes(
            record_id, scope, categories, metadata
        )

        # Verify metadata values are converted to strings for removal
        assert mock_glide_client.srem.call_count == 2
        mock_glide_client.srem.assert_any_call("metadata:count:42", [record_id])
        mock_glide_client.srem.assert_any_call("metadata:score:3.14", [record_id])

    @pytest.mark.asyncio
    async def test_remove_from_indexes_with_boolean_metadata_values(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test removing record with boolean metadata values (converted to strings)."""
        record_id = "boolean-metadata-record"
        scope = "/test"
        categories: list[str] = []
        metadata = {
            "is_active": True,
            "is_complete": False,
        }

        await valkey_storage._remove_from_indexes(
            record_id, scope, categories, metadata
        )

        # Verify boolean values are converted to strings for removal
        assert mock_glide_client.srem.call_count == 2
        mock_glide_client.srem.assert_any_call("metadata:is_active:True", [record_id])
        mock_glide_client.srem.assert_any_call("metadata:is_complete:False", [record_id])

    @pytest.mark.asyncio
    async def test_remove_from_indexes_with_all_fields_populated(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test removing record from all index structures."""
        record_id = "full-record"
        scope = "/agent/task"
        categories = ["planning", "execution"]
        metadata = {"agent_id": "agent-1", "priority": "high"}

        await valkey_storage._remove_from_indexes(
            record_id, scope, categories, metadata
        )

        # Verify scope removal
        mock_glide_client.zrem.assert_called_once_with("scope:/agent/task", [record_id])

        # Verify all removals (2 categories + 2 metadata = 4 srem calls)
        assert mock_glide_client.srem.call_count == 4

    @pytest.mark.asyncio
    async def test_remove_from_indexes_cleans_all_structures(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test that remove_from_indexes cleans all index structures completely."""
        record_id = "cleanup-record"
        scope = "/agent/task/subtask"
        categories = ["planning", "execution", "review"]
        metadata = {
            "agent_id": "agent-1",
            "task_type": "analysis",
            "priority": 5,
            "active": True,
        }

        await valkey_storage._remove_from_indexes(
            record_id, scope, categories, metadata
        )

        # Verify scope index cleanup
        mock_glide_client.zrem.assert_called_once_with(
            "scope:/agent/task/subtask", [record_id]
        )

        # Verify category index cleanup (3 categories)
        mock_glide_client.srem.assert_any_call("category:planning", [record_id])
        mock_glide_client.srem.assert_any_call("category:execution", [record_id])
        mock_glide_client.srem.assert_any_call("category:review", [record_id])

        # Verify metadata index cleanup (4 metadata entries)
        mock_glide_client.srem.assert_any_call("metadata:agent_id:agent-1", [record_id])
        mock_glide_client.srem.assert_any_call("metadata:task_type:analysis", [record_id])
        mock_glide_client.srem.assert_any_call("metadata:priority:5", [record_id])
        mock_glide_client.srem.assert_any_call("metadata:active:True", [record_id])

        # Verify total number of removals (3 categories + 4 metadata = 7 srem calls)
        assert mock_glide_client.srem.call_count == 7

    @pytest.mark.asyncio
    async def test_update_then_remove_indexes_consistency(
        self, valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test that update and remove operations use consistent key naming."""
        record_id = "consistency-record"
        scope = "/test/scope"
        categories = ["cat1", "cat2"]
        metadata = {"key1": "value1", "key2": 123}
        timestamp = 1704067200.0

        # Update indexes
        await valkey_storage._update_indexes(
            record_id, scope, categories, metadata, timestamp
        )

        # Capture the keys used in update
        zadd_key = mock_glide_client.zadd.call_args[0][0]
        sadd_keys = [call[0][0] for call in mock_glide_client.sadd.call_args_list]

        # Reset mocks
        mock_glide_client.reset_mock()

        # Remove from indexes
        await valkey_storage._remove_from_indexes(
            record_id, scope, categories, metadata
        )

        # Capture the keys used in remove
        zrem_key = mock_glide_client.zrem.call_args[0][0]
        srem_keys = [call[0][0] for call in mock_glide_client.srem.call_args_list]

        # Verify scope keys match
        assert zadd_key == zrem_key

        # Verify category and metadata keys match
        assert set(sadd_keys) == set(srem_keys)
