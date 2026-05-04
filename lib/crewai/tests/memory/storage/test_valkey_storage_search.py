"""Tests for ValkeyStorage vector search operation."""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import AsyncMock, patch
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


def create_mock_ft_search_response(
    records: list[tuple[MemoryRecord, float]]
) -> list[int | dict[str, dict[str, str]]]:
    """Create a mock FT.SEARCH response in native dict format.

    Args:
        records: List of (MemoryRecord, score) tuples to include in response.

    Returns:
        Mock FT.SEARCH response in the native format:
        [total_count, {doc_key: {field: value, ...}, ...}]
    """
    if not records:
        return [0]

    docs: dict[str, dict[str, str]] = {}

    for record, score in records:
        doc_key = f"record:{record.id}"

        # Build field dict
        fields: dict[str, str] = {}
        fields["id"] = record.id
        fields["content"] = record.content
        fields["scope"] = record.scope
        fields["categories"] = json.dumps(record.categories)
        fields["metadata"] = json.dumps(record.metadata)
        fields["importance"] = str(record.importance)
        fields["created_at"] = record.created_at.isoformat()
        fields["last_accessed"] = record.last_accessed.isoformat()
        fields["source"] = record.source or ""
        fields["private"] = "true" if record.private else "false"

        # Add score (Valkey Search returns cosine distance, not similarity)
        # Convert similarity to distance: distance = 2 * (1 - similarity)
        distance = 2.0 * (1.0 - score)
        fields["score"] = str(distance)

        # Add embedding if present
        if record.embedding:
            fields["embedding"] = json.dumps(record.embedding)

        docs[doc_key] = fields

    return [len(records), docs]


class TestValkeyStorageVectorSearch:
    """Tests for ValkeyStorage vector search operation."""

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_search_with_no_filters_returns_all_records(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test search with no filters returns all records."""
        # Create test records
        record1 = MemoryRecord(
            id="record-1",
            content="First test record",
            scope="/test",
            categories=["cat1"],
            metadata={"key": "value1"},
            importance=0.8,
            created_at=datetime(2024, 1, 1, 10, 0, 0),
            last_accessed=datetime(2024, 1, 1, 11, 0, 0),
            embedding=[0.1, 0.2, 0.3, 0.4],
        )
        record2 = MemoryRecord(
            id="record-2",
            content="Second test record",
            scope="/test",
            categories=["cat2"],
            metadata={"key": "value2"},
            importance=0.6,
            created_at=datetime(2024, 1, 2, 10, 0, 0),
            last_accessed=datetime(2024, 1, 2, 11, 0, 0),
            embedding=[0.2, 0.3, 0.4, 0.5],
        )

        # Mock FT.INFO to simulate index exists
        mock_ft_list.return_value = [b"memory_index"]
        # Mock FT.SEARCH to return both records
        mock_ft_search.return_value = create_mock_ft_search_response([
            (record1, 0.95),
            (record2, 0.85),
        ])

        # Perform search with no filters
        query_embedding = [0.1, 0.2, 0.3, 0.4]
        results = await valkey_storage.asearch(query_embedding, limit=10)

        # Verify ft.search was called
        mock_ft_search.assert_called_once()

        # Verify query contains only KNN part (no filters)
        call_args = mock_ft_search.call_args
        query = call_args[0][2]  # 3rd positional arg: query string
        assert "*=>[KNN 10 @embedding $BLOB AS score]" in query
        assert "@scope" not in query
        assert "@categories" not in query

        # Verify results
        assert len(results) == 2
        assert results[0][0].id == "record-1"
        assert results[0][1] == 0.95
        assert results[1][0].id == "record-2"
        assert results[1][1] == 0.85

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_search_with_scope_filter_only(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test search with scope filter only."""
        record1 = MemoryRecord(
            id="record-1",
            content="Record in scope",
            scope="/agent/task",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        mock_ft_list.return_value = [b"memory_index"]
        mock_ft_search.return_value = create_mock_ft_search_response([(record1, 0.9)])

        query_embedding = [0.1, 0.2, 0.3, 0.4]
        results = await valkey_storage.asearch(
            query_embedding,
            scope_prefix="/agent",
            limit=10
        )

        # Verify query contains scope filter
        call_args = mock_ft_search.call_args
        query = call_args[0][2]
        assert "(@scope:{/agent*})=>[KNN 10 @embedding $BLOB AS score]" in query

        # Verify results
        assert len(results) == 1
        assert results[0][0].id == "record-1"
        assert results[0][0].scope == "/agent/task"

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_search_with_category_filter_only(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test search with category filter only."""
        record1 = MemoryRecord(
            id="record-1",
            content="Record with planning category",
            scope="/test",
            categories=["planning"],
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        mock_ft_list.return_value = [b"memory_index"]
        mock_ft_search.return_value = create_mock_ft_search_response([(record1, 0.88)])

        query_embedding = [0.1, 0.2, 0.3, 0.4]
        results = await valkey_storage.asearch(
            query_embedding,
            categories=["planning", "execution"],
            limit=10
        )

        # Verify query contains category filter with OR logic
        call_args = mock_ft_search.call_args
        query = call_args[0][2]
        assert "(@categories:{planning|execution})=>[KNN 10 @embedding $BLOB AS score]" in query

        # Verify results
        assert len(results) == 1
        assert results[0][0].id == "record-1"
        assert "planning" in results[0][0].categories

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_search_with_metadata_filter_only(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test search with metadata filter only."""
        record1 = MemoryRecord(
            id="record-1",
            content="Record with metadata",
            scope="/test",
            metadata={"agent_id": "agent-1", "priority": "high"},
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        mock_ft_list.return_value = [b"memory_index"]
        mock_ft_search.return_value = create_mock_ft_search_response([(record1, 0.92)])

        query_embedding = [0.1, 0.2, 0.3, 0.4]
        results = await valkey_storage.asearch(
            query_embedding,
            metadata_filter={"agent_id": "agent-1", "priority": "high"},
            limit=10
        )

        # Verify query contains metadata filters (AND logic)
        call_args = mock_ft_search.call_args
        query = call_args[0][2]
        assert "@agent_id:{agent\\-1}" in query or "@agent_id:{agent-1}" in query
        assert "@priority:{high}" in query
        assert "=>[KNN 10 @embedding $BLOB AS score]" in query

        # Verify results
        assert len(results) == 1
        assert results[0][0].id == "record-1"
        assert results[0][0].metadata["agent_id"] == "agent-1"
        assert results[0][0].metadata["priority"] == "high"

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_search_with_combined_filters(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test search with combined filters (scope + categories + metadata)."""
        record1 = MemoryRecord(
            id="record-1",
            content="Record matching all filters",
            scope="/agent/task",
            categories=["planning"],
            metadata={"agent_id": "agent-1"},
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        mock_ft_list.return_value = [b"memory_index"]
        mock_ft_search.return_value = create_mock_ft_search_response([(record1, 0.93)])

        query_embedding = [0.1, 0.2, 0.3, 0.4]
        results = await valkey_storage.asearch(
            query_embedding,
            scope_prefix="/agent",
            categories=["planning"],
            metadata_filter={"agent_id": "agent-1"},
            limit=10
        )

        # Verify query contains all filters
        call_args = mock_ft_search.call_args
        query = call_args[0][2]
        assert "@scope:{/agent*}" in query
        assert "@categories:{planning}" in query
        assert "@agent_id:{agent\\-1}" in query or "@agent_id:{agent-1}" in query
        assert "=>[KNN 10 @embedding $BLOB AS score]" in query

        # Verify results
        assert len(results) == 1
        assert results[0][0].id == "record-1"

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_search_respects_limit_parameter(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test search respects limit parameter."""
        records = [
            (
                MemoryRecord(
                    id=f"record-{i}",
                    content=f"Record {i}",
                    scope="/test",
                    embedding=[0.1 * i, 0.2 * i, 0.3 * i, 0.4 * i],
                ),
                0.9 - (i * 0.1)
            )
            for i in range(1, 6)
        ]

        mock_ft_list.return_value = [b"memory_index"]
        mock_ft_search.return_value = create_mock_ft_search_response(records[:3])

        query_embedding = [0.1, 0.2, 0.3, 0.4]
        results = await valkey_storage.asearch(query_embedding, limit=3)

        # Verify KNN limit in query
        call_args = mock_ft_search.call_args
        query = call_args[0][2]
        assert "=>[KNN 3 @embedding $BLOB AS score]" in query

        # Verify results respect limit
        assert len(results) == 3

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_search_respects_min_score_parameter(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test search respects min_score parameter."""
        record1 = MemoryRecord(
            id="record-1",
            content="High score record",
            scope="/test",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )
        record2 = MemoryRecord(
            id="record-2",
            content="Medium score record",
            scope="/test",
            embedding=[0.2, 0.3, 0.4, 0.5],
        )
        record3 = MemoryRecord(
            id="record-3",
            content="Low score record",
            scope="/test",
            embedding=[0.3, 0.4, 0.5, 0.6],
        )

        mock_ft_list.return_value = [b"memory_index"]
        mock_ft_search.return_value = create_mock_ft_search_response([
            (record1, 0.95),
            (record2, 0.75),
            (record3, 0.55),
        ])

        query_embedding = [0.1, 0.2, 0.3, 0.4]
        results = await valkey_storage.asearch(
            query_embedding,
            limit=10,
            min_score=0.7
        )

        # Verify only records with score >= 0.7 are returned
        assert len(results) == 2
        assert results[0][0].id == "record-1"
        assert results[0][1] == 0.95
        assert results[1][0].id == "record-2"
        assert results[1][1] == 0.75

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_search_returns_results_ordered_by_descending_score(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test search returns results ordered by descending score."""
        record1 = MemoryRecord(
            id="record-1",
            content="Medium score",
            scope="/test",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )
        record2 = MemoryRecord(
            id="record-2",
            content="Highest score",
            scope="/test",
            embedding=[0.2, 0.3, 0.4, 0.5],
        )
        record3 = MemoryRecord(
            id="record-3",
            content="Lowest score",
            scope="/test",
            embedding=[0.3, 0.4, 0.5, 0.6],
        )

        mock_ft_list.return_value = [b"memory_index"]
        mock_ft_search.return_value = create_mock_ft_search_response([
            (record1, 0.75),
            (record2, 0.95),
            (record3, 0.55),
        ])

        query_embedding = [0.1, 0.2, 0.3, 0.4]
        results = await valkey_storage.asearch(query_embedding, limit=10)

        # Verify results are ordered by descending score
        assert len(results) == 3
        assert results[0][0].id == "record-2"
        assert results[0][1] == 0.95
        assert results[1][0].id == "record-1"
        assert results[1][1] == 0.75
        assert results[2][0].id == "record-3"
        assert results[2][1] == 0.55

        # Verify scores are in descending order
        for i in range(len(results) - 1):
            assert results[i][1] >= results[i + 1][1]

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_search_with_empty_results(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test search with no matching results."""
        mock_ft_list.return_value = [b"memory_index"]
        mock_ft_search.return_value = [0]  # Total count = 0

        query_embedding = [0.1, 0.2, 0.3, 0.4]
        results = await valkey_storage.asearch(query_embedding, limit=10)

        # Verify empty results
        assert len(results) == 0

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_search_with_special_characters_in_scope(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test search with special characters in scope prefix."""
        record1 = MemoryRecord(
            id="record-1",
            content="Record with special scope",
            scope="/agent:task-1",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        mock_ft_list.return_value = [b"memory_index"]
        mock_ft_search.return_value = create_mock_ft_search_response([(record1, 0.9)])

        query_embedding = [0.1, 0.2, 0.3, 0.4]
        results = await valkey_storage.asearch(
            query_embedding,
            scope_prefix="/agent:task",
            limit=10
        )

        # Verify query contains escaped scope
        call_args = mock_ft_search.call_args
        query = call_args[0][2]
        assert "@scope:{/agent\\:task*}" in query or "@scope:{/agent:task*}" in query

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_search_with_special_characters_in_categories(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test search with special characters in categories."""
        record1 = MemoryRecord(
            id="record-1",
            content="Record with special category",
            scope="/test",
            categories=["plan:execute"],
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        mock_ft_list.return_value = [b"memory_index"]
        mock_ft_search.return_value = create_mock_ft_search_response([(record1, 0.9)])

        query_embedding = [0.1, 0.2, 0.3, 0.4]
        results = await valkey_storage.asearch(
            query_embedding,
            categories=["plan:execute"],
            limit=10
        )

        # Verify query contains escaped category
        call_args = mock_ft_search.call_args
        query = call_args[0][2]
        assert "@categories:{plan\\:execute}" in query or "@categories:{plan:execute}" in query

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_search_with_numeric_metadata_values(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test search with numeric metadata values."""
        record1 = MemoryRecord(
            id="record-1",
            content="Record with numeric metadata",
            scope="/test",
            metadata={"count": 42, "score": 3.14},
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        mock_ft_list.return_value = [b"memory_index"]
        mock_ft_search.return_value = create_mock_ft_search_response([(record1, 0.9)])

        query_embedding = [0.1, 0.2, 0.3, 0.4]
        results = await valkey_storage.asearch(
            query_embedding,
            metadata_filter={"count": 42, "score": 3.14},
            limit=10
        )

        # Verify query contains string-converted metadata values
        call_args = mock_ft_search.call_args
        query = call_args[0][2]
        assert "@count:{42}" in query
        assert "@score:{3" in query and "14}" in query

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_search_with_embedding_blob_parameter(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test search passes embedding as BLOB parameter."""
        record1 = MemoryRecord(
            id="record-1",
            content="Test record",
            scope="/test",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        mock_ft_list.return_value = [b"memory_index"]
        mock_ft_search.return_value = create_mock_ft_search_response([(record1, 0.9)])

        query_embedding = [0.1, 0.2, 0.3, 0.4]
        results = await valkey_storage.asearch(query_embedding, limit=10)

        # Verify ft.search was called with search options containing BLOB param
        call_args = mock_ft_search.call_args
        # The 4th positional arg is the FtSearchOptions
        search_options = call_args[0][3]
        # The options object should have params with BLOB
        assert search_options is not None

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_search_results_sorted_by_score(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test search results are sorted by score (descending) automatically."""
        record1 = MemoryRecord(
            id="record-1",
            content="Test record",
            scope="/test",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        mock_ft_list.return_value = [b"memory_index"]
        mock_ft_search.return_value = create_mock_ft_search_response([(record1, 0.9)])

        query_embedding = [0.1, 0.2, 0.3, 0.4]
        results = await valkey_storage.asearch(query_embedding, limit=10)

        # Verify ft.search was called (results are auto-sorted by vector search)
        mock_ft_search.assert_called_once()

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_search_with_return_fields(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test search includes RETURN clause with all record fields."""
        record1 = MemoryRecord(
            id="record-1",
            content="Test record",
            scope="/test",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        mock_ft_list.return_value = [b"memory_index"]
        mock_ft_search.return_value = create_mock_ft_search_response([(record1, 0.9)])

        query_embedding = [0.1, 0.2, 0.3, 0.4]
        results = await valkey_storage.asearch(query_embedding, limit=10)

        # Verify ft.search was called with search options containing return fields
        call_args = mock_ft_search.call_args
        search_options = call_args[0][3]
        assert search_options is not None
        # The FtSearchOptions should have return_fields set
        assert search_options.return_fields is not None
        assert len(search_options.return_fields) == 11  # All fields including score

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.VectorFieldAttributesHnsw")
    @patch("crewai.memory.storage.valkey_storage.ft.create")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_search_handles_valkey_search_not_available(
        self, mock_ft_list: AsyncMock, mock_ft_create: AsyncMock,
        mock_vector_attrs: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test search raises error when Valkey Search module is not available."""
        # Mock FT.INFO to fail (index doesn't exist)
        mock_ft_list.return_value = []
        # Mock FT.CREATE to fail (Search module not available)
        mock_ft_create.side_effect = Exception("ERR unknown command 'ft.create'")

        query_embedding = [0.1, 0.2, 0.3, 0.4]

        with pytest.raises(RuntimeError, match="Valkey Search module is not available"):
            await valkey_storage.asearch(query_embedding, limit=10)

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_search_handles_ft_search_error(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test search handles FT.SEARCH errors gracefully."""
        mock_ft_list.return_value = [b"memory_index"]
        mock_ft_search.side_effect = Exception("ERR unknown command 'FT.SEARCH'")

        query_embedding = [0.1, 0.2, 0.3, 0.4]

        with pytest.raises(RuntimeError, match="Valkey Search module is not available"):
            await valkey_storage.asearch(query_embedding, limit=10)

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_search_handles_malformed_ft_search_response(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test search handles malformed FT.SEARCH response gracefully."""
        mock_ft_list.return_value = [b"memory_index"]
        mock_ft_search.return_value = None  # Malformed response

        query_embedding = [0.1, 0.2, 0.3, 0.4]
        results = await valkey_storage.asearch(query_embedding, limit=10)

        # Verify empty results are returned (graceful handling)
        assert len(results) == 0

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_search_handles_missing_score_field(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test search handles missing score field in results."""
        record1 = MemoryRecord(
            id="record-1",
            content="Test record",
            scope="/test",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        # Create mock response without score field (dict format)
        docs = {
            f"record:{record1.id}": {
                "id": record1.id,
                "content": record1.content,
                "scope": record1.scope,
                "categories": str(record1.categories),
                "metadata": str(record1.metadata),
                "importance": str(record1.importance),
                "created_at": record1.created_at.isoformat(),
                "last_accessed": record1.last_accessed.isoformat(),
                "source": record1.source or "",
                "private": "false",
                # No score field
            }
        }

        mock_ft_list.return_value = [b"memory_index"]
        mock_ft_search.return_value = [1, docs]

        query_embedding = [0.1, 0.2, 0.3, 0.4]
        results = await valkey_storage.asearch(query_embedding, limit=10)

        # Verify record is returned with default score of 0.0
        assert len(results) == 1
        assert results[0][0].id == "record-1"
        assert results[0][1] == 0.0

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_search_filters_out_records_with_deserialization_errors(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test search filters out records that fail deserialization."""
        valid_record = MemoryRecord(
            id="valid-record",
            content="Valid record",
            scope="/test",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        # Create mock response with one valid and one invalid record (dict format)
        docs = {
            f"record:{valid_record.id}": {
                "id": valid_record.id,
                "content": valid_record.content,
                "scope": valid_record.scope,
                "categories": str(valid_record.categories),
                "metadata": str(valid_record.metadata),
                "importance": str(valid_record.importance),
                "created_at": valid_record.created_at.isoformat(),
                "last_accessed": valid_record.last_accessed.isoformat(),
                "source": valid_record.source or "",
                "private": "false",
                "score": "0.1",
            },
            "record:invalid-record": {
                "id": "invalid-record",
                # Missing content, scope, and other required fields
                "score": "0.2",
            },
        }

        mock_ft_list.return_value = [b"memory_index"]
        mock_ft_search.return_value = [2, docs]

        query_embedding = [0.1, 0.2, 0.3, 0.4]
        results = await valkey_storage.asearch(query_embedding, limit=10)

        # Verify only valid record is returned
        assert len(results) == 1
        assert results[0][0].id == "valid-record"

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_search_converts_cosine_distance_to_similarity(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test search converts Valkey Search cosine distance to similarity score."""
        record1 = MemoryRecord(
            id="record-1",
            content="Test record",
            scope="/test",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        # Create mock response with distance score (dict format)
        docs = {
            f"record:{record1.id}": {
                "id": record1.id,
                "content": record1.content,
                "scope": record1.scope,
                "categories": str(record1.categories),
                "metadata": str(record1.metadata),
                "importance": str(record1.importance),
                "created_at": record1.created_at.isoformat(),
                "last_accessed": record1.last_accessed.isoformat(),
                "source": record1.source or "",
                "private": "false",
                "score": "0.1",  # Distance = 0.1
            }
        }

        mock_ft_list.return_value = [b"memory_index"]
        mock_ft_search.return_value = [1, docs]

        query_embedding = [0.1, 0.2, 0.3, 0.4]
        results = await valkey_storage.asearch(query_embedding, limit=10)

        # Verify similarity score is correctly converted
        assert len(results) == 1
        assert results[0][0].id == "record-1"
        # Distance 0.1 -> Similarity = 1 - (0.1 / 2) = 0.95
        assert abs(results[0][1] - 0.95) < 0.01

    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    def test_search_sync_wrapper(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test that sync search wrapper calls async implementation."""
        record1 = MemoryRecord(
            id="record-1",
            content="Test record",
            scope="/test",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        mock_ft_list.return_value = [b"memory_index"]
        mock_ft_search.return_value = create_mock_ft_search_response([(record1, 0.9)])

        query_embedding = [0.1, 0.2, 0.3, 0.4]
        results = valkey_storage.search(query_embedding, limit=10)

        # Verify ft.search was called
        assert mock_ft_search.call_count >= 1

        # Verify results
        assert len(results) == 1
        assert results[0][0].id == "record-1"
        assert results[0][1] == 0.9

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_search_with_multiple_categories_uses_or_logic(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test search with multiple categories uses OR logic."""
        record1 = MemoryRecord(
            id="record-1",
            content="Record with one matching category",
            scope="/test",
            categories=["planning"],
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        mock_ft_list.return_value = [b"memory_index"]
        mock_ft_search.return_value = create_mock_ft_search_response([(record1, 0.9)])

        query_embedding = [0.1, 0.2, 0.3, 0.4]
        results = await valkey_storage.asearch(
            query_embedding,
            categories=["planning", "execution", "review"],
            limit=10
        )

        # Verify query contains OR logic for categories
        call_args = mock_ft_search.call_args
        query = call_args[0][2]
        assert "@categories:{planning|execution|review}" in query

        # Verify record with only one matching category is returned
        assert len(results) == 1
        assert results[0][0].id == "record-1"

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_search_with_multiple_metadata_filters_uses_and_logic(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test search with multiple metadata filters uses AND logic."""
        record1 = MemoryRecord(
            id="record-1",
            content="Record matching all metadata",
            scope="/test",
            metadata={"agent_id": "agent-1", "priority": "high", "status": "active"},
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        mock_ft_list.return_value = [b"memory_index"]
        mock_ft_search.return_value = create_mock_ft_search_response([(record1, 0.9)])

        query_embedding = [0.1, 0.2, 0.3, 0.4]
        results = await valkey_storage.asearch(
            query_embedding,
            metadata_filter={"agent_id": "agent-1", "priority": "high", "status": "active"},
            limit=10
        )

        # Verify query contains AND logic for metadata
        call_args = mock_ft_search.call_args
        query = call_args[0][2]
        assert "@agent_id:" in query
        assert "@priority:" in query
        assert "@status:" in query

        # Verify record matching all metadata is returned
        assert len(results) == 1
        assert results[0][0].id == "record-1"

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_search_with_zero_limit_returns_empty(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test search with limit=0 returns empty results."""
        mock_ft_list.return_value = [b"memory_index"]
        mock_ft_search.return_value = [0]

        query_embedding = [0.1, 0.2, 0.3, 0.4]
        results = await valkey_storage.asearch(query_embedding, limit=0)

        # Verify empty results
        assert len(results) == 0

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_search_with_min_score_one_filters_all(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test search with min_score=1.0 filters out all non-perfect matches."""
        record1 = MemoryRecord(
            id="record-1",
            content="High score but not perfect",
            scope="/test",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )

        mock_ft_list.return_value = [b"memory_index"]
        mock_ft_search.return_value = create_mock_ft_search_response([(record1, 0.99)])

        query_embedding = [0.1, 0.2, 0.3, 0.4]
        results = await valkey_storage.asearch(
            query_embedding,
            limit=10,
            min_score=1.0
        )

        # Verify all results are filtered out
        assert len(results) == 0

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_search_with_min_score_zero_returns_all(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Test search with min_score=0.0 returns all results."""
        record1 = MemoryRecord(
            id="record-1",
            content="High score",
            scope="/test",
            embedding=[0.1, 0.2, 0.3, 0.4],
        )
        record2 = MemoryRecord(
            id="record-2",
            content="Low score",
            scope="/test",
            embedding=[0.2, 0.3, 0.4, 0.5],
        )

        mock_ft_list.return_value = [b"memory_index"]
        mock_ft_search.return_value = create_mock_ft_search_response([
            (record1, 0.95),
            (record2, 0.05),
        ])

        query_embedding = [0.1, 0.2, 0.3, 0.4]
        results = await valkey_storage.asearch(
            query_embedding,
            limit=10,
            min_score=0.0
        )

        # Verify all results are returned
        assert len(results) == 2
        assert results[0][0].id == "record-1"
        assert results[1][0].id == "record-2"