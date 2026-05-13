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
        """Metadata filter is applied as a post-filter (issue #5794).

        ``metadata`` is not part of the ``memory_index`` FT schema, so the
        FT.SEARCH query string must not reference metadata keys. The filter is
        applied in Python after FT.SEARCH returns.
        """
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

        # Metadata predicates must NOT be injected into the FT.SEARCH query
        # (their schema fields do not exist on memory_index).
        call_args = mock_ft_search.call_args
        query = call_args[0][2]
        assert "@agent_id" not in query
        assert "@priority" not in query
        # KNN clause is still present, with the overfetch limit applied so the
        # post-filter has enough candidates to return ``limit`` hits.
        assert "[KNN 100 @embedding $BLOB AS score]" in query

        # Post-filter retains the record whose metadata matches every filter key.
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
        """Combined filters: scope + categories pushed down, metadata post-filtered."""
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

        # Scope and categories are valid index fields and stay in the query.
        call_args = mock_ft_search.call_args
        query = call_args[0][2]
        assert "@scope:{/agent*}" in query
        assert "@categories:{planning}" in query
        # Metadata predicates must NOT be in the FT.SEARCH query.
        assert "@agent_id" not in query
        # Overfetch limit is applied when metadata_filter is supplied.
        assert "[KNN 100 @embedding $BLOB AS score]" in query

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
        """Numeric metadata filters work via post-filter (string-coerced compare)."""
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

        # Metadata predicates must NOT be in the FT.SEARCH query; the
        # post-filter is responsible for matching numeric values.
        call_args = mock_ft_search.call_args
        query = call_args[0][2]
        assert "@count" not in query
        assert "@score" not in query

        # Post-filter accepts numeric values (both int and float) and returns
        # the matching record.
        assert len(results) == 1
        assert results[0][0].metadata["count"] == 42
        assert results[0][0].metadata["score"] == 3.14

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
        """Multiple metadata filters use AND logic in the post-filter."""
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

        # Metadata predicates must NOT be in the FT.SEARCH query — they are
        # not part of the index schema.
        call_args = mock_ft_search.call_args
        query = call_args[0][2]
        assert "@agent_id" not in query
        assert "@priority" not in query
        assert "@status" not in query

        # AND logic: every filter key must match for the record to be kept.
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


class TestValkeyStorageMetadataPostFilter:
    """Regression tests for issue #5794.

    The ``memory_index`` FT schema only contains fixed fields
    (``embedding``, ``scope``, ``categories``, ``created_at``, ``importance``).
    Arbitrary metadata keys are NOT indexed, so ``metadata_filter`` must be
    applied as a Python post-filter over FT.SEARCH results — never injected
    into the FT.SEARCH query string.
    """

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_metadata_filter_not_pushed_into_ft_query(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """No ``@<key>`` clause for any metadata key is emitted to FT.SEARCH."""
        record = MemoryRecord(
            id="record-1",
            content="A record",
            scope="/test",
            metadata={"agent_id": "agent-1", "priority": "high"},
            embedding=[0.1, 0.2, 0.3, 0.4],
        )
        mock_ft_list.return_value = [b"memory_index"]
        mock_ft_search.return_value = create_mock_ft_search_response([(record, 0.9)])

        await valkey_storage.asearch(
            [0.1, 0.2, 0.3, 0.4],
            metadata_filter={"agent_id": "agent-1", "priority": "high"},
            limit=5,
        )

        query = mock_ft_search.call_args[0][2]
        # The bug was: metadata keys were emitted as @<key>:{value} clauses
        # that reference non-existent FT schema fields.
        assert "@agent_id" not in query
        assert "@priority" not in query
        # The KNN clause uses the overfetched limit when metadata_filter is set.
        assert "[KNN 50 @embedding $BLOB AS score]" in query

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_metadata_filter_excludes_records_with_missing_key(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Records whose metadata is missing a required key are dropped."""
        matching = MemoryRecord(
            id="match-1",
            content="Has agent_id",
            scope="/test",
            metadata={"agent_id": "agent-1"},
            embedding=[0.1, 0.2, 0.3, 0.4],
        )
        missing_key = MemoryRecord(
            id="missing-1",
            content="Missing agent_id entirely",
            scope="/test",
            metadata={"unrelated": "value"},
            embedding=[0.2, 0.3, 0.4, 0.5],
        )
        empty_metadata = MemoryRecord(
            id="empty-1",
            content="No metadata",
            scope="/test",
            metadata={},
            embedding=[0.3, 0.4, 0.5, 0.6],
        )
        mock_ft_list.return_value = [b"memory_index"]
        mock_ft_search.return_value = create_mock_ft_search_response([
            (matching, 0.95),
            (missing_key, 0.85),
            (empty_metadata, 0.75),
        ])

        results = await valkey_storage.asearch(
            [0.1, 0.2, 0.3, 0.4],
            metadata_filter={"agent_id": "agent-1"},
            limit=10,
        )

        # Only the record whose metadata has agent_id=agent-1 survives.
        assert [rec.id for rec, _ in results] == ["match-1"]

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_metadata_filter_excludes_records_with_mismatched_value(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Records whose metadata value differs from the filter are dropped."""
        agent_one = MemoryRecord(
            id="agent-1-rec",
            content="agent-1",
            scope="/test",
            metadata={"agent_id": "agent-1"},
            embedding=[0.1, 0.2, 0.3, 0.4],
        )
        agent_two = MemoryRecord(
            id="agent-2-rec",
            content="agent-2",
            scope="/test",
            metadata={"agent_id": "agent-2"},
            embedding=[0.2, 0.3, 0.4, 0.5],
        )
        mock_ft_list.return_value = [b"memory_index"]
        mock_ft_search.return_value = create_mock_ft_search_response([
            (agent_one, 0.9),
            (agent_two, 0.8),
        ])

        results = await valkey_storage.asearch(
            [0.1, 0.2, 0.3, 0.4],
            metadata_filter={"agent_id": "agent-1"},
            limit=10,
        )

        assert [rec.id for rec, _ in results] == ["agent-1-rec"]

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_metadata_filter_and_logic_requires_every_key(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Multi-key filters require every key/value pair to match (AND)."""
        full_match = MemoryRecord(
            id="full",
            content="Matches both",
            scope="/test",
            metadata={"agent_id": "agent-1", "priority": "high"},
            embedding=[0.1, 0.2, 0.3, 0.4],
        )
        partial = MemoryRecord(
            id="partial",
            content="Only agent_id matches",
            scope="/test",
            metadata={"agent_id": "agent-1", "priority": "low"},
            embedding=[0.2, 0.3, 0.4, 0.5],
        )
        mock_ft_list.return_value = [b"memory_index"]
        mock_ft_search.return_value = create_mock_ft_search_response([
            (full_match, 0.95),
            (partial, 0.93),
        ])

        results = await valkey_storage.asearch(
            [0.1, 0.2, 0.3, 0.4],
            metadata_filter={"agent_id": "agent-1", "priority": "high"},
            limit=10,
        )

        assert [rec.id for rec, _ in results] == ["full"]

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_metadata_filter_matches_numeric_values_via_string_compare(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Numeric metadata values match when stringified equal to the filter."""
        record = MemoryRecord(
            id="numeric",
            content="Numeric metadata",
            scope="/test",
            metadata={"count": 42, "ratio": 3.14, "active": True},
            embedding=[0.1, 0.2, 0.3, 0.4],
        )
        other = MemoryRecord(
            id="numeric-other",
            content="Different count",
            scope="/test",
            metadata={"count": 41, "ratio": 3.14, "active": True},
            embedding=[0.2, 0.3, 0.4, 0.5],
        )
        mock_ft_list.return_value = [b"memory_index"]
        mock_ft_search.return_value = create_mock_ft_search_response([
            (record, 0.9),
            (other, 0.85),
        ])

        results = await valkey_storage.asearch(
            [0.1, 0.2, 0.3, 0.4],
            metadata_filter={"count": 42, "ratio": 3.14, "active": True},
            limit=10,
        )

        assert [rec.id for rec, _ in results] == ["numeric"]

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_empty_metadata_filter_dict_skips_post_filter(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """An empty ``{}`` filter behaves like ``None`` (no overfetch, no filtering)."""
        record1 = MemoryRecord(
            id="r1",
            content="r1",
            scope="/test",
            metadata={"k": "v"},
            embedding=[0.1, 0.2, 0.3, 0.4],
        )
        record2 = MemoryRecord(
            id="r2",
            content="r2",
            scope="/test",
            metadata={},
            embedding=[0.2, 0.3, 0.4, 0.5],
        )
        mock_ft_list.return_value = [b"memory_index"]
        mock_ft_search.return_value = create_mock_ft_search_response([
            (record1, 0.9),
            (record2, 0.8),
        ])

        results = await valkey_storage.asearch(
            [0.1, 0.2, 0.3, 0.4],
            metadata_filter={},
            limit=10,
        )

        query = mock_ft_search.call_args[0][2]
        # No overfetch — KNN limit stays at the caller-requested limit.
        assert "[KNN 10 @embedding $BLOB AS score]" in query
        # All records returned (no filtering performed).
        assert [rec.id for rec, _ in results] == ["r1", "r2"]

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_metadata_filter_overfetches_to_preserve_caller_limit(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """KNN is overfetched so post-filter can still return ``limit`` hits.

        Without overfetch, KNN would return at most ``limit`` candidates and
        the post-filter could drop every one, leaving the caller with fewer
        than the requested ``limit`` results. The overfetch ensures the
        post-filter has enough candidates to satisfy the caller in the common
        case where ~10% of candidates match the metadata predicate.
        """
        matching = [
            MemoryRecord(
                id=f"match-{i}",
                content=f"match {i}",
                scope="/test",
                metadata={"tier": "gold"},
                embedding=[0.1 * (i + 1), 0.2, 0.3, 0.4],
            )
            for i in range(5)
        ]
        non_matching = [
            MemoryRecord(
                id=f"skip-{i}",
                content=f"skip {i}",
                scope="/test",
                metadata={"tier": "silver"},
                embedding=[0.1 * (i + 1), 0.3, 0.4, 0.5],
            )
            for i in range(15)
        ]

        # FT.SEARCH returns 20 candidates (mix of gold/silver) sorted by score.
        # If the KNN had been capped at limit=5, the silver records mixed in
        # would have starved out the gold matches and ``results`` would have
        # been shorter than 5.
        combined = []
        score = 0.99
        for rec in matching + non_matching:
            combined.append((rec, score))
            score -= 0.01

        mock_ft_list.return_value = [b"memory_index"]
        mock_ft_search.return_value = create_mock_ft_search_response(combined)

        results = await valkey_storage.asearch(
            [0.1, 0.2, 0.3, 0.4],
            metadata_filter={"tier": "gold"},
            limit=5,
        )

        query = mock_ft_search.call_args[0][2]
        # KNN limit is overfetched (limit * OVERFETCH = 50) to give the
        # post-filter enough candidates.
        assert "[KNN 50 @embedding $BLOB AS score]" in query

        # All five matching records are returned, in score-descending order.
        assert [rec.id for rec, _ in results] == [
            "match-0", "match-1", "match-2", "match-3", "match-4"
        ]

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_metadata_filter_truncates_to_caller_limit(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """When post-filter retains more than ``limit`` records, truncate."""
        records = [
            MemoryRecord(
                id=f"rec-{i}",
                content=f"rec {i}",
                scope="/test",
                metadata={"tier": "gold"},
                embedding=[0.1 * (i + 1), 0.2, 0.3, 0.4],
            )
            for i in range(8)
        ]
        scored = [(rec, 0.9 - 0.05 * i) for i, rec in enumerate(records)]

        mock_ft_list.return_value = [b"memory_index"]
        mock_ft_search.return_value = create_mock_ft_search_response(scored)

        results = await valkey_storage.asearch(
            [0.1, 0.2, 0.3, 0.4],
            metadata_filter={"tier": "gold"},
            limit=3,
        )

        # Caller asked for 3, so only the top 3 (by score) are returned even
        # though 8 candidates matched the metadata filter.
        assert len(results) == 3
        assert [rec.id for rec, _ in results] == ["rec-0", "rec-1", "rec-2"]

    @pytest.mark.asyncio
    @patch("crewai.memory.storage.valkey_storage.ft.search")
    @patch("crewai.memory.storage.valkey_storage.ft.list")
    async def test_metadata_filter_does_not_affect_scope_or_categories_pushdown(
        self, mock_ft_list: AsyncMock, mock_ft_search: AsyncMock,
        valkey_storage: ValkeyStorage, mock_glide_client: AsyncMock
    ) -> None:
        """Scope/categories predicates are still pushed into FT.SEARCH.

        They are valid index fields and benefit from server-side filtering.
        Only ``metadata_filter`` is moved to the Python side.
        """
        record = MemoryRecord(
            id="rec",
            content="rec",
            scope="/agent/task",
            categories=["planning"],
            metadata={"agent_id": "agent-1"},
            embedding=[0.1, 0.2, 0.3, 0.4],
        )
        mock_ft_list.return_value = [b"memory_index"]
        mock_ft_search.return_value = create_mock_ft_search_response([(record, 0.9)])

        await valkey_storage.asearch(
            [0.1, 0.2, 0.3, 0.4],
            scope_prefix="/agent",
            categories=["planning"],
            metadata_filter={"agent_id": "agent-1"},
            limit=5,
        )

        query = mock_ft_search.call_args[0][2]
        assert "@scope:{/agent*}" in query
        assert "@categories:{planning}" in query
        # Metadata predicates remain on the Python side.
        assert "@agent_id" not in query


class TestValkeyStorageMatchesMetadataFilter:
    """Unit tests for the ``_matches_metadata_filter`` static helper."""

    def test_returns_true_when_all_keys_match(self) -> None:
        assert ValkeyStorage._matches_metadata_filter(
            {"a": "1", "b": "2", "c": "3"},
            {"a": "1", "b": "2"},
        )

    def test_returns_false_when_key_missing(self) -> None:
        assert not ValkeyStorage._matches_metadata_filter(
            {"a": "1"},
            {"a": "1", "b": "2"},
        )

    def test_returns_false_when_value_mismatches(self) -> None:
        assert not ValkeyStorage._matches_metadata_filter(
            {"a": "1", "b": "wrong"},
            {"a": "1", "b": "2"},
        )

    def test_coerces_to_string_for_comparison(self) -> None:
        # Filter pre-stringified by caller still matches typed record value.
        assert ValkeyStorage._matches_metadata_filter(
            {"count": 42, "ratio": 3.14, "flag": True},
            {"count": "42", "ratio": "3.14", "flag": "True"},
        )
        # And vice versa: typed filter matches typed record value.
        assert ValkeyStorage._matches_metadata_filter(
            {"count": 42},
            {"count": 42},
        )

    def test_empty_filter_always_matches(self) -> None:
        assert ValkeyStorage._matches_metadata_filter({}, {})
        assert ValkeyStorage._matches_metadata_filter({"a": "1"}, {})