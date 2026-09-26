"""Storage backend protocol for the unified memory system."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from typing import Any, Optional, Protocol, runtime_checkable

from crewai.memory.types import MemoryRecord, ScopeInfo


class EmbeddingDimensionMismatchError(ValueError):
    """Raised when an embedding's dimensionality doesn't match the existing store.

    The most common cause is upgrading CrewAI across the default-embedder
    change (text-embedding-3-small, 1536 dims → text-embedding-3-large,
    3072 dims) while keeping a local memory store created before the upgrade.

    Deliberately not a ``RuntimeError``: background-save plumbing treats
    ``RuntimeError`` as interpreter/executor shutdown and silently drops the
    save, which would swallow this actionable migration error.
    """

    def __init__(self, stored_dim: int, new_dim: int) -> None:
        self.stored_dim = stored_dim
        self.new_dim = new_dim
        super().__init__(
            f"Embedding dimension mismatch: this memory store contains "
            f"{stored_dim}-dimensional vectors, but the current embedder produced "
            f"a {new_dim}-dimensional vector.\n\n"
            "This usually means the store was created with a different embedding "
            "model. CrewAI's default embedder changed from "
            "text-embedding-3-small (1536 dims) to text-embedding-3-large "
            "(3072 dims), so memory stores created before the upgrade are "
            "incompatible with the new default.\n\n"
            "To fix, do one of the following:\n"
            "  - Reset local memory so it is rebuilt with the new embedder:\n"
            "      crewai reset-memories --memory   (or crew.reset_memories())\n"
            "  - Keep existing memories by pinning the previous embedder:\n"
            '      embedder={"provider": "openai", '
            '"config": {"model": "text-embedding-3-small"}}'
        )


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for pluggable memory storage backends."""

    def save(self, records: list[MemoryRecord]) -> None:
        """Save memory records to storage.

        Args:
            records: List of memory records to persist.
        """
        ...

    def search(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        """Search for memories by vector similarity with optional filters.

        Args:
            query_embedding: Embedding vector for the query.
            scope_prefix: Optional scope path prefix to filter results.
            categories: Optional list of categories to filter by.
            metadata_filter: Optional metadata key-value filter.
            limit: Maximum number of results to return.
            min_score: Minimum similarity score threshold.

        Returns:
            List of (MemoryRecord, score) tuples ordered by relevance.
        """
        ...

    def delete(
        self,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        """Delete memories matching the given criteria.

        Args:
            scope_prefix: Optional scope path prefix.
            categories: Optional list of categories.
            record_ids: Optional list of record IDs to delete.
            older_than: Optional cutoff datetime (delete older records).
            metadata_filter: Optional metadata key-value filter.

        Returns:
            Number of records deleted.
        """
        ...

    def update(self, record: MemoryRecord) -> None:
        """Update an existing record. Replaces the record with the same ID."""
        ...

    def get_record(self, record_id: str) -> MemoryRecord | None:
        """Return a single record by ID, or None if not found.

        Args:
            record_id: The unique ID of the record.

        Returns:
            The MemoryRecord, or None if no record with that ID exists.
        """
        ...

    def list_records(
        self,
        scope_prefix: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        """List records in a scope, newest first.

        Args:
            scope_prefix: Optional scope path prefix to filter by.
            limit: Maximum number of records to return.
            offset: Number of records to skip (for pagination).

        Returns:
            List of MemoryRecord, ordered by created_at descending.
        """
        ...

    def get_scope_info(self, scope: str) -> ScopeInfo:
        """Get information about a scope.

        Args:
            scope: The scope path.

        Returns:
            ScopeInfo with record count, categories, date range, child scopes.
        """
        ...

    def list_scopes(self, parent: str = "/") -> list[str]:
        """List immediate child scopes under a parent path.

        Args:
            parent: Parent scope path (default root).

        Returns:
            List of immediate child scope paths.
        """
        ...

    def list_categories(self, scope_prefix: str | None = None) -> dict[str, int]:
        """List categories and their counts within a scope.

        Args:
            scope_prefix: Optional scope to limit to (None = global).

        Returns:
            Mapping of category name to record count.
        """
        ...

    def count(self, scope_prefix: str | None = None) -> int:
        """Count records in scope (and subscopes).

        Args:
            scope_prefix: Optional scope path (None = all).

        Returns:
            Number of records.
        """
        ...

    def reset(self, scope_prefix: str | None = None) -> None:
        """Reset (delete all) memories in scope.

        Args:
            scope_prefix: Optional scope path (None = reset all).
        """
        ...

    async def asave(self, records: list[MemoryRecord]) -> None:
        """Save memory records asynchronously."""
        ...

    async def asearch(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        """Search for memories asynchronously."""
        ...

    async def adelete(
        self,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        """Delete memories asynchronously."""
        ...

class RedisStorageBackend:
    """Distributed Redis storage backend implementing the StorageBackend protocol."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
    ):
        try:
            import redis
        except ImportError as err:
            raise ImportError(
                "Please install the redis package to use RedisStorageBackend: `pip install redis`"
            ) from err

        self.client = redis.Redis(
            host=host,
            port=port,
            db=db,
            password=password,
            decode_responses=True,
            socket_connect_timeout=5.0,
            socket_timeout=5.0,
        )

    def _get_key(self, record_id: str) -> str:
        return f"crewai:memory:{record_id}"

    def save(self, records: list[MemoryRecord]) -> None:
        with self.client.pipeline() as pipe:
            for record in records:
                if hasattr(record, "model_dump_json"):
                    record_json = record.model_dump_json()
                else:
                    record_data = record.__dict__.copy()
                    record_json = json.dumps(record_data, default=str)

                pipe.set(self._get_key(record.id), record_json)
            pipe.execute()

    def get_record(self, record_id: str) -> MemoryRecord | None:
        data = self.client.get(self._get_key(record_id))
        if not data:
            return None
        raw_data = json.loads(data)
        for dt_field in ("created_at", "last_accessed"):
            if dt_field in raw_data and isinstance(raw_data[dt_field], str):
                raw_data[dt_field] = datetime.fromisoformat(raw_data[dt_field])
        return MemoryRecord(**raw_data)

    def update(self, record: MemoryRecord) -> None:
        self.save([record])

    def delete(
        self,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        if (
            categories is not None
            or older_than is not None
            or metadata_filter is not None
            or (scope_prefix is not None and not record_ids)
        ):
            raise NotImplementedError(
                "RedisStorageBackend.delete currently supports only record_ids filtering."
            )

        if record_ids:
            keys_to_delete = [self._get_key(rid) for rid in record_ids]
            return (
                self.client.delete(*keys_to_delete) if keys_to_delete else 0
            )
        return 0

    def search(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        raise NotImplementedError(
            "RedisStorageBackend.search is not implemented yet."
        )

    def list_records(
        self,
        scope_prefix: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        raise NotImplementedError(
            "RedisStorageBackend.list_records is not implemented yet."
        )

    def get_scope_info(self, scope: str) -> ScopeInfo:
        raise NotImplementedError(
            "RedisStorageBackend.get_scope_info is not implemented yet."
        )

    def list_scopes(self, parent: str = "/") -> list[str]:
        raise NotImplementedError(
            "RedisStorageBackend.list_scopes is not implemented yet."
        )

    def list_categories(
        self, scope_prefix: str | None = None
    ) -> dict[str, int]:
        raise NotImplementedError(
            "RedisStorageBackend.list_categories is not implemented yet."
        )

    def count(self, scope_prefix: str | None = None) -> int:
        if scope_prefix is not None:
            raise NotImplementedError("Scoped count is not implemented yet.")
        count = 0
        for _ in self.client.scan_iter("crewai:memory:*"):
            count += 1
        return count

    def reset(self, scope_prefix: str | None = None) -> None:
        if scope_prefix is not None:
            raise NotImplementedError("Scoped reset is not implemented yet.")
        batch: list[str] = []
        for key in self.client.scan_iter("crewai:memory:*"):
            batch.append(key)
            if len(batch) == 500:
                self.client.delete(*batch)
                batch.clear()
        if batch:
            self.client.delete(*batch)

    # --- Async Implementations with thread offloading ---
    async def asave(self, records: list[MemoryRecord]) -> None:
        await asyncio.to_thread(self.save, records)

    async def asearch(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        return await asyncio.to_thread(
            self.search,
            query_embedding,
            scope_prefix,
            categories,
            metadata_filter,
            limit,
            min_score,
        )

    async def adelete(
        self,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        return await asyncio.to_thread(
            self.delete,
            scope_prefix,
            categories,
            record_ids,
            older_than,
            metadata_filter,
        )
