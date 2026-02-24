"""Storage backend protocol for the unified memory system."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from crewai.memory.types import MemoryRecord, ScopeInfo


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
