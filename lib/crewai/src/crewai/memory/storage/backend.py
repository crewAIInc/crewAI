"""Storage backend protocol for the unified memory system.

Per-tenant isolation contract
-----------------------------
Every read method on this Protocol takes ``tenant_id`` as a **required
keyword-only** argument. The required-without-default form is deliberate:
mypy --strict turns any forgotten caller into a CI failure, which is the
static guarantee behind the isolation invariant described in
``design-docs/0001-per-tenant-memory-isolation.md``.

``save`` and ``update`` do not take ``tenant_id`` -- the tenant lives on
the record itself, and a separate parameter would invite a "record says A,
param says B" mismatch.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from crewai.memory.types import MemoryRecord, ScopeInfo


@runtime_checkable
class StorageBackend(Protocol):
    """Protocol for pluggable memory storage backends."""

    def save(self, records: list[MemoryRecord]) -> None:
        """Save memory records to storage.

        The tenant_id is read from each record, not from a parameter.

        Args:
            records: List of memory records to persist.
        """
        ...

    def search(
        self,
        query_embedding: list[float],
        *,
        tenant_id: str,
        user_id: str | None = None,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        """Search for memories by vector similarity with optional filters.

        Args:
            query_embedding: Embedding vector for the query.
            tenant_id: Isolation key. Backends MUST push this into the
                vector query so foreign-tenant rows never enter the
                candidate pool.
            user_id: Optional sub-tenant identity for further filtering.
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
        *,
        tenant_id: str,
        user_id: str | None = None,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        """Delete memories matching the given criteria (scoped to tenant).

        Args:
            tenant_id: Isolation key. Only rows owned by this tenant are eligible.
            user_id: Optional sub-tenant filter.
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
        """Update an existing record. Replaces the record with the same ID.

        The tenant_id is read from the record itself.
        """
        ...

    def get_record(
        self, record_id: str, *, tenant_id: str, user_id: str | None = None
    ) -> MemoryRecord | None:
        """Return a single record by ID, or None if not found in the tenant.

        Args:
            record_id: The unique ID of the record.
            tenant_id: Isolation key. A record found by ID but owned by a
                different tenant is treated as not-found.
            user_id: Optional sub-tenant filter.

        Returns:
            The MemoryRecord, or None if no record with that ID exists for the tenant.
        """
        ...

    def list_records(
        self,
        *,
        tenant_id: str,
        user_id: str | None = None,
        scope_prefix: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        """List records in a scope (scoped to tenant), newest first.

        Args:
            tenant_id: Isolation key.
            user_id: Optional sub-tenant filter.
            scope_prefix: Optional scope path prefix to filter by.
            limit: Maximum number of records to return.
            offset: Number of records to skip (for pagination).

        Returns:
            List of MemoryRecord, ordered by created_at descending.
        """
        ...

    def get_scope_info(
        self, scope: str, *, tenant_id: str, user_id: str | None = None
    ) -> ScopeInfo:
        """Get information about a scope (scoped to tenant).

        Args:
            scope: The scope path.
            tenant_id: Isolation key.
            user_id: Optional sub-tenant filter.

        Returns:
            ScopeInfo with record count, categories, date range, child scopes.
        """
        ...

    def list_scopes(
        self, parent: str = "/", *, tenant_id: str, user_id: str | None = None
    ) -> list[str]:
        """List immediate child scopes under a parent path (scoped to tenant).

        Args:
            parent: Parent scope path (default root).
            tenant_id: Isolation key.
            user_id: Optional sub-tenant filter.

        Returns:
            List of immediate child scope paths.
        """
        ...

    def list_categories(
        self,
        *,
        tenant_id: str,
        user_id: str | None = None,
        scope_prefix: str | None = None,
    ) -> dict[str, int]:
        """List categories and their counts within a scope (scoped to tenant).

        Args:
            tenant_id: Isolation key.
            user_id: Optional sub-tenant filter.
            scope_prefix: Optional scope to limit to (None = whole tenant).

        Returns:
            Mapping of category name to record count.
        """
        ...

    def count(
        self,
        *,
        tenant_id: str,
        user_id: str | None = None,
        scope_prefix: str | None = None,
    ) -> int:
        """Count records in scope (scoped to tenant).

        Args:
            tenant_id: Isolation key.
            user_id: Optional sub-tenant filter.
            scope_prefix: Optional scope path (None = whole tenant).

        Returns:
            Number of records.
        """
        ...

    def reset(
        self,
        *,
        tenant_id: str,
        user_id: str | None = None,
        scope_prefix: str | None = None,
    ) -> None:
        """Reset (delete all) memories within a tenant.

        There is no "reset everything across all tenants" path. An operator
        who needs that calls reset for each tenant deliberately.

        Args:
            tenant_id: Isolation key. Only this tenant's rows are wiped.
            user_id: Optional sub-tenant filter.
            scope_prefix: Optional scope path (None = whole tenant).
        """
        ...

    async def asave(self, records: list[MemoryRecord]) -> None:
        """Save memory records asynchronously."""
        ...

    async def asearch(
        self,
        query_embedding: list[float],
        *,
        tenant_id: str,
        user_id: str | None = None,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        """Search for memories asynchronously (scoped to tenant)."""
        ...

    async def adelete(
        self,
        *,
        tenant_id: str,
        user_id: str | None = None,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        """Delete memories asynchronously (scoped to tenant)."""
        ...
