"""ScopedStorage: the single chokepoint for the per-tenant memory isolation invariant.

A ScopedStorage wraps any StorageBackend and binds every operation it forwards
to a fixed (tenant_id, user_id) pair. Three contracts are held here, and
**nowhere else**:

1. **Stamp on write.** Every record passed to save()/update() is model_copied
   with tenant_id set to the wrapper's bound tenant. A record that arrives
   already stamped with a *different* tenant raises PermissionError -- silent
   relabel masks bugs.
2. **Inject on read.** Every read forwarded to the underlying backend carries
   the tenant_id predicate. The wrapper has no API to omit it.
3. **Verify on return.** After the backend returns rows, the wrapper re-checks
   r.tenant_id == self._tenant_id on every row. A foreign-tenant row leaking
   through raises RuntimeError -- loudly, not silently filtered.

The triple contract is defense in depth. The Protocol's required keyword arg
catches forgotten parameters at type-check time; the backend's pushed-down
predicate is the SQL/Qdrant-level filter; this wrapper is the runtime guard
that fires when either of the first two fails.

If you add a new read method, it MUST go through the tenant predicate.
If you add a new write method, it MUST go through _stamp().

See design-docs/0001-per-tenant-memory-isolation.md for the full design.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from crewai.memory.types import MemoryRecord


if TYPE_CHECKING:
    from crewai.memory.storage.backend import StorageBackend
    from crewai.memory.types import ScopeInfo


_DEFAULT_TENANT = "_default"


class ScopedStorage:
    """A tenant-bound proxy around a StorageBackend.

    Cheap to construct (two strings + one reference), so it is intended to be
    created per request rather than cached. A long-lived Memory instance can
    serve many tenants concurrently by building a fresh ScopedStorage per call.
    """

    def __init__(
        self,
        inner: StorageBackend,
        *,
        tenant_id: str,
        user_id: str | None = None,
    ) -> None:
        if not tenant_id:
            raise ValueError("ScopedStorage requires a non-empty tenant_id")
        self._inner = inner
        self._tenant_id = tenant_id
        self._user_id = user_id

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @property
    def user_id(self) -> str | None:
        return self._user_id

    # ------------------------------------------------------------------
    # Write path: stamp every record before it reaches the backend.
    # ------------------------------------------------------------------

    def _stamp(self, records: list[MemoryRecord]) -> list[MemoryRecord]:
        stamped: list[MemoryRecord] = []
        for r in records:
            if r.tenant_id and r.tenant_id != _DEFAULT_TENANT and r.tenant_id != self._tenant_id:
                # Refuse to silently relabel. A caller mixing tenants is a bug,
                # not something the storage layer should paper over.
                raise PermissionError(
                    f"ScopedStorage bound to tenant_id={self._tenant_id!r} "
                    f"refused to save record tenant_id={r.tenant_id!r}. "
                    "Cross-tenant writes through a scoped handle are not allowed."
                )
            updates: dict[str, Any] = {"tenant_id": self._tenant_id}
            if self._user_id is not None and r.user_id is None:
                updates["user_id"] = self._user_id
            stamped.append(r.model_copy(update=updates))
        return stamped

    def save(self, records: list[MemoryRecord]) -> None:
        self._inner.save(self._stamp(records))

    def update(self, record: MemoryRecord) -> None:
        self._inner.update(self._stamp([record])[0])

    async def asave(self, records: list[MemoryRecord]) -> None:
        await self._inner.asave(self._stamp(records))

    # ------------------------------------------------------------------
    # Read path: inject tenant predicate, then verify every returned row.
    # ------------------------------------------------------------------

    def _verify(self, records: list[MemoryRecord]) -> None:
        """Raise RuntimeError if any record's tenant does not match.

        Loud over silent. A broken backend filter must surface; quietly
        filtering out the leak hides the bug for the next person.
        """
        for r in records:
            if r.tenant_id != self._tenant_id:
                raise RuntimeError(
                    f"Backend returned a cross-tenant row: "
                    f"expected tenant_id={self._tenant_id!r}, got {r.tenant_id!r} "
                    f"(record id={r.id!r}). Refusing to serve."
                )

    def search(
        self,
        query_embedding: list[float],
        *,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        results = self._inner.search(
            query_embedding,
            tenant_id=self._tenant_id,
            user_id=self._user_id,
            scope_prefix=scope_prefix,
            categories=categories,
            metadata_filter=metadata_filter,
            limit=limit,
            min_score=min_score,
        )
        self._verify([r for r, _ in results])
        return results

    async def asearch(
        self,
        query_embedding: list[float],
        *,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        results = await self._inner.asearch(
            query_embedding,
            tenant_id=self._tenant_id,
            user_id=self._user_id,
            scope_prefix=scope_prefix,
            categories=categories,
            metadata_filter=metadata_filter,
            limit=limit,
            min_score=min_score,
        )
        self._verify([r for r, _ in results])
        return results

    def get_record(self, record_id: str) -> MemoryRecord | None:
        record = self._inner.get_record(
            record_id, tenant_id=self._tenant_id, user_id=self._user_id
        )
        if record is None:
            return None
        self._verify([record])
        return record

    def list_records(
        self,
        *,
        scope_prefix: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        records = self._inner.list_records(
            tenant_id=self._tenant_id,
            user_id=self._user_id,
            scope_prefix=scope_prefix,
            limit=limit,
            offset=offset,
        )
        self._verify(records)
        return records

    def delete(
        self,
        *,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        return self._inner.delete(
            tenant_id=self._tenant_id,
            user_id=self._user_id,
            scope_prefix=scope_prefix,
            categories=categories,
            record_ids=record_ids,
            older_than=older_than,
            metadata_filter=metadata_filter,
        )

    async def adelete(
        self,
        *,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        return await self._inner.adelete(
            tenant_id=self._tenant_id,
            user_id=self._user_id,
            scope_prefix=scope_prefix,
            categories=categories,
            record_ids=record_ids,
            older_than=older_than,
            metadata_filter=metadata_filter,
        )

    def reset(self, *, scope_prefix: str | None = None) -> None:
        self._inner.reset(
            tenant_id=self._tenant_id,
            user_id=self._user_id,
            scope_prefix=scope_prefix,
        )

    def get_scope_info(self, scope: str) -> ScopeInfo:
        return self._inner.get_scope_info(
            scope, tenant_id=self._tenant_id, user_id=self._user_id
        )

    def list_scopes(self, parent: str = "/") -> list[str]:
        return self._inner.list_scopes(
            parent, tenant_id=self._tenant_id, user_id=self._user_id
        )

    def list_categories(self, scope_prefix: str | None = None) -> dict[str, int]:
        return self._inner.list_categories(
            tenant_id=self._tenant_id,
            user_id=self._user_id,
            scope_prefix=scope_prefix,
        )

    def count(self, scope_prefix: str | None = None) -> int:
        return self._inner.count(
            tenant_id=self._tenant_id,
            user_id=self._user_id,
            scope_prefix=scope_prefix,
        )

    def touch_records(self, record_ids: list[str]) -> None:
        """Pass-through for non-isolation-relevant maintenance.

        touch_records is a write to last_accessed and does not need a tenant
        predicate because it operates on specific record ids the caller
        already retrieved through a scoped read. If those ids leak across
        tenants somehow, the underlying backend's per-row tenant_id is
        unchanged.
        """
        touch = getattr(self._inner, "touch_records", None)
        if touch is not None:
            touch(record_ids)

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if close is not None:
            close()
