"""LanceDB storage backend for the unified memory system."""

from __future__ import annotations

import contextvars
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any

from crewai_core.lock_store import lock as store_lock
import lancedb  # type: ignore[import-untyped]

from crewai.memory.types import MemoryRecord, ScopeInfo


_logger = logging.getLogger(__name__)

# Default embedding vector dimensionality (matches OpenAI text-embedding-3-small).
# Used when creating new tables and for zero-vector placeholder scans.
# Callers can override via the ``vector_dim`` constructor parameter.
DEFAULT_VECTOR_DIM = 1536

# Safety cap on the number of rows returned by a single scan query.
# Prevents unbounded memory use when scanning large tables for scope info,
# listing, or deletion. Internal only -- not user-configurable.
_SCAN_ROWS_LIMIT = 50_000

# Retry settings for LanceDB commit conflicts (optimistic concurrency).
# Under heavy write load (many concurrent saves), the table version can
# advance rapidly. 5 retries with 0.2s base delay (0.2 + 0.4 + 0.8 + 1.6 + 3.2 = 6.2s max)
# gives enough headroom to catch up with version advancement.
_MAX_RETRIES = 5
_RETRY_BASE_DELAY = 0.2  # seconds; doubles on each retry

# Tenant isolation: every row carries a tenant_id. Pre-isolation tables are
# migrated to this default tenant when first opened, so existing single-tenant
# deployments keep working unchanged.
_DEFAULT_TENANT = "_default"


def _sql_quote(value: str) -> str:
    """Escape a string literal for use inside a LanceDB WHERE clause.

    LanceDB uses SQL-like single-quoted string literals. The only escape is
    doubling a single quote. Centralizing this here keeps every tenant_id /
    user_id / scope predicate using the same escape so Bandit's S608
    rule does not fire and so a hostile tenant_id cannot break out of the
    quoted literal.
    """
    return value.replace("'", "''")


def _tenant_where(tenant_id: str, user_id: str | None = None) -> str:
    """Build the WHERE fragment that pins a query to one tenant (and optionally one user).

    Every read path in this storage assembles its WHERE clause by starting
    from this fragment and ANDing on top. There is no read path that does
    not call this function.
    """
    clause = f"tenant_id = '{_sql_quote(tenant_id)}'"
    if user_id is not None:
        clause += f" AND user_id = '{_sql_quote(user_id)}'"
    return clause


class LanceDBStorage:
    """LanceDB-backed storage for the unified memory system."""

    def __init__(
        self,
        path: str | Path | None = None,
        table_name: str = "memories",
        vector_dim: int | None = None,
        compact_every: int = 100,
    ) -> None:
        """Initialize LanceDB storage.

        Args:
            path: Directory path for the LanceDB database. Defaults to
                  ``$CREWAI_STORAGE_DIR/memory`` if the env var is set,
                  otherwise ``db_storage_path() / memory`` (platform data dir).
            table_name: Name of the table for memory records.
            vector_dim: Dimensionality of the embedding vector. When ``None``
                  (default), the dimension is auto-detected from the existing
                  table schema or from the first saved embedding.
            compact_every: Number of ``save()`` calls between automatic
                  background compactions.  Each ``save()`` creates one new
                  fragment file; compaction merges them, keeping query
                  performance consistent.  Set to 0 to disable.
        """
        if path is None:
            storage_dir = os.environ.get("CREWAI_STORAGE_DIR")
            if storage_dir:
                path = Path(storage_dir) / "memory"
            else:
                from crewai_core.paths import db_storage_path

                path = Path(db_storage_path()) / "memory"
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._table_name = table_name
        self._db = lancedb.connect(str(self._path))

        try:
            import resource

            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            if soft < 4096:
                resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 4096), hard))
        except Exception:  # noqa: S110
            pass  # Windows or already at the max hard limit — safe to ignore

        self._compact_every = compact_every
        self._save_count = 0

        self._lock_name = f"lancedb:{self._path.resolve()}"

        # Try to open an existing table and infer dimension from its schema.
        # If no table exists yet, defer creation until the first save so the
        # dimension can be auto-detected from the embedder's actual output.
        try:
            self._table: Any = self._db.open_table(self._table_name)
            self._vector_dim: int = self._infer_dim_from_table(self._table)
            with store_lock(self._lock_name):
                self._ensure_tenant_columns()
                self._ensure_scope_index()
            self._compact_if_needed()
        except Exception:
            _logger.debug(
                "Failed to open existing LanceDB table %r", table_name, exc_info=True
            )
            self._table = None
            self._vector_dim = vector_dim or 0  # 0 = not yet known

        # Explicit dim provided: create the table immediately if it doesn't exist.
        if self._table is None and vector_dim is not None:
            self._vector_dim = vector_dim
            with store_lock(self._lock_name):
                self._table = self._create_table(vector_dim)

    @staticmethod
    def _infer_dim_from_table(table: Any) -> int:
        """Read vector dimension from an existing table's schema."""
        schema = table.schema
        for field in schema:
            if field.name == "vector":
                try:
                    return int(field.type.list_size)
                except Exception:
                    break
        return DEFAULT_VECTOR_DIM

    def _do_write(self, op: str, *args: Any, **kwargs: Any) -> Any:
        """Execute a single table write with retry on commit conflicts.

        Caller must already hold ``store_lock(self._lock_name)``.
        """
        delay = _RETRY_BASE_DELAY
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return getattr(self._table, op)(*args, **kwargs)
            except OSError as e:  # noqa: PERF203
                if "Commit conflict" not in str(e) or attempt >= _MAX_RETRIES:
                    raise
                _logger.debug(
                    "LanceDB commit conflict on %s (attempt %d/%d), retrying in %.1fs",
                    op,
                    attempt + 1,
                    _MAX_RETRIES,
                    delay,
                )
                try:
                    self._table = self._db.open_table(self._table_name)
                except Exception:
                    _logger.debug("Failed to re-open table during retry", exc_info=True)
                time.sleep(delay)
                delay *= 2
        return None  # unreachable, but satisfies type checker

    def _create_table(self, vector_dim: int) -> Any:
        """Create a new table with the given vector dimension.

        Caller must already hold ``store_lock(self._lock_name)``.
        """
        placeholder = [
            {
                "id": "__schema_placeholder__",
                "content": "",
                "scope": "/",
                "categories_str": "[]",
                "metadata_str": "{}",
                "importance": 0.5,
                "created_at": datetime.utcnow().isoformat(),
                "last_accessed": datetime.utcnow().isoformat(),
                "source": "",
                "private": False,
                "tenant_id": _DEFAULT_TENANT,
                "user_id": "",
                "vector": [0.0] * vector_dim,
            }
        ]
        try:
            table = self._db.create_table(self._table_name, placeholder)
        except ValueError:
            table = self._db.open_table(self._table_name)
        else:
            table.delete("id = '__schema_placeholder__'")
        return table

    def _ensure_tenant_columns(self) -> None:
        """Add ``tenant_id`` and ``user_id`` columns to an existing table if missing.

        This is the lazy schema upgrade for tables that were created before
        per-tenant isolation. Existing rows are stamped with ``_default`` so
        every read path's ``WHERE tenant_id = ?`` predicate matches. The
        upgrade is best-effort: if LanceDB does not support add_columns at
        runtime, or if the columns already exist, the exception is swallowed
        and the storage continues. The migrate CLI command is the supported
        path for explicitly stamping existing data.

        Caller must already hold ``store_lock(self._lock_name)``.
        """
        if self._table is None:
            return
        existing = {field.name for field in self._table.schema}
        to_add: dict[str, str] = {}
        if "tenant_id" not in existing:
            to_add["tenant_id"] = f"'{_DEFAULT_TENANT}'"
        if "user_id" not in existing:
            to_add["user_id"] = "''"
        if not to_add:
            return
        try:
            self._table.add_columns(to_add)
            _logger.info(
                "Migrated LanceDB table %r: added columns %s with default tenant=%r. "
                "Run `crewai memory migrate` to assign real tenants to existing rows.",
                self._table_name,
                sorted(to_add),
                _DEFAULT_TENANT,
            )
        except Exception:
            _logger.warning(
                "Could not auto-add tenant columns to LanceDB table %r. "
                "Existing rows will read back as tenant=%r via row-level defaults. "
                "Run `crewai memory migrate` if needed.",
                self._table_name,
                _DEFAULT_TENANT,
                exc_info=True,
            )

    def _ensure_scope_index(self) -> None:
        """Create a BTREE scalar index on the ``scope`` column if not present.

        A scalar index lets LanceDB skip a full table scan when filtering by
        scope prefix, which is the hot path for ``list_records``,
        ``get_scope_info``, and ``list_scopes``.  The call is best-effort:
        if the table is empty or the index already exists the exception is
        swallowed silently.
        """
        if self._table is None:
            return
        try:
            self._table.create_scalar_index("scope", index_type="BTREE", replace=False)
        except Exception:
            _logger.debug(
                "Scope index creation skipped (may already exist)", exc_info=True
            )

    def _compact_if_needed(self) -> None:
        """Spawn a background compaction on startup.

        Called whenever an existing table is opened so that fragments
        accumulated in previous sessions are silently merged before the
        first query.  ``optimize()`` returns quickly when the table is
        already compact, so the cost is negligible in the common case.
        """
        if self._table is None or self._compact_every <= 0:
            return
        self._compact_async()

    def _compact_async(self) -> None:
        """Fire-and-forget: compact the table in a daemon background thread."""
        ctx = contextvars.copy_context()
        threading.Thread(
            target=ctx.run,
            args=(self._compact_safe,),
            daemon=True,
            name="lancedb-compact",
        ).start()

    def _compact_safe(self) -> None:
        """Run ``table.optimize()`` in a background thread, absorbing errors."""
        try:
            if self._table is not None:
                with store_lock(self._lock_name):
                    self._table.optimize()
                    self._ensure_scope_index()
        except Exception:
            _logger.debug("LanceDB background compaction failed", exc_info=True)

    def _ensure_table(self, vector_dim: int | None = None) -> Any:
        """Return the table, creating it lazily if needed.

        Args:
            vector_dim: Dimension hint (e.g. from the first embedding).
                  Falls back to the stored ``_vector_dim`` or ``DEFAULT_VECTOR_DIM``.
        """
        if self._table is not None:
            return self._table
        dim = vector_dim or self._vector_dim or DEFAULT_VECTOR_DIM
        self._vector_dim = dim
        self._table = self._create_table(dim)
        return self._table

    def _record_to_row(self, record: MemoryRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "content": record.content,
            "scope": record.scope,
            "categories_str": json.dumps(record.categories),
            "metadata_str": json.dumps(record.metadata),
            "importance": record.importance,
            "created_at": record.created_at.isoformat(),
            "last_accessed": record.last_accessed.isoformat(),
            "source": record.source or "",
            "private": record.private,
            "tenant_id": record.tenant_id or _DEFAULT_TENANT,
            "user_id": record.user_id or "",
            "vector": record.embedding
            if record.embedding
            else [0.0] * self._vector_dim,
        }

    def _row_to_record(self, row: dict[str, Any]) -> MemoryRecord:
        def _parse_dt(val: Any) -> datetime:
            if val is None:
                return datetime.utcnow()
            if isinstance(val, datetime):
                return val
            s = str(val)
            return datetime.fromisoformat(s.replace("Z", "+00:00"))

        # Backward compat: pre-isolation rows have neither column; new rows
        # have tenant_id stamped on save. Either way, every record loaded
        # through this method has a non-empty tenant_id so downstream
        # filtering and the ScopedStorage double-check never see None.
        raw_tenant = row.get("tenant_id")
        tenant_id = str(raw_tenant) if raw_tenant else _DEFAULT_TENANT
        raw_user = row.get("user_id")
        user_id = str(raw_user) if raw_user else None

        return MemoryRecord(
            id=str(row["id"]),
            content=str(row["content"]),
            scope=str(row["scope"]),
            categories=json.loads(row["categories_str"])
            if row.get("categories_str")
            else [],
            metadata=json.loads(row["metadata_str"]) if row.get("metadata_str") else {},
            importance=float(row.get("importance", 0.5)),
            created_at=_parse_dt(row.get("created_at")),
            last_accessed=_parse_dt(row.get("last_accessed")),
            embedding=row.get("vector"),
            source=row.get("source") or None,
            private=bool(row.get("private", False)),
            tenant_id=tenant_id,
            user_id=user_id,
        )

    def save(self, records: list[MemoryRecord]) -> None:
        if not records:
            return
        # Auto-detect dimension from the first real embedding.
        dim = None
        for r in records:
            if r.embedding and len(r.embedding) > 0:
                dim = len(r.embedding)
                break
        is_new_table = self._table is None
        with store_lock(self._lock_name):
            self._ensure_table(vector_dim=dim)
            rows = [self._record_to_row(rec) for rec in records]
            for row in rows:
                if row["vector"] is None or len(row["vector"]) != self._vector_dim:
                    row["vector"] = [0.0] * self._vector_dim
            self._do_write("add", rows)
            if is_new_table:
                self._ensure_scope_index()
        # Auto-compact every N saves so fragment files don't pile up.
        self._save_count += 1
        if self._compact_every > 0 and self._save_count % self._compact_every == 0:
            self._compact_async()

    def update(self, record: MemoryRecord) -> None:
        """Update a record by ID. Preserves created_at, updates last_accessed."""
        with store_lock(self._lock_name):
            self._ensure_table()
            safe_id = str(record.id).replace("'", "''")
            self._do_write("delete", f"id = '{safe_id}'")
            row = self._record_to_row(record)
            if row["vector"] is None or len(row["vector"]) != self._vector_dim:
                row["vector"] = [0.0] * self._vector_dim
            self._do_write("add", [row])

    def touch_records(self, record_ids: list[str]) -> None:
        """Update last_accessed to now for the given record IDs.

        Uses a single batch ``table.update()`` call instead of N
        delete-and-re-add cycles, which is both faster and avoids
        unnecessary write amplification.

        Args:
            record_ids: IDs of records to touch.
        """
        if not record_ids or self._table is None:
            return
        with store_lock(self._lock_name):
            now = datetime.utcnow().isoformat()
            safe_ids = [str(rid).replace("'", "''") for rid in record_ids]
            ids_expr = ", ".join(f"'{rid}'" for rid in safe_ids)
            self._do_write(
                "update",
                where=f"id IN ({ids_expr})",
                values={"last_accessed": now},
            )

    def get_record(
        self, record_id: str, *, tenant_id: str, user_id: str | None = None
    ) -> MemoryRecord | None:
        """Return a single record by ID, or None if not found in the tenant.

        A record found by id but owned by a different tenant is treated as
        not-found, which is what the isolation invariant requires.
        """
        if self._table is None:
            return None
        safe_id = _sql_quote(str(record_id))
        where = f"id = '{safe_id}' AND {_tenant_where(tenant_id, user_id)}"
        rows = self._table.search().where(where).limit(1).to_list()
        if not rows:
            return None
        return self._row_to_record(rows[0])

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
        if self._table is None:
            return []
        query = self._table.search(query_embedding)
        # Tenant predicate is unconditional and pushed down so foreign-tenant
        # rows never enter the ANN candidate pool.
        where = _tenant_where(tenant_id, user_id)
        if scope_prefix is not None and scope_prefix.strip("/"):
            prefix = scope_prefix.rstrip("/")
            like_val = _sql_quote(prefix) + "%"
            where += f" AND scope LIKE '{like_val}'"
        query = query.where(where)
        results = query.limit(
            limit * 3 if (categories or metadata_filter) else limit
        ).to_list()
        out: list[tuple[MemoryRecord, float]] = []
        for row in results:
            record = self._row_to_record(row)
            if categories and not any(c in record.categories for c in categories):
                continue
            if metadata_filter and not all(
                record.metadata.get(k) == v for k, v in metadata_filter.items()
            ):
                continue
            distance = row.get("_distance", 0.0)
            score = 1.0 / (1.0 + float(distance)) if distance is not None else 1.0
            if score >= min_score:
                out.append((record, score))
            if len(out) >= limit:
                break
        return out[:limit]

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
        if self._table is None:
            return 0
        tenant_clause = _tenant_where(tenant_id, user_id)
        with store_lock(self._lock_name):
            # Fast path: pure record_ids delete with no other predicates.
            # If any of older_than / categories / metadata_filter is also
            # specified, fall through to the scan branch so those predicates
            # are honored AND intersected with record_ids.
            if record_ids and not (categories or metadata_filter or older_than):
                before = int(self._table.count_rows())
                ids_expr = ", ".join(f"'{_sql_quote(rid)}'" for rid in record_ids)
                self._do_write(
                    "delete", f"({tenant_clause}) AND id IN ({ids_expr})"
                )
                return before - int(self._table.count_rows())
            if categories or metadata_filter or (record_ids and older_than):
                rows = self._scan_rows(
                    scope_prefix, tenant_id=tenant_id, user_id=user_id
                )
                # When record_ids is provided alongside other predicates, the
                # delete is the INTERSECTION of all of them: a row must match
                # the predicates AND be in record_ids.
                allowed_ids: set[str] | None = (
                    set(record_ids) if record_ids else None
                )
                to_delete: list[str] = []
                for row in rows:
                    record = self._row_to_record(row)
                    if allowed_ids is not None and record.id not in allowed_ids:
                        continue
                    if categories and not any(
                        c in record.categories for c in categories
                    ):
                        continue
                    if metadata_filter and not all(
                        record.metadata.get(k) == v for k, v in metadata_filter.items()
                    ):
                        continue
                    if older_than and record.created_at >= older_than:
                        continue
                    to_delete.append(record.id)
                if not to_delete:
                    return 0
                before = int(self._table.count_rows())
                ids_expr = ", ".join(f"'{_sql_quote(rid)}'" for rid in to_delete)
                self._do_write(
                    "delete", f"({tenant_clause}) AND id IN ({ids_expr})"
                )
                return before - int(self._table.count_rows())
            conditions = [tenant_clause]
            if scope_prefix is not None and scope_prefix.strip("/"):
                prefix = scope_prefix.rstrip("/")
                if not prefix.startswith("/"):
                    prefix = "/" + prefix
                conditions.append(
                    f"(scope LIKE '{_sql_quote(prefix)}%' OR scope = '/')"
                )
            if older_than is not None:
                conditions.append(
                    f"created_at < '{_sql_quote(older_than.isoformat())}'"
                )
            where_expr = " AND ".join(conditions)
            before = int(self._table.count_rows())
            self._do_write("delete", where_expr)
            return before - int(self._table.count_rows())

    def _scan_rows(
        self,
        scope_prefix: str | None = None,
        limit: int = _SCAN_ROWS_LIMIT,
        columns: list[str] | None = None,
        *,
        tenant_id: str,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Scan rows scoped to a tenant, optionally filtered by scope prefix.

        Uses a full table scan (no vector query) so the limit is applied after
        the tenant + scope filter, not to ANN candidates before filtering.

        Args:
            scope_prefix: Optional scope path prefix to filter by.
            limit: Maximum number of rows to return (applied after filtering).
            columns: Optional list of column names to fetch. Pass only the
                columns you need for metadata operations to avoid reading the
                heavy ``vector`` column unnecessarily.
            tenant_id: Isolation key (required, keyword-only).
            user_id: Optional sub-tenant filter.
        """
        if self._table is None:
            return []
        q = self._table.search()
        where = _tenant_where(tenant_id, user_id)
        if scope_prefix is not None and scope_prefix.strip("/"):
            where += f" AND scope LIKE '{_sql_quote(scope_prefix.rstrip('/'))}%'"
        q = q.where(where)
        if columns is not None:
            q = q.select(columns)
        result: list[dict[str, Any]] = q.limit(limit).to_list()
        return result

    def list_records(
        self,
        *,
        tenant_id: str,
        user_id: str | None = None,
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
        rows = self._scan_rows(
            scope_prefix,
            limit=limit + offset,
            tenant_id=tenant_id,
            user_id=user_id,
        )
        records = [self._row_to_record(r) for r in rows]
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records[offset : offset + limit]

    def get_scope_info(
        self, scope: str, *, tenant_id: str, user_id: str | None = None
    ) -> ScopeInfo:
        scope = scope.rstrip("/") or "/"
        prefix = scope if scope != "/" else ""
        if prefix and not prefix.startswith("/"):
            prefix = "/" + prefix
        rows = self._scan_rows(
            prefix or None,
            columns=["scope", "categories_str", "created_at"],
            tenant_id=tenant_id,
            user_id=user_id,
        )
        if not rows:
            return ScopeInfo(
                path=scope or "/",
                record_count=0,
                categories=[],
                oldest_record=None,
                newest_record=None,
                child_scopes=[],
            )
        categories_set: set[str] = set()
        oldest: datetime | None = None
        newest: datetime | None = None
        child_prefix = (prefix + "/") if prefix else "/"
        children: set[str] = set()
        for row in rows:
            sc = str(row.get("scope", ""))
            if child_prefix and sc.startswith(child_prefix):
                rest = sc[len(child_prefix) :]
                first_component = rest.split("/", 1)[0]
                if first_component:
                    children.add(child_prefix + first_component)
            try:
                cat_str = row.get("categories_str") or "[]"
                categories_set.update(json.loads(cat_str))
            except Exception:  # noqa: S110
                pass
            created = row.get("created_at")
            if created:
                dt = (
                    datetime.fromisoformat(str(created).replace("Z", "+00:00"))
                    if isinstance(created, str)
                    else created
                )
                if isinstance(dt, datetime):
                    if oldest is None or dt < oldest:
                        oldest = dt
                    if newest is None or dt > newest:
                        newest = dt
        return ScopeInfo(
            path=scope or "/",
            record_count=len(rows),
            categories=sorted(categories_set),
            oldest_record=oldest,
            newest_record=newest,
            child_scopes=sorted(children),
        )

    def list_scopes(
        self,
        parent: str = "/",
        *,
        tenant_id: str,
        user_id: str | None = None,
    ) -> list[str]:
        parent = parent.rstrip("/") or ""
        prefix = (parent + "/") if parent else "/"
        rows = self._scan_rows(
            prefix if prefix != "/" else None,
            columns=["scope"],
            tenant_id=tenant_id,
            user_id=user_id,
        )
        children: set[str] = set()
        for row in rows:
            sc = str(row.get("scope", ""))
            if sc.startswith(prefix) and sc != (prefix.rstrip("/") or "/"):
                rest = sc[len(prefix) :]
                first_component = rest.split("/", 1)[0]
                if first_component:
                    children.add(prefix + first_component)
        return sorted(children)

    def list_categories(
        self,
        *,
        tenant_id: str,
        user_id: str | None = None,
        scope_prefix: str | None = None,
    ) -> dict[str, int]:
        rows = self._scan_rows(
            scope_prefix,
            columns=["categories_str"],
            tenant_id=tenant_id,
            user_id=user_id,
        )
        counts: dict[str, int] = {}
        for row in rows:
            cat_str = row.get("categories_str") or "[]"
            try:
                parsed = json.loads(cat_str)
            except Exception:  # noqa: S112
                continue
            for c in parsed:
                counts[c] = counts.get(c, 0) + 1
        return counts

    def count(
        self,
        *,
        tenant_id: str,
        user_id: str | None = None,
        scope_prefix: str | None = None,
    ) -> int:
        if self._table is None:
            return 0
        # Even an unfiltered count is scoped to a tenant; "count rows across
        # all tenants" is intentionally not exposed.
        info = self.get_scope_info(
            scope_prefix or "/", tenant_id=tenant_id, user_id=user_id
        )
        return info.record_count

    def reset(
        self,
        *,
        tenant_id: str,
        user_id: str | None = None,
        scope_prefix: str | None = None,
    ) -> None:
        """Reset (delete all) memories for this tenant.

        There is no "drop the whole table" path; resetting one tenant never
        wipes another tenant's data. To remove the entire on-disk table,
        delete the storage directory directly.
        """
        if self._table is None:
            return
        tenant_clause = _tenant_where(tenant_id, user_id)
        with store_lock(self._lock_name):
            if scope_prefix is None or scope_prefix.strip("/") == "":
                self._do_write("delete", tenant_clause)
                return
            prefix = scope_prefix.rstrip("/")
            if prefix:
                self._do_write(
                    "delete",
                    f"({tenant_clause}) AND scope >= '{_sql_quote(prefix)}' "
                    f"AND scope < '{_sql_quote(prefix)}/\uffff'",
                )

    def optimize(self) -> None:
        """Compact the table synchronously and refresh the scope index.

        Under normal usage this is called automatically in the background
        (every ``compact_every`` saves and on startup when the table is
        fragmented).  Call this explicitly only when you need the compaction
        to be complete before the next operation — for example immediately
        after a large bulk import, before a latency-sensitive recall.
        It is a no-op if the table does not exist.
        """
        if self._table is None:
            return
        with store_lock(self._lock_name):
            self._table.optimize()
            self._ensure_scope_index()

    async def asave(self, records: list[MemoryRecord]) -> None:
        self.save(records)

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
        return self.search(
            query_embedding,
            tenant_id=tenant_id,
            user_id=user_id,
            scope_prefix=scope_prefix,
            categories=categories,
            metadata_filter=metadata_filter,
            limit=limit,
            min_score=min_score,
        )

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
        return self.delete(
            tenant_id=tenant_id,
            user_id=user_id,
            scope_prefix=scope_prefix,
            categories=categories,
            record_ids=record_ids,
            older_than=older_than,
            metadata_filter=metadata_filter,
        )
