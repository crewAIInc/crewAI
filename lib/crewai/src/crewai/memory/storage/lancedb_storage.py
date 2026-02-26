"""LanceDB storage backend for the unified memory system."""

from __future__ import annotations

from datetime import datetime
import json
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, ClassVar

import lancedb

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


class LanceDBStorage:
    """LanceDB-backed storage for the unified memory system."""

    # Class-level registry: maps resolved database path -> shared write lock.
    # When multiple Memory instances (e.g. agent + crew) independently create
    # LanceDBStorage pointing at the same directory, they share one lock so
    # their writes don't conflict.
    # Uses RLock (reentrant) so callers can hold the lock for a batch of
    # operations while the individual methods re-acquire it without deadlocking.
    _path_locks: ClassVar[dict[str, threading.RLock]] = {}
    _path_locks_guard: ClassVar[threading.Lock] = threading.Lock()

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
                from crewai.utilities.paths import db_storage_path

                path = Path(db_storage_path()) / "memory"
        self._path = Path(path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._table_name = table_name
        self._db = lancedb.connect(str(self._path))

        # On macOS and Linux the default per-process open-file limit is 256.
        # A LanceDB table stores one file per fragment (one fragment per save()
        # call by default).  With hundreds of fragments, a single full-table
        # scan opens all of them simultaneously, exhausting the limit.
        # Raise it proactively so scans on large tables never hit OS error 24.
        try:
            import resource
            soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            if soft < 4096:
                resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 4096), hard))
        except Exception:
            pass  # Windows or already at the max hard limit — safe to ignore

        self._compact_every = compact_every
        self._save_count = 0

        # Get or create a shared write lock for this database path.
        resolved = str(self._path.resolve())
        with LanceDBStorage._path_locks_guard:
            if resolved not in LanceDBStorage._path_locks:
                LanceDBStorage._path_locks[resolved] = threading.RLock()
            self._write_lock = LanceDBStorage._path_locks[resolved]

        # Try to open an existing table and infer dimension from its schema.
        # If no table exists yet, defer creation until the first save so the
        # dimension can be auto-detected from the embedder's actual output.
        try:
            self._table: lancedb.table.Table | None = self._db.open_table(self._table_name)
            self._vector_dim: int = self._infer_dim_from_table(self._table)
            # Best-effort: create the scope index if it doesn't exist yet.
            self._ensure_scope_index()
            # Compact in the background if the table has accumulated many
            # fragments from previous runs (each save() creates one).
            self._compact_if_needed()
        except Exception:
            self._table = None
            self._vector_dim = vector_dim or 0  # 0 = not yet known

        # Explicit dim provided: create the table immediately if it doesn't exist.
        if self._table is None and vector_dim is not None:
            self._vector_dim = vector_dim
            self._table = self._create_table(vector_dim)

    @property
    def write_lock(self) -> threading.RLock:
        """The shared reentrant write lock for this database path.

        Callers can acquire this to hold the lock across multiple storage
        operations (e.g. delete + update + save as one atomic batch).
        Individual methods also acquire it internally, but since it's
        reentrant (RLock), the same thread won't deadlock.
        """
        return self._write_lock

    @staticmethod
    def _infer_dim_from_table(table: lancedb.table.Table) -> int:
        """Read vector dimension from an existing table's schema."""
        schema = table.schema
        for field in schema:
            if field.name == "vector":
                try:
                    return field.type.list_size
                except Exception:
                    break
        return DEFAULT_VECTOR_DIM

    def _retry_write(self, op: str, *args: Any, **kwargs: Any) -> Any:
        """Execute a table operation with retry on LanceDB commit conflicts.

        Args:
            op: Method name on the table object (e.g. "add", "delete").
            *args, **kwargs: Passed to the table method.

        LanceDB uses optimistic concurrency: if two transactions overlap,
        the second to commit fails with an ``OSError`` containing
        "Commit conflict". This helper retries with exponential backoff,
        refreshing the table reference before each retry so the retried
        call uses the latest committed version (not a stale reference).
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
                    op, attempt + 1, _MAX_RETRIES, delay,
                )
                # Refresh table to pick up the latest version before retrying.
                # The next getattr(self._table, op) will use the fresh table.
                try:
                    self._table = self._db.open_table(self._table_name)
                except Exception:  # noqa: S110
                    pass  # table refresh is best-effort
                time.sleep(delay)
                delay *= 2
        return None  # unreachable, but satisfies type checker

    def _create_table(self, vector_dim: int) -> lancedb.table.Table:
        """Create a new table with the given vector dimension."""
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
                "vector": [0.0] * vector_dim,
            }
        ]
        table = self._db.create_table(self._table_name, placeholder)
        table.delete("id = '__schema_placeholder__'")
        return table

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
            pass  # index already exists, table empty, or unsupported version

    # ------------------------------------------------------------------
    # Automatic background compaction
    # ------------------------------------------------------------------

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
        threading.Thread(
            target=self._compact_safe,
            daemon=True,
            name="lancedb-compact",
        ).start()

    def _compact_safe(self) -> None:
        """Run ``table.optimize()`` in a background thread, absorbing errors."""
        try:
            if self._table is not None:
                self._table.optimize()
                # Refresh the scope index so new fragments are covered.
                self._ensure_scope_index()
        except Exception:
            _logger.debug("LanceDB background compaction failed", exc_info=True)

    def _ensure_table(self, vector_dim: int | None = None) -> lancedb.table.Table:
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
            "vector": record.embedding if record.embedding else [0.0] * self._vector_dim,
        }

    def _row_to_record(self, row: dict[str, Any]) -> MemoryRecord:
        def _parse_dt(val: Any) -> datetime:
            if val is None:
                return datetime.utcnow()
            if isinstance(val, datetime):
                return val
            s = str(val)
            return datetime.fromisoformat(s.replace("Z", "+00:00"))

        return MemoryRecord(
            id=str(row["id"]),
            content=str(row["content"]),
            scope=str(row["scope"]),
            categories=json.loads(row["categories_str"]) if row.get("categories_str") else [],
            metadata=json.loads(row["metadata_str"]) if row.get("metadata_str") else {},
            importance=float(row.get("importance", 0.5)),
            created_at=_parse_dt(row.get("created_at")),
            last_accessed=_parse_dt(row.get("last_accessed")),
            embedding=row.get("vector"),
            source=row.get("source") or None,
            private=bool(row.get("private", False)),
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
        with self._write_lock:
            self._ensure_table(vector_dim=dim)
            rows = [self._record_to_row(r) for r in records]
            for r in rows:
                if r["vector"] is None or len(r["vector"]) != self._vector_dim:
                    r["vector"] = [0.0] * self._vector_dim
            self._retry_write("add", rows)
        # Create the scope index on the first save so it covers the initial dataset.
        if is_new_table:
            self._ensure_scope_index()
        # Auto-compact every N saves so fragment files don't pile up.
        self._save_count += 1
        if self._compact_every > 0 and self._save_count % self._compact_every == 0:
            self._compact_async()

    def update(self, record: MemoryRecord) -> None:
        """Update a record by ID. Preserves created_at, updates last_accessed."""
        with self._write_lock:
            self._ensure_table()
            safe_id = str(record.id).replace("'", "''")
            self._retry_write("delete", f"id = '{safe_id}'")
            row = self._record_to_row(record)
            if row["vector"] is None or len(row["vector"]) != self._vector_dim:
                row["vector"] = [0.0] * self._vector_dim
            self._retry_write("add", [row])

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
        with self._write_lock:
            now = datetime.utcnow().isoformat()
            safe_ids = [str(rid).replace("'", "''") for rid in record_ids]
            ids_expr = ", ".join(f"'{rid}'" for rid in safe_ids)
            self._retry_write(
                "update",
                where=f"id IN ({ids_expr})",
                values={"last_accessed": now},
            )

    def get_record(self, record_id: str) -> MemoryRecord | None:
        """Return a single record by ID, or None if not found."""
        if self._table is None:
            return None
        safe_id = str(record_id).replace("'", "''")
        rows = self._table.search().where(f"id = '{safe_id}'").limit(1).to_list()
        if not rows:
            return None
        return self._row_to_record(rows[0])

    def search(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        if self._table is None:
            return []
        query = self._table.search(query_embedding)
        if scope_prefix is not None and scope_prefix.strip("/"):
            prefix = scope_prefix.rstrip("/")
            like_val = prefix + "%"
            query = query.where(f"scope LIKE '{like_val}'")
        results = query.limit(limit * 3 if (categories or metadata_filter) else limit).to_list()
        out: list[tuple[MemoryRecord, float]] = []
        for row in results:
            record = self._row_to_record(row)
            if categories and not any(c in record.categories for c in categories):
                continue
            if metadata_filter and not all(record.metadata.get(k) == v for k, v in metadata_filter.items()):
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
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        if self._table is None:
            return 0
        with self._write_lock:
            if record_ids and not (categories or metadata_filter):
                before = self._table.count_rows()
                ids_expr = ", ".join(f"'{rid}'" for rid in record_ids)
                self._retry_write("delete", f"id IN ({ids_expr})")
                return before - self._table.count_rows()
            if categories or metadata_filter:
                rows = self._scan_rows(scope_prefix)
                to_delete: list[str] = []
                for row in rows:
                    record = self._row_to_record(row)
                    if categories and not any(c in record.categories for c in categories):
                        continue
                    if metadata_filter and not all(record.metadata.get(k) == v for k, v in metadata_filter.items()):
                        continue
                    if older_than and record.created_at >= older_than:
                        continue
                    to_delete.append(record.id)
                if not to_delete:
                    return 0
                before = self._table.count_rows()
                ids_expr = ", ".join(f"'{rid}'" for rid in to_delete)
                self._retry_write("delete", f"id IN ({ids_expr})")
                return before - self._table.count_rows()
            conditions = []
            if scope_prefix is not None and scope_prefix.strip("/"):
                prefix = scope_prefix.rstrip("/")
                if not prefix.startswith("/"):
                    prefix = "/" + prefix
                conditions.append(f"scope LIKE '{prefix}%' OR scope = '/'")
            if older_than is not None:
                conditions.append(f"created_at < '{older_than.isoformat()}'")
            if not conditions:
                before = self._table.count_rows()
                self._retry_write("delete", "id != ''")
                return before - self._table.count_rows()
            where_expr = " AND ".join(conditions)
            before = self._table.count_rows()
            self._retry_write("delete", where_expr)
            return before - self._table.count_rows()

    def _scan_rows(
        self,
        scope_prefix: str | None = None,
        limit: int = _SCAN_ROWS_LIMIT,
        columns: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Scan rows optionally filtered by scope prefix.

        Uses a full table scan (no vector query) so the limit is applied after
        the scope filter, not to ANN candidates before filtering.

        Args:
            scope_prefix: Optional scope path prefix to filter by.
            limit: Maximum number of rows to return (applied after filtering).
            columns: Optional list of column names to fetch.  Pass only the
                columns you need for metadata operations to avoid reading the
                heavy ``vector`` column unnecessarily.
        """
        if self._table is None:
            return []
        q = self._table.search()
        if scope_prefix is not None and scope_prefix.strip("/"):
            q = q.where(f"scope LIKE '{scope_prefix.rstrip('/')}%'")
        if columns is not None:
            q = q.select(columns)
        return q.limit(limit).to_list()

    def list_records(
        self, scope_prefix: str | None = None, limit: int = 200, offset: int = 0
    ) -> list[MemoryRecord]:
        """List records in a scope, newest first.

        Args:
            scope_prefix: Optional scope path prefix to filter by.
            limit: Maximum number of records to return.
            offset: Number of records to skip (for pagination).

        Returns:
            List of MemoryRecord, ordered by created_at descending.
        """
        rows = self._scan_rows(scope_prefix, limit=limit + offset)
        records = [self._row_to_record(r) for r in rows]
        records.sort(key=lambda r: r.created_at, reverse=True)
        return records[offset : offset + limit]

    def get_scope_info(self, scope: str) -> ScopeInfo:
        scope = scope.rstrip("/") or "/"
        prefix = scope if scope != "/" else ""
        if prefix and not prefix.startswith("/"):
            prefix = "/" + prefix
        rows = self._scan_rows(
            prefix or None,
            columns=["scope", "categories_str", "created_at"],
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
                rest = sc[len(child_prefix):]
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
                dt = datetime.fromisoformat(str(created).replace("Z", "+00:00")) if isinstance(created, str) else created
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

    def list_scopes(self, parent: str = "/") -> list[str]:
        parent = parent.rstrip("/") or ""
        prefix = (parent + "/") if parent else "/"
        rows = self._scan_rows(prefix if prefix != "/" else None, columns=["scope"])
        children: set[str] = set()
        for row in rows:
            sc = str(row.get("scope", ""))
            if sc.startswith(prefix) and sc != (prefix.rstrip("/") or "/"):
                rest = sc[len(prefix):]
                first_component = rest.split("/", 1)[0]
                if first_component:
                    children.add(prefix + first_component)
        return sorted(children)

    def list_categories(self, scope_prefix: str | None = None) -> dict[str, int]:
        rows = self._scan_rows(scope_prefix, columns=["categories_str"])
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

    def count(self, scope_prefix: str | None = None) -> int:
        if self._table is None:
            return 0
        if scope_prefix is None or scope_prefix.strip("/") == "":
            return self._table.count_rows()
        info = self.get_scope_info(scope_prefix)
        return info.record_count

    def reset(self, scope_prefix: str | None = None) -> None:
        if scope_prefix is None or scope_prefix.strip("/") == "":
            if self._table is not None:
                self._db.drop_table(self._table_name)
            self._table = None
            # Dimension is preserved; table will be recreated on next save.
            return
        if self._table is None:
            return
        prefix = scope_prefix.rstrip("/")
        if prefix:
            self._table.delete(f"scope >= '{prefix}' AND scope < '{prefix}/\uFFFF'")

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
        self._table.optimize()
        self._ensure_scope_index()

    async def asave(self, records: list[MemoryRecord]) -> None:
        self.save(records)

    async def asearch(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        return self.search(
            query_embedding,
            scope_prefix=scope_prefix,
            categories=categories,
            metadata_filter=metadata_filter,
            limit=limit,
            min_score=min_score,
        )

    async def adelete(
        self,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        return self.delete(
            scope_prefix=scope_prefix,
            categories=categories,
            record_ids=record_ids,
            older_than=older_than,
            metadata_filter=metadata_filter,
        )
