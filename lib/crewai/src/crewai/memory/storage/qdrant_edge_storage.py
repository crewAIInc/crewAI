"""Qdrant Edge storage backend for the unified memory system.

Uses a write-local/sync-central pattern for safe multi-process access.
Each worker process writes to its own local shard (keyed by PID). Reads
fan out to both local and central shards, merging results. On close,
local records are flushed to the shared central shard.
"""

from __future__ import annotations

import asyncio
import atexit
from datetime import datetime, timezone
import logging
import os
from pathlib import Path
import shutil
from typing import Any, Final
import uuid

from qdrant_edge import (
    CountRequest,
    Distance,
    EdgeConfig,
    EdgeShard,
    EdgeVectorParams,
    FacetRequest,
    FieldCondition,
    Filter,
    MatchValue,
    PayloadSchemaType,
    Point,
    Query,
    QueryRequest,
    ScrollRequest,
    UpdateOperation,
)

from crewai.memory.types import MemoryRecord, ScopeInfo


_logger = logging.getLogger(__name__)

VECTOR_NAME: Final[str] = "memory"

DEFAULT_VECTOR_DIM: Final[int] = 1536

_SCROLL_BATCH: Final[int] = 256


def _uuid_to_point_id(uuid_str: str) -> int:
    """Convert a UUID string to a stable Qdrant point ID.

    Falls back to hashing for non-UUID strings.
    """
    try:
        return uuid.UUID(uuid_str).int % (2**63 - 1)
    except ValueError:
        return int.from_bytes(uuid_str.encode()[:8].ljust(8, b"\x00"), "big") % (
            2**63 - 1
        )


def _build_scope_ancestors(scope: str) -> list[str]:
    """Build the list of all ancestor scopes for prefix filtering.

    For scope ``/crew/sales/agent``, returns
    ``["/", "/crew", "/crew/sales", "/crew/sales/agent"]``.
    """
    parts = scope.strip("/").split("/")
    ancestors: list[str] = ["/"]
    current = ""
    for part in parts:
        if part:
            current = f"{current}/{part}"
            ancestors.append(current)
    return ancestors


class QdrantEdgeStorage:
    """Qdrant Edge storage backend with write-local/sync-central pattern.

    Each worker process gets its own local shard for writes.
    Reads merge results from both local and central shards. On close,
    local records are flushed to the shared central shard.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        vector_dim: int | None = None,
    ) -> None:
        """Initialize Qdrant Edge storage.

        Args:
            path: Base directory for shard storage. Defaults to
                ``$CREWAI_STORAGE_DIR/memory/qdrant-edge`` or the
                platform data directory.
            vector_dim: Embedding vector dimensionality. Auto-detected
                from the first saved embedding when ``None``.
        """
        if path is None:
            storage_dir = os.environ.get("CREWAI_STORAGE_DIR")
            if storage_dir:
                path = Path(storage_dir) / "memory" / "qdrant-edge"
            else:
                from crewai.utilities.paths import db_storage_path

                path = Path(db_storage_path()) / "memory" / "qdrant-edge"

        self._base_path = Path(path)
        self._central_path = self._base_path / "central"
        self._local_path = self._base_path / f"worker-{os.getpid()}"
        self._vector_dim = vector_dim or 0
        self._config: EdgeConfig | None = None
        self._local_has_data = self._local_path.exists()
        self._closed = False
        self._indexes_created = False

        if self._vector_dim > 0:
            self._config = self._build_config(self._vector_dim)

        if self._config is None and self._central_path.exists():
            try:
                shard = EdgeShard.load(str(self._central_path))
                if shard.count(CountRequest()) > 0:
                    pts, _ = shard.scroll(
                        ScrollRequest(limit=1, with_payload=False, with_vector=True)
                    )
                    if pts and pts[0].vector:
                        vec = pts[0].vector
                        if isinstance(vec, dict) and VECTOR_NAME in vec:
                            vec_data = vec[VECTOR_NAME]
                            dim = len(vec_data) if isinstance(vec_data, list) else 0
                            if dim > 0:
                                self._vector_dim = dim
                                self._config = self._build_config(dim)
                shard.close()
            except Exception:
                _logger.debug("Failed to detect dim from central shard", exc_info=True)

        self._cleanup_orphaned_shards()
        atexit.register(self.close)

    @staticmethod
    def _build_config(dim: int) -> EdgeConfig:
        """Build an EdgeConfig for the given vector dimensionality."""
        return EdgeConfig(
            vectors={VECTOR_NAME: EdgeVectorParams(size=dim, distance=Distance.Cosine)},
        )

    def _open_shard(self, path: Path) -> EdgeShard:
        """Open an existing shard or create a new one at *path*."""
        path.mkdir(parents=True, exist_ok=True)
        try:
            return EdgeShard.load(str(path))
        except Exception:
            if self._config is None:
                raise
            return EdgeShard.create(str(path), self._config)

    def _ensure_indexes(self, shard: EdgeShard) -> None:
        """Create payload indexes for efficient filtering."""
        if self._indexes_created:
            return
        try:
            shard.update(
                UpdateOperation.create_field_index(
                    "scope_ancestors", PayloadSchemaType.Keyword
                )
            )
            shard.update(
                UpdateOperation.create_field_index(
                    "categories", PayloadSchemaType.Keyword
                )
            )
            shard.update(
                UpdateOperation.create_field_index(
                    "record_id", PayloadSchemaType.Keyword
                )
            )
            self._indexes_created = True
        except Exception:
            _logger.debug("Index creation failed (may already exist)", exc_info=True)

    def _record_to_point(self, record: MemoryRecord) -> Point:
        """Convert a MemoryRecord to a Qdrant Point."""
        return Point(
            id=_uuid_to_point_id(record.id),
            vector={
                VECTOR_NAME: record.embedding
                if record.embedding
                else [0.0] * self._vector_dim,
            },
            payload={
                "record_id": record.id,
                "content": record.content,
                "scope": record.scope,
                "scope_ancestors": _build_scope_ancestors(record.scope),
                "categories": record.categories,
                "metadata": record.metadata,
                "importance": record.importance,
                "created_at": record.created_at.isoformat(),
                "last_accessed": record.last_accessed.isoformat(),
                "source": record.source or "",
                "private": record.private,
            },
        )

    @staticmethod
    def _payload_to_record(
        payload: dict[str, Any],
        vector: dict[str, list[float]] | None = None,
    ) -> MemoryRecord:
        """Reconstruct a MemoryRecord from a Qdrant payload."""

        def _parse_dt(val: Any) -> datetime:
            if val is None:
                return datetime.now(timezone.utc).replace(tzinfo=None)
            if isinstance(val, datetime):
                return val
            return datetime.fromisoformat(str(val).replace("Z", "+00:00"))

        return MemoryRecord(
            id=str(payload["record_id"]),
            content=str(payload["content"]),
            scope=str(payload["scope"]),
            categories=payload.get("categories", []),
            metadata=payload.get("metadata", {}),
            importance=float(payload.get("importance", 0.5)),
            created_at=_parse_dt(payload.get("created_at")),
            last_accessed=_parse_dt(payload.get("last_accessed")),
            embedding=vector.get(VECTOR_NAME) if vector else None,
            source=payload.get("source") or None,
            private=bool(payload.get("private", False)),
        )

    @staticmethod
    def _build_scope_filter(scope_prefix: str | None) -> Filter | None:
        """Build a Qdrant Filter for scope prefix matching."""
        if scope_prefix is None or not scope_prefix.strip("/"):
            return None
        prefix = scope_prefix.rstrip("/")
        if not prefix.startswith("/"):
            prefix = "/" + prefix
        return Filter(
            must=[FieldCondition(key="scope_ancestors", match=MatchValue(value=prefix))]
        )

    @staticmethod
    def _scroll_all(
        shard: EdgeShard,
        filt: Filter | None = None,
        with_vector: bool = False,
    ) -> list[Any]:
        """Scroll all points matching a filter from a shard."""
        all_points: list[Any] = []
        offset = None
        while True:
            batch, next_offset = shard.scroll(
                ScrollRequest(
                    limit=_SCROLL_BATCH,
                    offset=offset,
                    with_payload=True,
                    with_vector=with_vector,
                    filter=filt,
                )
            )
            all_points.extend(batch)
            if next_offset is None or not batch:
                break
            offset = next_offset
        return all_points

    def save(self, records: list[MemoryRecord]) -> None:
        """Save records to the worker-local shard."""
        if not records:
            return

        if self._vector_dim == 0:
            for r in records:
                if r.embedding and len(r.embedding) > 0:
                    self._vector_dim = len(r.embedding)
                    break
        if self._config is None and self._vector_dim > 0:
            self._config = self._build_config(self._vector_dim)
        if self._config is None:
            self._config = self._build_config(DEFAULT_VECTOR_DIM)
            self._vector_dim = DEFAULT_VECTOR_DIM

        points = [self._record_to_point(r) for r in records]
        local = self._open_shard(self._local_path)
        try:
            self._ensure_indexes(local)
            local.update(UpdateOperation.upsert_points(points))
            local.flush()
            self._local_has_data = True
        finally:
            local.close()

    def search(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        """Search both central and local shards, merge results."""
        filt = self._build_scope_filter(scope_prefix)
        fetch_limit = limit * 3 if (categories or metadata_filter) else limit
        all_scored: list[tuple[dict[str, Any], float, bool]] = []

        for shard_path in (self._central_path, self._local_path):
            if not shard_path.exists():
                continue
            is_local = shard_path == self._local_path
            try:
                shard = EdgeShard.load(str(shard_path))
                results = shard.query(
                    QueryRequest(
                        query=Query.Nearest(list(query_embedding), using=VECTOR_NAME),
                        filter=filt,
                        limit=fetch_limit,
                        with_payload=True,
                        with_vector=False,
                    )
                )
                all_scored.extend(
                    (sp.payload or {}, float(sp.score), is_local) for sp in results
                )
                shard.close()
            except Exception:
                _logger.debug("Search failed on %s", shard_path, exc_info=True)

        seen: dict[str, tuple[dict[str, Any], float]] = {}
        local_ids: set[str] = set()
        for payload, score, is_local in all_scored:
            rid = payload["record_id"]
            if is_local:
                local_ids.add(rid)
                seen[rid] = (payload, score)
            elif rid not in local_ids:
                if rid not in seen or score > seen[rid][1]:
                    seen[rid] = (payload, score)

        ranked = sorted(seen.values(), key=lambda x: x[1], reverse=True)
        out: list[tuple[MemoryRecord, float]] = []
        for payload, score in ranked:
            record = self._payload_to_record(payload)
            if categories and not any(c in record.categories for c in categories):
                continue
            if metadata_filter and not all(
                record.metadata.get(k) == v for k, v in metadata_filter.items()
            ):
                continue
            if score < min_score:
                continue
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
        """Delete matching records from central shard."""
        total_deleted = 0
        for shard_path in (self._central_path, self._local_path):
            if not shard_path.exists():
                continue
            try:
                total_deleted += self._delete_from_shard_path(
                    shard_path,
                    scope_prefix,
                    categories,
                    record_ids,
                    older_than,
                    metadata_filter,
                )
            except Exception:
                _logger.debug("Delete failed on %s", shard_path, exc_info=True)
        return total_deleted

    def _delete_from_shard_path(
        self,
        shard_path: Path,
        scope_prefix: str | None,
        categories: list[str] | None,
        record_ids: list[str] | None,
        older_than: datetime | None,
        metadata_filter: dict[str, Any] | None,
    ) -> int:
        """Delete matching records from a shard at the given path."""
        shard = EdgeShard.load(str(shard_path))
        try:
            deleted = self._delete_from_shard(
                shard,
                scope_prefix,
                categories,
                record_ids,
                older_than,
                metadata_filter,
            )
            shard.flush()
        finally:
            shard.close()
        return deleted

    def _delete_from_shard(
        self,
        shard: EdgeShard,
        scope_prefix: str | None,
        categories: list[str] | None,
        record_ids: list[str] | None,
        older_than: datetime | None,
        metadata_filter: dict[str, Any] | None,
    ) -> int:
        """Delete matching records from a single shard, returning count deleted."""
        before = shard.count(CountRequest())

        if record_ids and not (categories or metadata_filter or older_than):
            point_ids: list[int | uuid.UUID | str] = [
                _uuid_to_point_id(rid) for rid in record_ids
            ]
            shard.update(UpdateOperation.delete_points(point_ids))
            return before - shard.count(CountRequest())

        if categories or metadata_filter or older_than:
            scope_filter = self._build_scope_filter(scope_prefix)
            points = self._scroll_all(shard, filt=scope_filter)
            allowed_ids: set[str] | None = set(record_ids) if record_ids else None
            to_delete: list[int | uuid.UUID | str] = []
            for pt in points:
                record = self._payload_to_record(pt.payload or {})
                if allowed_ids and record.id not in allowed_ids:
                    continue
                if categories and not any(c in record.categories for c in categories):
                    continue
                if metadata_filter and not all(
                    record.metadata.get(k) == v for k, v in metadata_filter.items()
                ):
                    continue
                if older_than and record.created_at >= older_than:
                    continue
                to_delete.append(pt.id)
            if to_delete:
                shard.update(UpdateOperation.delete_points(to_delete))
            return before - shard.count(CountRequest())

        scope_filter = self._build_scope_filter(scope_prefix)
        if scope_filter:
            shard.update(UpdateOperation.delete_points_by_filter(filter=scope_filter))
        else:
            points = self._scroll_all(shard)
            if points:
                all_ids: list[int | uuid.UUID | str] = [p.id for p in points]
                shard.update(UpdateOperation.delete_points(all_ids))
        return before - shard.count(CountRequest())

    def update(self, record: MemoryRecord) -> None:
        """Update a record by upserting with the same point ID."""
        if self._config is None:
            if record.embedding and len(record.embedding) > 0:
                self._vector_dim = len(record.embedding)
                self._config = self._build_config(self._vector_dim)
            else:
                self._config = self._build_config(DEFAULT_VECTOR_DIM)
                self._vector_dim = DEFAULT_VECTOR_DIM

        point = self._record_to_point(record)
        local = self._open_shard(self._local_path)
        try:
            self._ensure_indexes(local)
            local.update(UpdateOperation.upsert_points([point]))
            local.flush()
            self._local_has_data = True
        finally:
            local.close()

    def get_record(self, record_id: str) -> MemoryRecord | None:
        """Return a single record by ID, or None if not found."""
        point_id = _uuid_to_point_id(record_id)
        for shard_path in (self._local_path, self._central_path):
            if not shard_path.exists():
                continue
            try:
                shard = EdgeShard.load(str(shard_path))
                records = shard.retrieve([point_id], True, True)
                shard.close()
                if records:
                    payload = records[0].payload or {}
                    vec = records[0].vector
                    vec_dict = vec if isinstance(vec, dict) else None
                    return self._payload_to_record(payload, vec_dict)  # type: ignore[arg-type]
            except Exception:
                _logger.debug("get_record failed on %s", shard_path, exc_info=True)
        return None

    def list_records(
        self,
        scope_prefix: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        """List records in a scope, newest first."""
        filt = self._build_scope_filter(scope_prefix)
        all_records: list[MemoryRecord] = []
        seen_ids: set[str] = set()

        for shard_path in (self._local_path, self._central_path):
            if not shard_path.exists():
                continue
            try:
                shard = EdgeShard.load(str(shard_path))
                points = self._scroll_all(shard, filt=filt)
                shard.close()
                for pt in points:
                    rid = pt.payload["record_id"]
                    if rid not in seen_ids:
                        seen_ids.add(rid)
                        all_records.append(self._payload_to_record(pt.payload))
            except Exception:
                _logger.debug("list_records failed on %s", shard_path, exc_info=True)

        all_records.sort(key=lambda r: r.created_at, reverse=True)
        return all_records[offset : offset + limit]

    def get_scope_info(self, scope: str) -> ScopeInfo:
        """Get information about a scope."""
        scope = scope.rstrip("/") or "/"
        prefix = scope if scope != "/" else None
        filt = self._build_scope_filter(prefix)

        all_points: list[Any] = []
        for shard_path in (self._central_path, self._local_path):
            if not shard_path.exists():
                continue
            try:
                shard = EdgeShard.load(str(shard_path))
                all_points.extend(self._scroll_all(shard, filt=filt))
                shard.close()
            except Exception:
                _logger.debug("get_scope_info failed on %s", shard_path, exc_info=True)

        if not all_points:
            return ScopeInfo(
                path=scope,
                record_count=0,
                categories=[],
                oldest_record=None,
                newest_record=None,
                child_scopes=[],
            )

        seen: dict[str, Any] = {}
        for pt in all_points:
            rid = pt.payload["record_id"]
            if rid not in seen:
                seen[rid] = pt

        categories_set: set[str] = set()
        oldest: datetime | None = None
        newest: datetime | None = None
        child_prefix = (scope + "/") if scope != "/" else "/"
        children: set[str] = set()

        for pt in seen.values():
            payload = pt.payload
            sc = str(payload.get("scope", ""))
            if child_prefix and sc.startswith(child_prefix):
                rest = sc[len(child_prefix) :]
                first_component = rest.split("/", 1)[0]
                if first_component:
                    children.add(child_prefix + first_component)
            for c in payload.get("categories", []):
                categories_set.add(c)
            created = payload.get("created_at")
            if created:
                dt = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
                if oldest is None or dt < oldest:
                    oldest = dt
                if newest is None or dt > newest:
                    newest = dt

        return ScopeInfo(
            path=scope,
            record_count=len(seen),
            categories=sorted(categories_set),
            oldest_record=oldest,
            newest_record=newest,
            child_scopes=sorted(children),
        )

    def list_scopes(self, parent: str = "/") -> list[str]:
        """List immediate child scopes under a parent path."""
        parent = parent.rstrip("/") or ""
        prefix = (parent + "/") if parent else "/"

        all_scopes: set[str] = set()
        filt = self._build_scope_filter(prefix if prefix != "/" else None)
        for shard_path in (self._central_path, self._local_path):
            if not shard_path.exists():
                continue
            try:
                shard = EdgeShard.load(str(shard_path))
                points = self._scroll_all(shard, filt=filt)
                shard.close()
                for pt in points:
                    sc = str(pt.payload.get("scope", ""))
                    if sc.startswith(prefix) and sc != (prefix.rstrip("/") or "/"):
                        rest = sc[len(prefix) :]
                        first_component = rest.split("/", 1)[0]
                        if first_component:
                            all_scopes.add(prefix + first_component)
            except Exception:
                _logger.debug("list_scopes failed on %s", shard_path, exc_info=True)
        return sorted(all_scopes)

    def list_categories(self, scope_prefix: str | None = None) -> dict[str, int]:
        """List categories and their counts within a scope."""
        if not self._local_has_data and self._central_path.exists():
            try:
                shard = EdgeShard.load(str(self._central_path))
                try:
                    shard.update(
                        UpdateOperation.create_field_index(
                            "categories", PayloadSchemaType.Keyword
                        )
                    )
                except Exception:  # noqa: S110
                    pass
                filt = self._build_scope_filter(scope_prefix)
                facet_result = shard.facet(
                    FacetRequest(key="categories", limit=1000, filter=filt)
                )
                shard.close()
                return {str(hit.value): hit.count for hit in facet_result.hits}
            except Exception:
                _logger.debug("list_categories failed on central", exc_info=True)

        counts: dict[str, int] = {}
        for record in self.list_records(scope_prefix=scope_prefix, limit=50_000):
            for c in record.categories:
                counts[c] = counts.get(c, 0) + 1
        return counts

    def count(self, scope_prefix: str | None = None) -> int:
        """Count records in scope (and subscopes)."""
        filt = self._build_scope_filter(scope_prefix)
        if not self._local_has_data:
            if self._central_path.exists():
                try:
                    shard = EdgeShard.load(str(self._central_path))
                    result = shard.count(CountRequest(filter=filt))
                    shard.close()
                    return result
                except Exception:
                    _logger.debug("count failed on central", exc_info=True)
            return 0
        seen_ids: set[str] = set()
        for shard_path in (self._local_path, self._central_path):
            if not shard_path.exists():
                continue
            try:
                shard = EdgeShard.load(str(shard_path))
                for pt in self._scroll_all(shard, filt=filt):
                    seen_ids.add(pt.payload["record_id"])
                shard.close()
            except Exception:
                _logger.debug("count failed on %s", shard_path, exc_info=True)
        return len(seen_ids)

    def reset(self, scope_prefix: str | None = None) -> None:
        """Reset (delete all) memories in scope."""
        if scope_prefix is None or not scope_prefix.strip("/"):
            for shard_path in (self._central_path, self._local_path):
                if shard_path.exists():
                    shutil.rmtree(shard_path, ignore_errors=True)
            self._local_has_data = False
            self._indexes_created = False
            return

        self.delete(scope_prefix=scope_prefix)

    def touch_records(self, record_ids: list[str]) -> None:
        """Update last_accessed to now for the given record IDs."""
        if not record_ids:
            return
        now = datetime.now(timezone.utc).replace(tzinfo=None).isoformat()
        point_ids: list[int | uuid.UUID | str] = [
            _uuid_to_point_id(rid) for rid in record_ids
        ]
        for shard_path in (self._central_path, self._local_path):
            if not shard_path.exists():
                continue
            try:
                shard = EdgeShard.load(str(shard_path))
                shard.update(
                    UpdateOperation.set_payload(point_ids, {"last_accessed": now})
                )
                shard.flush()
                shard.close()
            except Exception:
                _logger.debug("touch_records failed on %s", shard_path, exc_info=True)

    def optimize(self) -> None:
        """Compact the central shard synchronously."""
        if not self._central_path.exists():
            return
        try:
            shard = EdgeShard.load(str(self._central_path))
            shard.optimize()
            shard.close()
        except Exception:
            _logger.debug("optimize failed", exc_info=True)

    def _upsert_to_central(self, points: list[Any]) -> None:
        """Convert scrolled points to Qdrant Points and upsert to central shard."""
        qdrant_points = [
            Point(
                id=pt.id,
                vector=pt.vector if pt.vector else {},
                payload=pt.payload if pt.payload else {},
            )
            for pt in points
        ]
        central = self._open_shard(self._central_path)
        try:
            self._ensure_indexes(central)
            central.update(UpdateOperation.upsert_points(qdrant_points))
            central.flush()
        finally:
            central.close()

    def flush_to_central(self) -> None:
        """Sync local shard records to the central shard."""
        if not self._local_has_data or not self._local_path.exists():
            return

        try:
            local = EdgeShard.load(str(self._local_path))
        except Exception:
            _logger.debug("flush_to_central: failed to open local shard", exc_info=True)
            return

        points = self._scroll_all(local, with_vector=True)
        local.close()

        if not points:
            shutil.rmtree(self._local_path, ignore_errors=True)
            self._local_has_data = False
            return

        self._upsert_to_central(points)
        shutil.rmtree(self._local_path, ignore_errors=True)
        self._local_has_data = False

    def close(self) -> None:
        """Flush local shard to central and clean up."""
        if self._closed:
            return
        self._closed = True
        atexit.unregister(self.close)
        try:
            self.flush_to_central()
        except Exception:
            _logger.debug("close: flush_to_central failed", exc_info=True)

    def _cleanup_orphaned_shards(self) -> None:
        """Sync and remove local shards from dead worker processes."""
        if not self._base_path.exists():
            return
        for entry in self._base_path.iterdir():
            if not entry.is_dir() or not entry.name.startswith("worker-"):
                continue
            pid_str = entry.name.removeprefix("worker-")
            try:
                pid = int(pid_str)
            except ValueError:
                continue
            if pid == os.getpid():
                continue
            try:
                os.kill(pid, 0)
                continue
            except ProcessLookupError:
                _logger.debug("Worker %d is dead, shard is orphaned", pid)
            except PermissionError:
                continue

            _logger.info("Cleaning up orphaned shard for dead worker %d", pid)
            try:
                orphan = EdgeShard.load(str(entry))
                points = self._scroll_all(orphan, with_vector=True)
                orphan.close()

                if not points:
                    shutil.rmtree(entry, ignore_errors=True)
                    continue

                if self._config is None:
                    for pt in points:
                        vec = pt.vector
                        if isinstance(vec, dict) and VECTOR_NAME in vec:
                            vec_data = vec[VECTOR_NAME]
                            if isinstance(vec_data, list) and len(vec_data) > 0:
                                self._vector_dim = len(vec_data)
                                self._config = self._build_config(self._vector_dim)
                                break

                if self._config is None:
                    _logger.warning(
                        "Cannot recover orphaned shard %s: vector dimension unknown",
                        entry,
                    )
                    continue

                self._upsert_to_central(points)
                shutil.rmtree(entry, ignore_errors=True)
            except Exception:
                _logger.warning(
                    "Failed to recover orphaned shard %s", entry, exc_info=True
                )

    async def asave(self, records: list[MemoryRecord]) -> None:
        """Save memory records asynchronously."""
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
        """Search for memories asynchronously."""
        return await asyncio.to_thread(
            self.search,
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
        """Delete memories asynchronously."""
        return await asyncio.to_thread(
            self.delete,
            scope_prefix=scope_prefix,
            categories=categories,
            record_ids=record_ids,
            older_than=older_than,
            metadata_filter=metadata_filter,
        )
