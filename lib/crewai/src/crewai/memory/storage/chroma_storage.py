"""Chroma storage backend for the unified memory system.

Keeps the records in a collection of a Chroma server, or of the in-process client of
the ``chromadb`` package. It works as ``LanceDBStorage`` does: the same fields, scores
of ``1 / (1 + distance)`` with the squared L2 distance of Chroma, the same dimension
checks. As in ``QdrantEdgeStorage``, scopes are matched by path and a search by category
is filtered by the server before the nearest records are taken.

No filter depends on a list in the metadata being absent: the scope paths are written
for every record, and a search by category also checks the JSON copy of the categories.
"""

from __future__ import annotations

from datetime import datetime
import json
import re
from typing import Any

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from chromadb.api.types import Metadata, PyEmbedding
from chromadb.errors import InvalidArgumentError, NotFoundError

from crewai.memory.storage.backend import EmbeddingDimensionMismatchError
from crewai.memory.types import MemoryRecord, ScopeInfo


# Default embedding vector dimensionality, as in LanceDBStorage. Used when the first
# write carries no embedding at all.
DEFAULT_VECTOR_DIM = 3072

# Safety cap on the number of records read by a single scan, as in LanceDBStorage:
# scope info, counts in a scope, categories and deletes by category or metadata.
_SCAN_ROWS_LIMIT = 50_000

# Records read by one request of a scan.
_SCAN_PAGE = 5_000

_DIMENSION_ERROR = re.compile(r"expecting embedding with dimension of (\d+), got (\d+)")


def _scope_ancestors(scope: str) -> list[str]:
    """Build the list of all ancestor scopes for scope filtering.

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


def _scope_where(scope_prefix: str | None) -> dict[str, Any] | None:
    """Build the filter of a scope and its subscopes, or None for the root scope."""
    if scope_prefix is None or not scope_prefix.strip("/"):
        return None
    prefix = scope_prefix.rstrip("/")
    if not prefix.startswith("/"):
        prefix = "/" + prefix
    return {"scope_ancestors": {"$contains": prefix}}


def _dimension_mismatch(error: InvalidArgumentError) -> EmbeddingDimensionMismatchError:
    """Turn a dimension refused by Chroma into EmbeddingDimensionMismatchError."""
    found = _DIMENSION_ERROR.search(str(error))
    if found is None:
        raise error
    return EmbeddingDimensionMismatchError(int(found.group(1)), int(found.group(2)))


def _parse_dt(val: Any) -> datetime:
    if val is None:
        return datetime.utcnow()
    if isinstance(val, datetime):
        return val
    return datetime.fromisoformat(str(val).replace("Z", "+00:00"))


class ChromaStorage:
    """Chroma-backed storage for the unified memory system."""

    def __init__(
        self,
        client: ClientAPI | None = None,
        collection_name: str = "memories",
        vector_dim: int | None = None,
    ) -> None:
        """Initialize Chroma storage.

        Args:
            client: A client of the ``chromadb`` package, such as
                  ``chromadb.HttpClient(host=..., port=...)`` or
                  ``chromadb.PersistentClient(path=...)``. Defaults to
                  ``chromadb.HttpClient()`` with its default host and port.
            collection_name: Name of the collection for memory records.
            vector_dim: Dimensionality of the embedding vector. When ``None``
                  (default), the dimension is read from the existing collection or
                  taken from the first saved embedding.
        """
        self._client = client if client is not None else chromadb.HttpClient()
        self._collection_name = collection_name
        self._collection: Collection | None = None
        self._vector_dim = 0  # 0 = not yet known
        collection = self._open(create=vector_dim is not None)
        if collection is not None:
            self._vector_dim = collection.get_model().dimension or vector_dim or 0

    def _open(self, create: bool) -> Collection | None:
        """Return the collection; create it if ``create``, else None if missing."""
        if self._collection is None:
            if create:
                self._collection = self._client.get_or_create_collection(
                    self._collection_name, embedding_function=None
                )
            else:
                try:
                    self._collection = self._client.get_collection(
                        self._collection_name, embedding_function=None
                    )
                except NotFoundError:
                    return None
        return self._collection

    def _batch_size(self) -> int:
        return int(self._client.get_max_batch_size())

    def _record_to_row(
        self, record: MemoryRecord
    ) -> tuple[list[float], dict[str, Any]]:
        metadata: dict[str, Any] = {
            "scope": record.scope,
            "scope_ancestors": _scope_ancestors(record.scope),
            # The list for the category filter of a search. Chroma takes no empty
            # list: None leaves the key out of an add and removes it on an upsert.
            "categories": list(record.categories) or None,
            "categories_str": json.dumps(record.categories),
            "metadata_str": json.dumps(record.metadata),
            "importance": record.importance,
            "created_at": record.created_at.isoformat(),
            "last_accessed": record.last_accessed.isoformat(),
            "source": record.source or "",
            "private": record.private,
        }
        # As in LanceDBStorage, a missing or wrong-sized embedding becomes zeros.
        embedding = record.embedding or []
        if len(embedding) == self._vector_dim:
            return list(embedding), metadata
        return [0.0] * self._vector_dim, metadata

    @staticmethod
    def _row_to_record(
        record_id: str,
        document: str | None,
        metadata: dict[str, Any],
        embedding: Any = None,
    ) -> MemoryRecord:
        return MemoryRecord(
            id=str(record_id),
            content=str(document or ""),
            scope=str(metadata.get("scope", "/")),
            categories=json.loads(metadata["categories_str"])
            if metadata.get("categories_str")
            else [],
            metadata=json.loads(metadata["metadata_str"])
            if metadata.get("metadata_str")
            else {},
            importance=float(metadata.get("importance", 0.5)),
            created_at=_parse_dt(metadata.get("created_at")),
            last_accessed=_parse_dt(metadata.get("last_accessed")),
            embedding=[float(x) for x in embedding] if embedding is not None else None,
            source=metadata.get("source") or None,
            private=bool(metadata.get("private", False)),
        )

    def _write(
        self,
        collection: Collection,
        records: list[MemoryRecord],
        upsert: bool = False,
    ) -> None:
        """Add or upsert records in batches of the size the server takes."""
        rows = [self._record_to_row(record) for record in records]
        size = self._batch_size()
        for start in range(0, len(records), size):
            batch = records[start : start + size]
            ids = [record.id for record in batch]
            documents = [record.content for record in batch]
            embeddings: list[PyEmbedding] = [v for v, _ in rows[start : start + size]]
            metadatas: list[Metadata] = [m for _, m in rows[start : start + size]]
            try:
                if upsert:
                    collection.upsert(
                        ids=ids,
                        embeddings=embeddings,
                        documents=documents,
                        metadatas=metadatas,
                    )
                else:
                    collection.add(
                        ids=ids,
                        embeddings=embeddings,
                        documents=documents,
                        metadatas=[
                            {k: v for k, v in metadata.items() if v is not None}
                            for metadata in metadatas
                        ],
                    )
            except InvalidArgumentError as e:
                raise _dimension_mismatch(e) from e

    def _scan(
        self, scope_prefix: str | None = None, capped: bool = True
    ) -> list[tuple[str, dict[str, Any]]]:
        """Return ids and metadata in a scope, at most ``_SCAN_ROWS_LIMIT`` if capped."""
        collection = self._open(create=False)
        if collection is None:
            return []
        limit = _SCAN_ROWS_LIMIT if capped else None
        where = _scope_where(scope_prefix)
        out: list[tuple[str, dict[str, Any]]] = []
        while limit is None or len(out) < limit:
            page = _SCAN_PAGE if limit is None else min(_SCAN_PAGE, limit - len(out))
            rows = collection.get(
                where=where, include=["metadatas"], limit=page, offset=len(out)
            )
            metadatas = rows["metadatas"] or []
            out.extend(
                (record_id, dict(metadata or {}))
                for record_id, metadata in zip(rows["ids"], metadatas, strict=False)
            )
            if len(rows["ids"]) < page:
                break
        return out

    def _delete_ids(self, collection: Collection, ids: list[str]) -> int:
        size = self._batch_size()
        for start in range(0, len(ids), size):
            collection.delete(ids=ids[start : start + size])
        return len(ids)

    def save(self, records: list[MemoryRecord]) -> None:
        if not records:
            return
        # Auto-detect dimension from the first real embedding and validate the
        # whole batch against it, as LanceDBStorage does.
        dim = None
        for r in records:
            if r.embedding and len(r.embedding) > 0:
                if dim is None:
                    dim = len(r.embedding)
                elif len(r.embedding) != dim:
                    raise EmbeddingDimensionMismatchError(dim, len(r.embedding))
        is_new_collection = self._open(create=False) is None
        if (
            not is_new_collection
            and dim
            and self._vector_dim
            and dim != self._vector_dim
        ):
            raise EmbeddingDimensionMismatchError(self._vector_dim, dim)
        collection = self._open(create=True)
        if collection is None:
            return
        if is_new_collection or not self._vector_dim:
            self._vector_dim = dim or self._vector_dim or DEFAULT_VECTOR_DIM
        self._write(collection, records)

    def update(self, record: MemoryRecord) -> None:
        """Update a record by ID: the stored record is replaced, or added if missing."""
        if (
            self._open(create=False) is not None
            and record.embedding
            and self._vector_dim
            and len(record.embedding) != self._vector_dim
        ):
            raise EmbeddingDimensionMismatchError(
                self._vector_dim, len(record.embedding)
            )
        collection = self._open(create=True)
        if collection is None:
            return
        self._vector_dim = self._vector_dim or DEFAULT_VECTOR_DIM
        self._write(collection, [record], upsert=True)

    def touch_records(self, record_ids: list[str]) -> None:
        """Update last_accessed to now for the given record IDs.

        Args:
            record_ids: IDs of records to touch.
        """
        collection = self._open(create=False)
        if not record_ids or collection is None:
            return
        now = datetime.utcnow().isoformat()
        size = self._batch_size()
        for start in range(0, len(record_ids), size):
            ids = [str(rid) for rid in record_ids[start : start + size]]
            collection.update(ids=ids, metadatas=[{"last_accessed": now} for _ in ids])

    def get_record(self, record_id: str) -> MemoryRecord | None:
        """Return a single record by ID, or None if not found."""
        collection = self._open(create=False)
        if collection is None:
            return None
        rows = collection.get(
            ids=[str(record_id)], include=["documents", "metadatas", "embeddings"]
        )
        if not rows["ids"]:
            return None
        documents = rows["documents"]
        metadatas = rows["metadatas"]
        embeddings = rows["embeddings"]
        return self._row_to_record(
            rows["ids"][0],
            documents[0] if documents else None,
            dict(metadatas[0] or {}) if metadatas else {},
            embeddings[0] if embeddings is not None else None,
        )

    def search(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        collection = self._open(create=False)
        if collection is None:
            return []
        if self._vector_dim and len(query_embedding) != self._vector_dim:
            raise EmbeddingDimensionMismatchError(
                self._vector_dim, len(query_embedding)
            )
        where = _scope_where(scope_prefix)
        if categories:
            # Match categories on the server, before the nearest records are taken.
            # A record without categories is left out by its JSON copy.
            any_category: list[dict[str, Any]] = [
                {"categories": {"$contains": c}} for c in dict.fromkeys(categories)
            ]
            where = {
                "$and": [
                    *([where] if where else []),
                    {"categories_str": {"$ne": "[]"}},
                    any_category[0]
                    if len(any_category) == 1
                    else {"$or": any_category},
                ]
            }
        n_results = limit * 3 if metadata_filter else limit
        if n_results <= 0:
            return []
        query: list[PyEmbedding] = [list(query_embedding)]
        try:
            results = collection.query(
                query_embeddings=query,
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "embeddings", "distances"],
            )
        except InvalidArgumentError as e:
            raise _dimension_mismatch(e) from e
        ids = results["ids"][0]
        documents = (results["documents"] or [[]])[0]
        metadatas = (results["metadatas"] or [[]])[0]
        embeddings = (results["embeddings"] or [[]])[0]
        distances = (results["distances"] or [[]])[0]
        out: list[tuple[MemoryRecord, float]] = []
        for record_id, document, metadata, embedding, distance in zip(
            ids, documents, metadatas, embeddings, distances, strict=False
        ):
            record = self._row_to_record(
                record_id, document, dict(metadata or {}), embedding
            )
            if categories and not any(c in record.categories for c in categories):
                continue
            if metadata_filter and not all(
                record.metadata.get(k) == v for k, v in metadata_filter.items()
            ):
                continue
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
        collection = self._open(create=False)
        if collection is None:
            return 0
        if record_ids and not (categories or metadata_filter):
            ids = list(dict.fromkeys(str(rid) for rid in record_ids))
            found = collection.get(ids=ids, include=[])["ids"]
            return self._delete_ids(collection, list(found))
        if categories or metadata_filter:
            to_delete: list[str] = []
            for record_id, metadata in self._scan(scope_prefix):
                record = self._row_to_record(record_id, None, metadata)
                if categories and not any(c in record.categories for c in categories):
                    continue
                if metadata_filter and not all(
                    record.metadata.get(k) == v for k, v in metadata_filter.items()
                ):
                    continue
                if older_than and record.created_at >= older_than:
                    continue
                to_delete.append(record.id)
            return self._delete_ids(collection, to_delete)
        rows = self._scan(scope_prefix, capped=False)
        if older_than is not None:
            # As LanceDBStorage, which compares the stored ISO strings.
            cutoff = older_than.isoformat()
            rows = [
                (rid, m) for rid, m in rows if str(m.get("created_at", "")) < cutoff
            ]
        return self._delete_ids(collection, [rid for rid, _ in rows])

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
        collection = self._open(create=False)
        if collection is None or limit <= 0 or offset < 0:
            return []
        rows = self._scan(scope_prefix, capped=False)
        # As LanceDBStorage, which orders by the stored ISO strings.
        rows.sort(key=lambda row: str(row[1].get("created_at", "")), reverse=True)
        page = [rid for rid, _ in rows[offset : offset + limit]]
        found: dict[str, MemoryRecord] = {}
        size = self._batch_size()
        for start in range(0, len(page), size):
            got = collection.get(
                ids=page[start : start + size],
                include=["documents", "metadatas", "embeddings"],
            )
            documents = got["documents"] or []
            metadatas = got["metadatas"] or []
            embeddings = got["embeddings"]
            for index, record_id in enumerate(got["ids"]):
                found[record_id] = self._row_to_record(
                    record_id,
                    documents[index] if index < len(documents) else None,
                    dict(metadatas[index] or {}) if index < len(metadatas) else {},
                    embeddings[index] if embeddings is not None else None,
                )
        return [found[rid] for rid in page if rid in found]

    def get_scope_info(self, scope: str) -> ScopeInfo:
        scope = scope.rstrip("/") or "/"
        prefix = scope if scope != "/" else ""
        if prefix and not prefix.startswith("/"):
            prefix = "/" + prefix
        rows = self._scan(prefix or None)
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
        for _, metadata in rows:
            sc = str(metadata.get("scope", ""))
            if child_prefix and sc.startswith(child_prefix):
                rest = sc[len(child_prefix) :]
                first_component = rest.split("/", 1)[0]
                if first_component:
                    children.add(child_prefix + first_component)
            try:
                categories_set.update(
                    json.loads(metadata.get("categories_str") or "[]")
                )
            except Exception:  # noqa: S110
                pass
            created = metadata.get("created_at")
            if created:
                dt = _parse_dt(created)
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
        children: set[str] = set()
        for _, metadata in self._scan(prefix if prefix != "/" else None):
            sc = str(metadata.get("scope", ""))
            if sc.startswith(prefix) and sc != (prefix.rstrip("/") or "/"):
                rest = sc[len(prefix) :]
                first_component = rest.split("/", 1)[0]
                if first_component:
                    children.add(prefix + first_component)
        return sorted(children)

    def list_categories(self, scope_prefix: str | None = None) -> dict[str, int]:
        counts: dict[str, int] = {}
        for _, metadata in self._scan(scope_prefix):
            try:
                parsed = json.loads(metadata.get("categories_str") or "[]")
            except Exception:  # noqa: S112
                continue
            for c in parsed:
                counts[c] = counts.get(c, 0) + 1
        return counts

    def count(self, scope_prefix: str | None = None) -> int:
        collection = self._open(create=False)
        if collection is None:
            return 0
        if scope_prefix is None or scope_prefix.strip("/") == "":
            return int(collection.count())
        return self.get_scope_info(scope_prefix).record_count

    def reset(self, scope_prefix: str | None = None) -> None:
        if scope_prefix is None or scope_prefix.strip("/") == "":
            collection = self._open(create=False)
            if collection is not None:
                # The records first, then the collection.
                ids = [rid for rid, _ in self._scan(None, capped=False)]
                self._delete_ids(collection, ids)
            try:
                self._client.delete_collection(self._collection_name)
            except NotFoundError:
                pass
            self._collection = None
            return
        self.delete(scope_prefix=scope_prefix)

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
