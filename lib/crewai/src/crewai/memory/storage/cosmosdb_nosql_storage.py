"""Azure CosmosDB NoSQL storage backend for the unified memory system.

Stores :class:`crewai.memory.types.MemoryRecord` instances in an Azure
CosmosDB for NoSQL container with vector search enabled.

Container layout (created automatically on first use):

* Partition key:  ``/scope`` (hash) — most queries filter by scope prefix.
* Vector index:   ``/embedding`` (diskANN, cosine distance).
* Document fields::

    {
        "id": "<record id>",            # also used as Cosmos doc id
        "scope": "/...",                # partition key
        "content": "...",
        "categories": ["..."],
        "metadata": {...},
        "importance": 0.5,
        "created_at": "ISO-8601",
        "last_accessed": "ISO-8601",
        "source": "...",
        "private": false,
        "embedding": [..]               # vector
    }

The optional ``azure-cosmos`` / ``azure-identity`` dependencies are imported
lazily so simply ``import crewai`` does not require the extra to be
installed; instantiating :class:`CosmosDBNoSqlStorage` does.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
from typing import Any

from crewai.memory.types import MemoryRecord, ScopeInfo


_logger = logging.getLogger(__name__)

DEFAULT_VECTOR_DIM = 1536
DEFAULT_DATABASE_NAME = "crewai_memory"
DEFAULT_CONTAINER_NAME = "memories"

_INSTALL_HINT = (
    "CosmosDBNoSqlStorage requires the optional 'cosmosdb' extra. "
    "Install it with: pip install 'crewai[cosmosdb]'"
)


def _require_azure_cosmos() -> Any:
    """Lazy-import :mod:`azure.cosmos` with a clear error message."""
    try:
        import azure.cosmos  # noqa: F401
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    import azure.cosmos as azure_cosmos

    return azure_cosmos


def _default_indexing_policy() -> dict[str, Any]:
    return {
        "indexingMode": "consistent",
        "automatic": True,
        "includedPaths": [{"path": "/*"}],
        "excludedPaths": [
            {"path": "/embedding/*"},
            {"path": '/"_etag"/?'},
        ],
        "vectorIndexes": [{"path": "/embedding", "type": "diskANN"}],
    }


def _default_vector_embedding_policy(dimensions: int) -> dict[str, Any]:
    return {
        "vectorEmbeddings": [
            {
                "path": "/embedding",
                "dataType": "float32",
                "distanceFunction": "cosine",
                "dimensions": dimensions,
            }
        ]
    }


def _to_naive_utc(dt: datetime) -> datetime:
    """Convert an aware datetime to naive UTC; leave naive datetimes as-is."""
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return _to_naive_utc(value)
    if value is None:
        return datetime.utcnow()
    return _to_naive_utc(datetime.fromisoformat(str(value).replace("Z", "+00:00")))


def _quote(value: str) -> str:
    """SQL-quote a string literal for safe interpolation in Cosmos SQL."""
    return "'" + str(value).replace("'", "''") + "'"


# Cosmos VectorDistance() returns higher == better for cosine/dot-product
# (it is rescaled to behave like a similarity), and lower == better for
# euclidean. We inspect the configured distance function to score correctly.
_HIGHER_BETTER = {"cosine", "dotproduct", "dot_product"}
_LOWER_BETTER = {"euclidean"}


def _score_from_distance(raw: float, distance_function: str) -> float:
    """Map Cosmos' raw score to a 0..1 'higher is better' score."""
    fn = (distance_function or "cosine").lower().replace("_", "")
    if fn in _HIGHER_BETTER:
        # Cosmos cosine/dot-product: already similarity-like (~0..1, higher better).
        return float(raw)
    # Euclidean: monotonic transform so higher = better.
    return 1.0 / (1.0 + float(raw))


class CosmosDBNoSqlStorage:
    """CosmosDB NoSQL backend for crewAI's unified memory system."""

    def __init__(
        self,
        cosmos_host: str | None = None,
        key: str | None = None,
        token_credential: Any | None = None,
        database_name: str = DEFAULT_DATABASE_NAME,
        container_name: str = DEFAULT_CONTAINER_NAME,
        vector_dim: int = DEFAULT_VECTOR_DIM,
        indexing_policy: dict[str, Any] | None = None,
        vector_embedding_policy: dict[str, Any] | None = None,
        offer_throughput: int | None = None,
        create_container: bool = True,
        user_agent: str = "CrewAI-CosmosDB-Memory-Python",
        *,
        connection_string: str | None = None,
    ) -> None:
        """Initialize the CosmosDB storage backend.

        Args:
            cosmos_host: Cosmos account endpoint (``https://<acct>.documents.azure.com:443/``).
                Required unless ``connection_string`` is given.
            key: Account primary/secondary key. Mutually exclusive with ``token_credential``.
            token_credential: ``azure.core.credentials.TokenCredential`` (e.g.
                ``DefaultAzureCredential``). Mutually exclusive with ``key``.
            database_name: Database to use; created if missing.
            container_name: Container to use; created if missing.
            vector_dim: Embedding dimensionality used at container creation.
            indexing_policy: Override indexing policy (default enables vector index).
            vector_embedding_policy: Override vector embedding policy.
            offer_throughput: Provisioned RU/s for new containers (None = serverless).
            create_container: If False, expect the container to already exist.
            user_agent: Suffix appended to the Cosmos SDK user agent.
            connection_string: Cosmos account connection string (host + key). When
                provided it is used instead of ``cosmos_host`` + ``key`` /
                ``token_credential``.
        """
        azure_cosmos = _require_azure_cosmos()

        if connection_string is not None:
            if key is not None or token_credential is not None:
                raise ValueError(
                    "Provide 'connection_string' on its own, not together with "
                    "'key' or 'token_credential'."
                )
            self._client = azure_cosmos.CosmosClient.from_connection_string(
                connection_string, user_agent=user_agent
            )
        else:
            if key is not None and token_credential is not None:
                raise ValueError(
                    "Provide either 'key' or 'token_credential', not both."
                )
            if key is None and token_credential is None:
                raise ValueError(
                    "Provide one of 'key', 'token_credential' or 'connection_string'."
                )
            if not cosmos_host:
                raise ValueError(
                    "'cosmos_host' is required when authenticating with a key or "
                    "token credential."
                )
            credential: Any = key if key is not None else token_credential
            self._client = azure_cosmos.CosmosClient(
                cosmos_host, credential, user_agent=user_agent
            )

        self._cosmos_host = cosmos_host
        self._database_name = database_name
        self._container_name = container_name
        self._vector_dim = vector_dim

        self._database = self._client.create_database_if_not_exists(
            id=database_name,
            offer_throughput=offer_throughput,
        )
        partition_key = azure_cosmos.PartitionKey(path="/scope", kind="Hash")
        effective_vector_policy = (
            vector_embedding_policy or _default_vector_embedding_policy(vector_dim)
        )
        embeddings_def = (effective_vector_policy.get("vectorEmbeddings") or [{}])[0]
        self._distance_function = embeddings_def.get("distanceFunction", "cosine")
        if create_container:
            self._container = self._database.create_container_if_not_exists(
                id=container_name,
                partition_key=partition_key,
                indexing_policy=indexing_policy or _default_indexing_policy(),
                vector_embedding_policy=effective_vector_policy,
                offer_throughput=offer_throughput,
            )
        else:
            self._container = self._database.get_container_client(container_name)

    # ------------------------------------------------------------------
    # constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_env(cls, **overrides: Any) -> CosmosDBNoSqlStorage:
        """Build a storage backend from environment variables.

        Recognised variables (first matching auth source wins):

        * ``AZURE_COSMOS_CONNECTION_STRING`` -- full account connection string.
        * ``AZURE_COSMOS_HOST`` + ``AZURE_COSMOS_KEY`` -- endpoint + account key.
        * ``AZURE_COSMOS_HOST`` alone -- endpoint authenticated with
          ``azure.identity.DefaultAzureCredential`` (requires the ``azure-identity``
          package from the ``cosmosdb`` extra).

        Optional overrides via env: ``AZURE_COSMOS_DATABASE_NAME``,
        ``AZURE_COSMOS_CONTAINER_NAME``, ``AZURE_COSMOS_VECTOR_DIM``. Any keyword
        ``overrides`` take precedence over the environment.

        Raises:
            ValueError: If no usable authentication variables are set.
        """
        import os

        kwargs: dict[str, Any] = {
            "database_name": os.environ.get(
                "AZURE_COSMOS_DATABASE_NAME", DEFAULT_DATABASE_NAME
            ),
            "container_name": os.environ.get(
                "AZURE_COSMOS_CONTAINER_NAME", DEFAULT_CONTAINER_NAME
            ),
        }
        vector_dim = os.environ.get("AZURE_COSMOS_VECTOR_DIM")
        if vector_dim:
            kwargs["vector_dim"] = int(vector_dim)
        kwargs.update(overrides)

        connection_string = os.environ.get("AZURE_COSMOS_CONNECTION_STRING")
        host = os.environ.get("AZURE_COSMOS_HOST")
        key = os.environ.get("AZURE_COSMOS_KEY")

        if connection_string:
            return cls(connection_string=connection_string, **kwargs)
        if host and key:
            return cls(cosmos_host=host, key=key, **kwargs)
        if host:
            try:
                from azure.identity import DefaultAzureCredential
            except ImportError as exc:
                raise ImportError(_INSTALL_HINT) from exc
            return cls(
                cosmos_host=host,
                token_credential=DefaultAzureCredential(),
                **kwargs,
            )
        raise ValueError(
            "CosmosDBNoSqlStorage.from_env() requires "
            "'AZURE_COSMOS_CONNECTION_STRING', or 'AZURE_COSMOS_HOST' with "
            "'AZURE_COSMOS_KEY' (or DefaultAzureCredential) to be set."
        )

    # ------------------------------------------------------------------
    # (de)serialisation
    # ------------------------------------------------------------------

    def _record_to_doc(self, record: MemoryRecord) -> dict[str, Any]:
        return {
            "id": record.id,
            "scope": record.scope or "/",
            "content": record.content,
            "categories": list(record.categories),
            "metadata": dict(record.metadata),
            "importance": float(record.importance),
            "created_at": _to_naive_utc(record.created_at).isoformat(),
            "last_accessed": _to_naive_utc(record.last_accessed).isoformat(),
            "source": record.source,
            "private": bool(record.private),
            "embedding": list(record.embedding)
            if record.embedding is not None
            else [0.0] * self._vector_dim,
        }

    def _doc_to_record(self, doc: dict[str, Any]) -> MemoryRecord:
        return MemoryRecord(
            id=str(doc["id"]),
            content=str(doc.get("content", "")),
            scope=str(doc.get("scope", "/")),
            categories=list(doc.get("categories") or []),
            metadata=dict(doc.get("metadata") or {}),
            importance=float(doc.get("importance", 0.5)),
            created_at=_parse_dt(doc.get("created_at")),
            last_accessed=_parse_dt(doc.get("last_accessed")),
            embedding=doc.get("embedding"),
            source=doc.get("source") or None,
            private=bool(doc.get("private", False)),
        )

    # ------------------------------------------------------------------
    # filter helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _scope_prefix_filter(scope_prefix: str | None) -> str | None:
        """Return a SQL fragment selecting docs whose scope is under ``scope_prefix``."""
        if scope_prefix is None:
            return None
        prefix = scope_prefix.rstrip("/")
        if not prefix or prefix == "":
            return None
        if not prefix.startswith("/"):
            prefix = "/" + prefix
        return f"(c.scope = {_quote(prefix)} OR STARTSWITH(c.scope, {_quote(prefix + '/')}))"

    @staticmethod
    def _categories_filter(categories: list[str] | None) -> str | None:
        if not categories:
            return None
        clauses = [f"ARRAY_CONTAINS(c.categories, {_quote(cat)})" for cat in categories]
        return "(" + " OR ".join(clauses) + ")"

    @staticmethod
    def _metadata_filter(metadata_filter: dict[str, Any] | None) -> str | None:
        if not metadata_filter:
            return None
        clauses: list[str] = []
        for key, value in metadata_filter.items():
            if not key.replace("_", "").isalnum():
                raise ValueError(
                    f"metadata_filter key {key!r} must be alphanumeric/underscore"
                )
            if isinstance(value, str):
                clauses.append(f"c.metadata.{key} = {_quote(value)}")
            elif isinstance(value, bool):
                clauses.append(f"c.metadata.{key} = {'true' if value else 'false'}")
            elif isinstance(value, (int, float)):
                clauses.append(f"c.metadata.{key} = {value}")
            else:
                clauses.append(f"c.metadata.{key} = {_quote(str(value))}")
        return "(" + " AND ".join(clauses) + ")"

    # ------------------------------------------------------------------
    # write API
    # ------------------------------------------------------------------

    def save(self, records: list[MemoryRecord]) -> None:
        for record in records:
            self._container.upsert_item(body=self._record_to_doc(record))

    def update(self, record: MemoryRecord) -> None:
        self._container.upsert_item(body=self._record_to_doc(record))

    # ------------------------------------------------------------------
    # read API
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        if not query_embedding:
            return []

        where_parts = [
            part
            for part in (
                self._scope_prefix_filter(scope_prefix),
                self._categories_filter(categories),
                self._metadata_filter(metadata_filter),
            )
            if part
        ]
        where_clause = (" WHERE " + " AND ".join(where_parts)) if where_parts else ""

        # Cosmos NoSQL VectorDistance does not currently accept parameter
        # placeholders, so the embedding is interpolated. Values come from
        # the embedder, not user-supplied text, so injection risk is bounded.
        sql = (
            f"SELECT TOP @top c.id, c.scope, c.content, c.categories, c.metadata, "  # noqa: S608
            f"c.importance, c.created_at, c.last_accessed, c.source, c.private, "
            f"c.embedding, "
            f"VectorDistance(c.embedding, {list(query_embedding)}) AS _distance "
            f"FROM c{where_clause} "
            f"ORDER BY VectorDistance(c.embedding, {list(query_embedding)})"
        )
        parameters = [{"name": "@top", "value": limit}]
        items = list(
            self._container.query_items(
                query=sql,
                parameters=parameters,
                enable_cross_partition_query=True,
            )
        )
        out: list[tuple[MemoryRecord, float]] = []
        for doc in items:
            distance = float(doc.pop("_distance", 1.0))
            score = _score_from_distance(distance, self._distance_function)
            if score < min_score:
                continue
            out.append((self._doc_to_record(doc), score))
        return out

    def get_record(self, record_id: str) -> MemoryRecord | None:
        sql = "SELECT TOP 1 * FROM c WHERE c.id = @id"
        items = list(
            self._container.query_items(
                query=sql,
                parameters=[{"name": "@id", "value": record_id}],
                enable_cross_partition_query=True,
            )
        )
        if not items:
            return None
        return self._doc_to_record(items[0])

    def list_records(
        self,
        scope_prefix: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        scope_clause = self._scope_prefix_filter(scope_prefix)
        where = (" WHERE " + scope_clause) if scope_clause else ""
        sql = (
            f"SELECT * FROM c{where} ORDER BY c.created_at DESC "  # noqa: S608
            f"OFFSET {int(offset)} LIMIT {int(limit)}"
        )
        items = list(
            self._container.query_items(
                query=sql,
                enable_cross_partition_query=True,
            )
        )
        return [self._doc_to_record(doc) for doc in items]

    def get_scope_info(self, scope: str) -> ScopeInfo:
        scope = scope.rstrip("/") or "/"
        clause = self._scope_prefix_filter(scope) or "true"
        where = f" WHERE {clause}"
        agg_sql = (
            f"SELECT COUNT(1) AS n, MIN(c.created_at) AS oldest, "  # noqa: S608
            f"MAX(c.created_at) AS newest FROM c{where}"
        )
        agg_rows = list(
            self._container.query_items(
                query=agg_sql, enable_cross_partition_query=True
            )
        )
        agg = agg_rows[0] if agg_rows else {}
        record_count = int(agg.get("n", 0) or 0)
        if record_count == 0:
            return ScopeInfo(
                path=scope,
                record_count=0,
                categories=[],
                oldest_record=None,
                newest_record=None,
                child_scopes=[],
            )
        oldest = _parse_dt(agg["oldest"]) if agg.get("oldest") else None
        newest = _parse_dt(agg["newest"]) if agg.get("newest") else None

        # Distinct categories, flattened and de-duplicated server-side.
        cat_sql = f"SELECT DISTINCT VALUE cat FROM c JOIN cat IN c.categories{where}"  # noqa: S608
        categories = sorted(
            str(cat)
            for cat in self._container.query_items(
                query=cat_sql, enable_cross_partition_query=True
            )
        )

        # Distinct scopes -> immediate child scopes.
        scope_sql = f"SELECT DISTINCT VALUE c.scope FROM c{where}"  # noqa: S608
        child_prefix = (scope.rstrip("/") + "/") if scope != "/" else "/"
        children: set[str] = set()
        for raw in self._container.query_items(
            query=scope_sql, enable_cross_partition_query=True
        ):
            sc = str(raw)
            if child_prefix and sc.startswith(child_prefix):
                rest = sc[len(child_prefix) :]
                first = rest.split("/", 1)[0]
                if first:
                    children.add(child_prefix + first)

        return ScopeInfo(
            path=scope,
            record_count=record_count,
            categories=categories,
            oldest_record=oldest,
            newest_record=newest,
            child_scopes=sorted(children),
        )

    def list_scopes(self, parent: str = "/") -> list[str]:
        parent = parent.rstrip("/") or ""
        prefix = (parent + "/") if parent else "/"
        clause = self._scope_prefix_filter(parent or None)
        sql_where = (" WHERE " + clause) if clause else ""
        sql = f"SELECT DISTINCT VALUE c.scope FROM c{sql_where}"  # noqa: S608
        scopes = list(
            self._container.query_items(query=sql, enable_cross_partition_query=True)
        )
        children: set[str] = set()
        for raw in scopes:
            sc = str(raw)
            if not sc.startswith(prefix):
                continue
            if sc == prefix.rstrip("/") or sc == "/":
                continue
            rest = sc[len(prefix) :]
            first = rest.split("/", 1)[0]
            if first:
                children.add(prefix + first)
        return sorted(children)

    def list_categories(self, scope_prefix: str | None = None) -> dict[str, int]:
        clause = self._scope_prefix_filter(scope_prefix)
        where = (" WHERE " + clause) if clause else ""
        sql = (
            f"SELECT cat AS category, COUNT(1) AS n "  # noqa: S608
            f"FROM c JOIN cat IN c.categories{where} GROUP BY cat"
        )
        rows = list(
            self._container.query_items(query=sql, enable_cross_partition_query=True)
        )
        counts: dict[str, int] = {}
        for row in rows:
            cat = row.get("category")
            if cat is not None:
                counts[str(cat)] = int(row.get("n", 0) or 0)
        return counts

    def count(self, scope_prefix: str | None = None) -> int:
        clause = self._scope_prefix_filter(scope_prefix)
        where = (" WHERE " + clause) if clause else ""
        sql = f"SELECT VALUE COUNT(1) FROM c{where}"  # noqa: S608
        rows = list(
            self._container.query_items(query=sql, enable_cross_partition_query=True)
        )
        return int(rows[0]) if rows else 0

    # ------------------------------------------------------------------
    # delete API
    # ------------------------------------------------------------------

    def delete(
        self,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        # Cosmos NoSQL has no native bulk-delete-by-query in GA today, so we
        # query the matching set first and delete document-by-document.
        where_parts: list[str] = []
        if (clause := self._scope_prefix_filter(scope_prefix)) is not None:
            where_parts.append(clause)
        if (clause := self._categories_filter(categories)) is not None:
            where_parts.append(clause)
        if (clause := self._metadata_filter(metadata_filter)) is not None:
            where_parts.append(clause)
        if record_ids:
            id_list = ", ".join(_quote(rid) for rid in record_ids)
            where_parts.append(f"c.id IN ({id_list})")
        if older_than is not None:
            where_parts.append(
                f"c.created_at < {_quote(_to_naive_utc(older_than).isoformat())}"
            )
        if not where_parts:
            # Refuse to wipe the whole container via delete() — use reset() for that.
            raise ValueError(
                "delete() requires at least one filter (scope_prefix, categories, "
                "record_ids, older_than, or metadata_filter). Use reset() to wipe "
                "the entire container."
            )
        where_clause = " AND ".join(where_parts)
        sql = f"SELECT c.id, c.scope FROM c WHERE {where_clause}"  # noqa: S608
        targets = list(
            self._container.query_items(query=sql, enable_cross_partition_query=True)
        )
        deleted = 0
        for target in targets:
            try:
                self._container.delete_item(
                    item=target["id"], partition_key=target["scope"]
                )
                deleted += 1
            except Exception:  # noqa: PERF203
                _logger.debug(
                    "Failed to delete item %s in scope %s",
                    target.get("id"),
                    target.get("scope"),
                    exc_info=True,
                )
        return deleted

    def reset(self, scope_prefix: str | None = None) -> None:
        if scope_prefix is None or scope_prefix.strip("/") == "":
            azure_cosmos = _require_azure_cosmos()
            try:
                self._database.delete_container(self._container_name)
            except azure_cosmos.exceptions.CosmosResourceNotFoundError:
                # Container already absent — nothing to drop, safe to recreate.
                pass
            # Any other error (throttling, auth, etc.) must propagate: silently
            # continuing would recreate/return a container that still holds
            # every record while reporting a successful reset.
            partition_key = azure_cosmos.PartitionKey(path="/scope", kind="Hash")
            self._container = self._database.create_container_if_not_exists(
                id=self._container_name,
                partition_key=partition_key,
                indexing_policy=_default_indexing_policy(),
                vector_embedding_policy=_default_vector_embedding_policy(
                    self._vector_dim
                ),
            )
            return
        # Per-scope reset = delete with scope_prefix.
        self.delete(scope_prefix=scope_prefix)

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Release the underlying ``CosmosClient`` (idempotent).

        ``Memory.close()`` calls this when the backend defines it; without it the
        client's HTTP connection pool would leak for the process lifetime.
        """
        client = getattr(self, "_client", None)
        if client is None:
            return
        try:
            close = getattr(client, "close", None)
            if callable(close):
                close()
        except Exception:  # pragma: no cover - defensive cleanup
            _logger.debug("Failed to close Cosmos client", exc_info=True)
        finally:
            self._client = None

    def __enter__(self) -> CosmosDBNoSqlStorage:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # async API (delegates to sync via threadpool)
    # ------------------------------------------------------------------

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
