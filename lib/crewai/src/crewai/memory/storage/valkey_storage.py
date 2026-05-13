"""Valkey-backed storage for the unified memory system.

This module provides ValkeyStorage, a distributed storage backend that implements
the StorageBackend protocol using Valkey-GLIDE as the underlying data store.
It supports vector similarity search via Valkey Search module and provides
efficient indexing for scope, category, and metadata filtering.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from datetime import datetime
import json
import logging
import threading
from typing import Any

from glide import (
    BackoffStrategy,
    ConfigurationError,
    ConnectionError,
    DataType,
    DistanceMetricType,
    Field,
    FtCreateOptions,
    FtSearchLimit,
    FtSearchOptions,
    GlideClient,
    GlideClientConfiguration,
    NodeAddress,
    NumericField,
    RangeByIndex,
    RangeByScore,
    ReturnField,
    ScoreBoundary,
    ServerCredentials,
    TagField,
    VectorAlgorithm,
    VectorField,
    VectorFieldAttributesFlat,
    VectorFieldAttributesHnsw,
    VectorType,
    ft,
)
import numpy as np

from crewai.memory.types import MemoryRecord, ScopeInfo


_logger = logging.getLogger(__name__)


class ValkeyStorage:
    """Valkey-backed storage for the unified memory system.

    Provides distributed, high-performance storage using Valkey-GLIDE client.
    Implements the StorageBackend protocol with both sync and async methods.

    This implementation supports standalone Valkey mode only. Cluster mode is
    not supported in this version.

    Example:
        >>> storage = ValkeyStorage(host="localhost", port=6379)
        >>> record = MemoryRecord(content="test", embedding=[0.1, 0.2])
        >>> storage.save([record])
        >>> retrieved = storage.get_record(record.id)
    """

    # ------------------------------------------------------------------
    # Key helpers — single source of truth for Valkey key patterns.
    # Note: dynamic parts (scope, category, metadata values) are not
    # encoded here because Valkey keys are opaque byte strings and the
    # ':' delimiter is only meaningful to our own code.  If cross-tenant
    # isolation is required, callers should validate inputs before
    # passing them to the storage layer.
    # ------------------------------------------------------------------

    @staticmethod
    def _record_key(record_id: str) -> str:
        return f"record:{record_id}"

    @staticmethod
    def _scope_key(scope: str) -> str:
        return f"scope:{scope}"

    @staticmethod
    def _category_key(category: str) -> str:
        return f"category:{category}"

    @staticmethod
    def _metadata_key(key: str, value: str) -> str:
        return f"metadata:{key}:{value}"

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        use_tls: bool = False,
        tls_ca_cert_path: str | None = None,
        tls_client_cert_path: str | None = None,
        tls_client_key_path: str | None = None,
        vector_dim: int = 1536,
        index_algorithm: str = "HNSW",
    ) -> None:
        """Initialize Valkey storage with connection parameters and vector index config.

        Note: This implementation supports standalone Valkey mode only.
        Cluster mode is not supported in this version.

        TLS Support: Basic TLS encryption is supported via ``use_tls=True``.
        Custom CA and client certificates are defined in the GLIDE Rust core
        but not yet exposed in the Python bindings. The ``tls_*_path``
        parameters are accepted for forward compatibility and will be wired
        in once the Python client adds support.

        Args:
            host: Valkey server hostname.
            port: Valkey server port.
            db: Database number to use (standalone mode only).
            password: Optional password for authentication.
            use_tls: Enable TLS/SSL encryption for connections.
            tls_ca_cert_path: Path to CA certificate (forward-compat, not yet wired).
            tls_client_cert_path: Path to client certificate (forward-compat, not yet wired).
            tls_client_key_path: Path to client key (forward-compat, not yet wired).
            vector_dim: Dimension of embedding vectors (default 1536 for OpenAI).
            index_algorithm: Vector index algorithm ("HNSW" or "FLAT").
        """
        self._host = host
        self._port = port
        self._db = db
        self._password = password
        self._use_tls = use_tls
        self._vector_dim = vector_dim
        self._index_algorithm = index_algorithm
        self._client: GlideClient | None = None
        self._index_created = False
        self._sync_lock = threading.Lock()

        # Reject TLS cert paths until the GLIDE Python client exposes them
        if tls_ca_cert_path or tls_client_cert_path or tls_client_key_path:
            raise NotImplementedError(
                "Custom TLS certificates are not yet supported by the "
                "valkey-glide Python client. Use use_tls=True for basic "
                "TLS with system CA certificates."
            )

        # Write lock for compatibility with memory system
        # Note: Valkey handles concurrency at the server level, so this is a no-op lock
        self._write_lock = threading.RLock()

        # Async lock for lazy client initialization (prevents connection races)
        self._client_lock: asyncio.Lock | None = None

    def _get_client_lock(self) -> asyncio.Lock:
        """Get or create the async client lock (lazy to avoid event loop binding at init)."""
        if self._client_lock is None:
            self._client_lock = asyncio.Lock()
        return self._client_lock

    async def _get_client(self) -> GlideClient:
        """Get or create Valkey client with lazy initialization.

        Uses double-check locking to prevent concurrent callers from
        creating multiple client instances.

        Returns:
            Initialized GlideClient instance.

        Raises:
            RuntimeError: If connection to Valkey fails.
            TimeoutError: If connection attempt times out (10 seconds).
        """
        if self._client is None:
            async with self._get_client_lock():
                if self._client is None:
                    try:
                        # Build node address with explicit host and port
                        node = NodeAddress(host=self._host, port=self._port)

                        # Build configuration
                        config = GlideClientConfiguration(
                            addresses=[node],
                            database_id=self._db,
                            use_tls=self._use_tls,
                            credentials=(
                                ServerCredentials(password=self._password)
                                if self._password
                                else None
                            ),
                            request_timeout=2000,  # 2 seconds for FT.SEARCH and other commands
                            reconnect_strategy=BackoffStrategy(
                                num_of_retries=5,
                                factor=200,  # milliseconds
                                exponent_base=2,
                            ),
                        )

                        # Add connection timeout (10 seconds)
                        try:
                            self._client = await asyncio.wait_for(
                                GlideClient.create(config), timeout=10.0
                            )
                        except asyncio.TimeoutError as e:
                            _logger.error(
                                f"Connection timeout after 10 seconds to Valkey at {self._host}:{self._port}"
                            )
                            raise TimeoutError(
                                f"Connection timeout to Valkey at {self._host}:{self._port}. "
                                "Ensure Valkey is running and accessible."
                            ) from e

                        _logger.info(
                            f"Connected to Valkey at {self._host}:{self._port} (db={self._db}, tls={self._use_tls})"
                        )

                    except (ConfigurationError, ConnectionError) as e:
                        _logger.error(f"Failed to create Valkey client: {e}")
                        raise RuntimeError(
                            f"Cannot connect to Valkey at {self._host}:{self._port}"
                        ) from e

        return self._client

    @property
    def write_lock(self) -> threading.RLock:
        """Write lock for compatibility with memory system.

        Note: Valkey handles concurrency at the server level with atomic operations,
        so this lock is primarily for API compatibility with other storage backends.
        """
        return self._write_lock

    def _run_async(self, coro: Coroutine[Any, Any, Any]) -> Any:
        """Bridge async operations to sync context.

        Uses a dedicated background thread with a persistent event loop so the
        Valkey client (and its TCP connection) can be reused across calls.

        Concurrent sync callers are serialized via a lock to avoid overloading
        the single-threaded background event loop (e.g. when the encoding flow
        dispatches parallel searches from a ThreadPoolExecutor).

        Args:
            coro: Coroutine to execute.

        Returns:
            Result of the coroutine execution.
        """
        with self._sync_lock:
            bg_loop = self._get_or_create_loop()
            future = asyncio.run_coroutine_threadsafe(coro, bg_loop)
            return future.result()

    # ------------------------------------------------------------------
    # Persistent event-loop helpers
    # ------------------------------------------------------------------
    # Class-level: a single background event loop shared by ALL ValkeyStorage
    # instances.  This is intentional — the loop is just an I/O scheduler and
    # the glide client handles per-connection state internally.
    # _bg_lock guards loop creation; _sync_lock (instance-level, set in
    # __init__) serialises sync callers so they don't flood the loop.
    # ------------------------------------------------------------------
    _bg_loop: asyncio.AbstractEventLoop | None = None
    _bg_thread: threading.Thread | None = None
    _bg_lock: threading.Lock = threading.Lock()

    @classmethod
    def _get_or_create_loop(cls) -> asyncio.AbstractEventLoop:
        """Return a long-lived event loop running on a background daemon thread."""
        if cls._bg_loop is not None and cls._bg_loop.is_running():
            return cls._bg_loop

        with cls._bg_lock:
            # Double-check after acquiring lock
            if cls._bg_loop is not None and cls._bg_loop.is_running():
                return cls._bg_loop

            loop = asyncio.new_event_loop()
            thread = threading.Thread(
                target=loop.run_forever, daemon=True, name="valkey-io"
            )
            thread.start()
            cls._bg_loop = loop
            cls._bg_thread = thread
            return loop

    async def __aenter__(self) -> ValkeyStorage:
        """Async context manager entry."""
        await self._get_client()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit."""
        await self.aclose()

    async def aclose(self) -> None:
        """Async close: release the Glide client connection."""
        if self._client:
            await self._client.close()
            self._client = None

    def close(self) -> None:
        """Sync close: release the Glide client connection.

        Called by Memory.close() for deterministic cleanup.
        """
        if self._client:
            try:
                bg_loop = type(self)._bg_loop
                if bg_loop is not None and bg_loop.is_running():
                    asyncio.run_coroutine_threadsafe(
                        self._client.close(), bg_loop
                    ).result(timeout=5)
                else:
                    close_result = self._client.close()
                    if asyncio.iscoroutine(close_result):
                        asyncio.run(close_result)
            except Exception as e:
                _logger.debug(f"Error closing Valkey client: {e}")
            finally:
                self._client = None

    def __del__(self) -> None:
        """Cleanup client connection on deletion."""
        if self._client:
            try:
                bg_loop = type(self)._bg_loop
                if bg_loop is not None and bg_loop.is_running():
                    asyncio.run_coroutine_threadsafe(self._client.close(), bg_loop)
                else:
                    close_result = self._client.close()
                    if asyncio.iscoroutine(close_result):
                        asyncio.run(close_result)
            except Exception as e:
                _logger.debug(f"Error closing client during cleanup: {e}")

    def _embedding_to_bytes(self, embedding: list[float]) -> bytes:
        """Convert embedding list to binary format for Valkey Search.

        Args:
            embedding: List of floats representing the embedding vector.

        Returns:
            Binary representation as float32 array.
        """
        return np.array(embedding, dtype=np.float32).tobytes()

    def _bytes_to_embedding(self, data: bytes) -> list[float]:
        """Convert binary format back to embedding list.

        Args:
            data: Binary data from Valkey.

        Returns:
            List of floats representing the embedding vector.
        """
        arr = np.frombuffer(data, dtype=np.float32)
        return [float(x) for x in arr]

    def _record_to_dict(self, record: MemoryRecord) -> dict[str, str | bytes]:
        """Convert MemoryRecord to Valkey hash fields.

        Args:
            record: Memory record to serialize.

        Returns:
            Dictionary of field names to string/bytes values.

        Raises:
            ValueError: If serialization fails for any field.
        """
        try:
            result: dict[str, str | bytes] = {
                "id": record.id,
                "content": record.content,
                "scope": record.scope,
                "categories": ",".join(record.categories)
                if record.categories
                else "",  # TAG field format
                "metadata": json.dumps(record.metadata),
                "importance": str(record.importance),
                "created_at": record.created_at.isoformat(),
                "last_accessed": record.last_accessed.isoformat(),
                "source": record.source or "",
                "private": "true" if record.private else "false",
            }

            # Add embedding as binary vector field if present
            if record.embedding:
                result["embedding"] = self._embedding_to_bytes(record.embedding)
            else:
                result["embedding"] = b""  # Empty bytes for no embedding

            return result
        except (TypeError, ValueError) as e:
            raise ValueError(f"Failed to serialize record {record.id}: {e}") from e

    def _dict_to_record(
        self, data: dict[str, Any] | dict[bytes, bytes]
    ) -> MemoryRecord | None:
        """Convert Valkey hash fields to MemoryRecord.

        Args:
            data: Dictionary of field names to values from Valkey (may be bytes or str keys/values).

        Returns:
            Reconstructed MemoryRecord, or None if deserialization fails.
        """
        try:
            # Convert bytes keys/values to strings if needed
            str_data: dict[str, Any] = {}
            for key, value in data.items():
                str_key = key.decode("utf-8") if isinstance(key, bytes) else key

                # Handle value conversion - keep embedding as bytes
                if isinstance(value, bytes):
                    if str_key == "embedding":
                        # Keep embedding as bytes - don't try to decode
                        str_data[str_key] = value
                    else:
                        # Try to decode other fields as UTF-8
                        try:
                            str_data[str_key] = value.decode("utf-8")
                        except UnicodeDecodeError:
                            # Keep as bytes if decode fails
                            str_data[str_key] = value
                else:
                    str_data[str_key] = value

            # Deserialize embedding if present
            embedding: list[float] | None = None
            embedding_data = str_data.get("embedding")
            if embedding_data:
                if isinstance(embedding_data, bytes):
                    if len(embedding_data) > 0:
                        embedding = self._bytes_to_embedding(embedding_data)
                    # else: empty bytes, leave embedding as None
                elif isinstance(embedding_data, str) and embedding_data:
                    # Fallback for string representation
                    try:
                        embedding = json.loads(embedding_data)
                    except json.JSONDecodeError:
                        # Invalid JSON, leave as None
                        pass

            # Parse categories - handle both TAG format (comma-separated) and JSON format
            categories_str = str_data.get("categories", "")
            if categories_str:
                if categories_str.startswith("["):
                    # JSON format (legacy)
                    categories = json.loads(categories_str)
                else:
                    # TAG format (comma-separated)
                    categories = [
                        c.strip() for c in categories_str.split(",") if c.strip()
                    ]
            else:
                categories = []

            return MemoryRecord(
                id=str_data["id"],
                content=str_data["content"],
                scope=str_data["scope"],
                categories=categories,
                metadata=json.loads(str_data["metadata"]),
                importance=float(str_data["importance"]),
                created_at=datetime.fromisoformat(str_data["created_at"]),
                last_accessed=datetime.fromisoformat(str_data["last_accessed"]),
                embedding=embedding,
                source=str_data.get("source") or None,
                private=str_data.get("private", "false").lower() == "true",
            )
        except (KeyError, ValueError, TypeError) as e:
            # Try to get ID from data for error logging
            record_id = "unknown"
            try:
                if data:
                    # Try both bytes and str keys
                    id_value = data.get(b"id") if b"id" in data else data.get("id")  # type: ignore[call-overload]
                    if id_value:
                        record_id = (
                            id_value.decode("utf-8")
                            if isinstance(id_value, bytes)
                            else str(id_value)
                        )
            except Exception as id_error:
                _logger.debug(
                    f"Could not extract record ID for error logging: {id_error}"
                )
            _logger.error(f"Failed to deserialize record {record_id}: {e}")
            return None

    async def _ensure_vector_index(self) -> None:
        """Create Valkey Search vector index if it doesn't exist.

        Creates an index named 'memory_index' on record:* hashes with:
        - Vector field for embeddings (HNSW or FLAT algorithm)
        - TAG fields for scope and categories
        - NUMERIC fields for created_at and importance

        Raises:
            RuntimeError: If Valkey Search module is not available.
        """
        if self._index_created:
            return

        client = await self._get_client()

        try:
            # Check if index already exists
            existing = await ft.list(client)
            names = {
                i.decode("utf-8") if isinstance(i, bytes) else str(i)
                for i in (existing or [])
            }
            if "memory_index" in names:
                _logger.debug("Vector index 'memory_index' already exists")
                self._index_created = True
                return
        except Exception as e:
            _logger.debug("Could not list indexes, will attempt create: %s", e)

        try:
            # Build vector field attributes using the concrete subclass
            vector_attrs: VectorFieldAttributesHnsw | VectorFieldAttributesFlat
            if self._index_algorithm == "HNSW":
                algorithm = VectorAlgorithm.HNSW
                vector_attrs = VectorFieldAttributesHnsw(
                    dimensions=self._vector_dim,
                    distance_metric=DistanceMetricType.COSINE,
                    type=VectorType.FLOAT32,
                )
            else:
                algorithm = VectorAlgorithm.FLAT
                vector_attrs = VectorFieldAttributesFlat(
                    dimensions=self._vector_dim,
                    distance_metric=DistanceMetricType.COSINE,
                    type=VectorType.FLOAT32,
                )

            # Build schema
            schema: list[Field] = [
                VectorField("embedding", algorithm, vector_attrs),
                TagField("scope"),
                TagField("categories", separator=","),
                NumericField("created_at"),
                NumericField("importance"),
            ]

            # Create index using native ft.create
            options = FtCreateOptions(DataType.HASH, prefixes=["record:"])
            await ft.create(client, "memory_index", schema, options)

            _logger.info(
                "Created vector index 'memory_index' with %s algorithm (dim=%d)",
                self._index_algorithm,
                self._vector_dim,
            )
            self._index_created = True

        except Exception as e:
            error_msg = str(e).lower()
            if "unknown command" in error_msg or "ft.create" in error_msg:
                raise RuntimeError(
                    "Valkey Search module is not available. "
                    "Please ensure Valkey is running with the Search module loaded. "
                    "Use 'valkey/valkey-bundle:latest' Docker image or install the module separately."
                ) from e
            raise RuntimeError(f"Failed to create vector index: {e}") from e

    async def _update_indexes(
        self,
        record_id: str,
        scope: str,
        categories: list[str],
        metadata: dict[str, Any],
        timestamp: float,
    ) -> None:
        """Update all index structures for a record.

        Adds record ID to:
        - Scope sorted set with timestamp score
        - Category sets for all categories
        - Metadata index sets for all metadata key-value pairs

        Args:
            record_id: Unique identifier of the record.
            scope: Hierarchical scope path (e.g., "/agent/task").
            categories: List of category names.
            metadata: Dictionary of metadata key-value pairs.
            timestamp: Unix timestamp for scope index score.
        """
        client = await self._get_client()

        # Update scope index (sorted set with timestamp score)
        # Handle root scope "/" as special case
        scope_key = self._scope_key(scope)
        await client.zadd(scope_key, {record_id: timestamp})

        # Update category indexes (sets)
        for category in categories:
            category_key = self._category_key(category)
            await client.sadd(category_key, [record_id])

        # Update metadata indexes (sets for each key-value pair)
        for key, value in metadata.items():
            # Convert value to string for consistent key naming
            value_str = str(value)
            metadata_key = self._metadata_key(key, value_str)
            await client.sadd(metadata_key, [record_id])

    async def _remove_from_indexes(
        self,
        record_id: str,
        scope: str,
        categories: list[str],
        metadata: dict[str, Any],
    ) -> None:
        """Remove record from all index structures.

        Removes record ID from:
        - Scope sorted set
        - All category sets
        - All metadata index sets

        Args:
            record_id: Unique identifier of the record.
            scope: Hierarchical scope path.
            categories: List of category names.
            metadata: Dictionary of metadata key-value pairs.
        """
        client = await self._get_client()

        # Remove from scope index
        scope_key = self._scope_key(scope)
        await client.zrem(scope_key, [record_id])

        # Remove from category indexes
        for category in categories:
            category_key = self._category_key(category)
            await client.srem(category_key, [record_id])

        # Remove from metadata indexes
        for key, value in metadata.items():
            value_str = str(value)
            metadata_key = self._metadata_key(key, value_str)
            await client.srem(metadata_key, [record_id])

    async def asave(self, records: list[MemoryRecord]) -> None:
        """Save multiple records as a batch.

        Stores record fields in hash structure with key pattern "record:{id}".
        Stores embedding as binary vector field in record hash for Valkey Search auto-indexing.
        Updates scope sorted set, category sets, and metadata index sets.

        Note:
            Operations are issued as individual commands, not wrapped in
            MULTI/EXEC. Partial failures are possible under network errors.

        Args:
            records: List of memory records to save.

        Raises:
            ValueError: If serialization fails for any record.
            RuntimeError: If Valkey connection fails.
        """
        if not records:
            return

        client = await self._get_client()

        # Ensure vector index exists before saving
        await self._ensure_vector_index()

        # Build commands for atomic batch execution
        for record in records:
            record_key = self._record_key(record.id)

            # Convert record to hash fields (includes embedding as bytes)
            record_dict = self._record_to_dict(record)

            # Store record hash (Valkey Search will auto-index it)
            await client.hset(
                record_key,
                record_dict,  # type: ignore[arg-type]  # str keys are valid str|bytes
            )

            # Update all index structures
            timestamp = record.created_at.timestamp()
            await self._update_indexes(
                record.id,
                record.scope,
                record.categories,
                record.metadata,
                timestamp,
            )

    def save(self, records: list[MemoryRecord]) -> None:
        """Save multiple records atomically (sync wrapper).

        Args:
            records: List of memory records to save.

        Raises:
            ValueError: If serialization fails for any record.
            RuntimeError: If Valkey connection fails or called from async context.
        """
        self._run_async(self.asave(records))

    def get_record(self, record_id: str) -> MemoryRecord | None:
        """Retrieve record by ID.

        Fetches record hash from "record:{id}" key and deserializes all fields
        including datetime, JSON, and boolean values.

        Args:
            record_id: Unique identifier of the record to retrieve.

        Returns:
            MemoryRecord if found, None if record doesn't exist or deserialization fails.
        """
        result: MemoryRecord | None = self._run_async(self._aget_record(record_id))
        return result

    async def _aget_record(self, record_id: str) -> MemoryRecord | None:
        """Retrieve record by ID (async implementation).

        Args:
            record_id: Unique identifier of the record to retrieve.

        Returns:
            MemoryRecord if found, None if record doesn't exist or deserialization fails.
        """
        client = await self._get_client()
        record_key = self._record_key(record_id)

        try:
            # Fetch all fields from record hash
            data = await client.hgetall(record_key)

            if not data:
                # Record doesn't exist
                return None

            # Deserialize to MemoryRecord
            return self._dict_to_record(data)

        except Exception as e:
            _logger.error(f"Error retrieving record {record_id}: {e}")
            return None

    def update(self, record: MemoryRecord) -> None:
        """Update existing record or create new one.

        Preserves created_at timestamp from original record if it exists.
        Updates last_accessed timestamp to current time.
        Removes record from old indexes and adds to new indexes atomically.

        Args:
            record: Memory record to update.

        Raises:
            ValueError: If serialization fails.
            RuntimeError: If Valkey connection fails or called from async context.
        """
        self._run_async(self._aupdate(record))

    async def _aupdate(self, record: MemoryRecord) -> None:
        """Update existing record or create new one (async implementation).

        Args:
            record: Memory record to update.
        """
        client = await self._get_client()
        record_key = self._record_key(record.id)

        # Fetch existing record to preserve created_at and get old index values
        existing_data = await client.hgetall(record_key)

        if existing_data:
            # Convert bytes to strings for parsing (skip embedding which is binary)
            str_data: dict[str, str] = {}
            for key, value in existing_data.items():
                str_key = key.decode("utf-8") if isinstance(key, bytes) else key
                # Skip embedding field - it's binary data, not UTF-8
                if str_key == "embedding":
                    continue
                # Handle other binary fields gracefully
                if isinstance(value, bytes):
                    try:
                        str_value = value.decode("utf-8")
                    except UnicodeDecodeError:
                        continue  # Skip fields that can't be decoded
                else:
                    str_value = value
                str_data[str_key] = str_value

            # Preserve created_at from existing record
            try:
                original_created_at = datetime.fromisoformat(str_data["created_at"])
                record.created_at = original_created_at
            except (KeyError, ValueError) as e:
                _logger.warning(
                    f"Could not preserve created_at for record {record.id}: {e}"
                )

            # Update last_accessed to current time
            record.last_accessed = datetime.utcnow()

            # Parse old values for index cleanup
            try:
                old_scope = str_data.get("scope", "")
                # Handle both TAG format (comma-separated) and JSON format (legacy)
                categories_str = str_data.get("categories", "")
                if categories_str.startswith("["):
                    old_categories = json.loads(categories_str)
                else:
                    old_categories = [
                        c.strip() for c in categories_str.split(",") if c.strip()
                    ]
                old_metadata = json.loads(str_data.get("metadata", "{}"))
            except (json.JSONDecodeError, ValueError) as e:
                _logger.warning(
                    f"Could not parse old index values for record {record.id}: {e}"
                )
                old_scope = ""
                old_categories = []
                old_metadata = {}

            # Remove from old indexes
            await self._remove_from_indexes(
                record.id, old_scope, old_categories, old_metadata
            )

        # Convert record to hash fields
        record_dict = self._record_to_dict(record)

        # Store updated record hash
        await client.hset(
            record_key,
            record_dict,  # type: ignore[arg-type]  # str keys are valid str|bytes
        )

        # Add to new indexes
        timestamp = record.created_at.timestamp()
        await self._update_indexes(
            record.id, record.scope, record.categories, record.metadata, timestamp
        )

    async def adelete(
        self,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        """Delete records matching criteria.

        Supports deletion by record_ids, scope_prefix, categories, older_than, metadata_filter.
        Multiple criteria are combined with AND logic.

        Note:
            Operations are issued as individual commands, not wrapped in
            MULTI/EXEC. Partial failures are possible under network errors.

        Args:
            scope_prefix: Delete records in scope and subscopes.
            categories: Delete records matching any of these categories.
            record_ids: List of specific record IDs to delete.
            older_than: Delete records created before this datetime.
            metadata_filter: Delete records matching metadata key-value pairs.

        Returns:
            Count of deleted records.

        Raises:
            RuntimeError: If Valkey connection fails.
        """
        client = await self._get_client()

        # Step 1: Identify records to delete based on criteria
        ids_to_delete: set[str] = set()

        # Filter by record_ids
        if record_ids:
            ids_to_delete.update(record_ids)

        # Filter by scope_prefix
        if scope_prefix is not None:
            scope_ids = await self._find_records_by_scope(scope_prefix)
            if ids_to_delete:
                ids_to_delete &= set(scope_ids)  # AND logic
            else:
                ids_to_delete.update(scope_ids)

        # Filter by categories
        if categories:
            category_ids = await self._find_records_by_categories(categories)
            if ids_to_delete:
                ids_to_delete &= set(category_ids)  # AND logic
            else:
                ids_to_delete.update(category_ids)

        # Filter by older_than
        if older_than is not None:
            old_ids = await self._find_records_older_than(older_than)
            if ids_to_delete:
                ids_to_delete &= set(old_ids)  # AND logic
            else:
                ids_to_delete.update(old_ids)

        # Filter by metadata
        if metadata_filter:
            metadata_ids = await self._find_records_by_metadata(metadata_filter)
            if ids_to_delete:
                ids_to_delete &= set(metadata_ids)  # AND logic
            else:
                ids_to_delete.update(metadata_ids)

        # If no criteria specified, delete nothing
        if not ids_to_delete:
            return 0

        # Step 2: Fetch record data to identify which indexes to clean
        records_data = await self._fetch_records_for_deletion(list(ids_to_delete))

        # Step 3: Delete records and clean indexes
        for record_id, data in records_data.items():
            record_key = self._record_key(record_id)

            # Delete record hash (Valkey Search auto-removes from vector index)
            await client.delete([record_key])

            # Remove from all index structures
            await self._remove_from_indexes(
                record_id, data["scope"], data["categories"], data["metadata"]
            )

        return len(records_data)

    def delete(
        self,
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        record_ids: list[str] | None = None,
        older_than: datetime | None = None,
        metadata_filter: dict[str, Any] | None = None,
    ) -> int:
        """Delete records matching criteria (sync wrapper).

        Args:
            scope_prefix: Delete records in scope and subscopes.
            categories: Delete records matching any of these categories.
            record_ids: List of specific record IDs to delete.
            older_than: Delete records created before this datetime.
            metadata_filter: Delete records matching metadata key-value pairs.

        Returns:
            Count of deleted records.

        Raises:
            RuntimeError: If Valkey connection fails or called from async context.
        """
        result: int = self._run_async(
            self.adelete(
                scope_prefix=scope_prefix,
                categories=categories,
                record_ids=record_ids,
                older_than=older_than,
                metadata_filter=metadata_filter,
            )
        )
        return result

    async def _find_records_by_scope(self, scope_prefix: str) -> list[str]:
        """Find all record IDs in scope and subscopes.

        Args:
            scope_prefix: Scope path prefix to match.

        Returns:
            List of record IDs in matching scopes.
        """
        client = await self._get_client()
        record_ids: set[str] = set()

        # Scan for all scope keys
        cursor: str | bytes = "0"
        while True:
            result = await client.scan(cursor, match="scope:*", count=1000)
            cursor_new: str | bytes = result[0]  # type: ignore[assignment]
            keys: list[bytes] = result[1]  # type: ignore[assignment]

            for key_bytes in keys:
                # Extract scope path from key
                key_str = (
                    key_bytes.decode("utf-8")
                    if isinstance(key_bytes, bytes)
                    else key_bytes
                )
                scope_path = key_str.split(":", 1)[1] if ":" in key_str else ""

                # Check if scope matches prefix (boundary-safe)
                normalized_prefix = scope_prefix.rstrip("/") or "/"
                in_scope = (
                    normalized_prefix == "/"
                    or scope_path == normalized_prefix
                    or scope_path.startswith(f"{normalized_prefix}/")
                )
                if in_scope:
                    # Get all record IDs in this scope
                    scope_key = (
                        key_bytes.decode("utf-8")
                        if isinstance(key_bytes, bytes)
                        else key_bytes
                    )
                    members_result = await client.zrange(scope_key, RangeByIndex(0, -1))
                    # Convert bytes to strings
                    record_ids.update(
                        m.decode("utf-8") if isinstance(m, bytes) else str(m)
                        for m in members_result
                    )

            # Check if cursor is 0 (scan complete)
            cursor_str = (
                cursor_new.decode("utf-8")
                if isinstance(cursor_new, bytes)
                else cursor_new
            )
            if cursor_str == "0":
                break
            cursor = cursor_new

        return list(record_ids)

    async def _find_records_by_categories(self, categories: list[str]) -> list[str]:
        """Find all record IDs matching any of the categories.

        Args:
            categories: List of category names.

        Returns:
            List of record IDs with any of the categories.
        """
        client = await self._get_client()
        record_ids: set[str] = set()

        for category in categories:
            category_key = self._category_key(category)
            members = await client.smembers(category_key)
            # Convert bytes to strings
            str_members = [
                m.decode("utf-8") if isinstance(m, bytes) else m for m in members
            ]
            record_ids.update(str_members)

        return list(record_ids)

    async def _find_records_older_than(self, older_than: datetime) -> list[str]:
        """Find all record IDs created before the specified datetime.

        Args:
            older_than: Datetime threshold.

        Returns:
            List of record IDs created before older_than.
        """
        client = await self._get_client()
        record_ids: set[str] = set()
        threshold = older_than.timestamp()

        # Scan all scope keys and filter by timestamp
        cursor: str | bytes = "0"
        while True:
            result = await client.scan(cursor, match="scope:*", count=1000)
            cursor_new: str | bytes = result[0]  # type: ignore[assignment]
            keys: list[bytes] = result[1]  # type: ignore[assignment]

            for key_bytes in keys:
                # Get records with score (timestamp) less than threshold
                scope_key = (
                    key_bytes.decode("utf-8")
                    if isinstance(key_bytes, bytes)
                    else key_bytes
                )
                members_result = await client.zrange(
                    scope_key,
                    RangeByScore(
                        ScoreBoundary(0),
                        ScoreBoundary(threshold),
                    ),
                )
                # Convert bytes to strings
                record_ids.update(
                    m.decode("utf-8") if isinstance(m, bytes) else str(m)
                    for m in members_result
                )

            # Check if cursor is 0 (scan complete)
            cursor_str = (
                cursor_new.decode("utf-8")
                if isinstance(cursor_new, bytes)
                else cursor_new
            )
            if cursor_str == "0":
                break
            cursor = cursor_new

        return list(record_ids)

    async def _find_records_by_metadata(
        self, metadata_filter: dict[str, Any]
    ) -> list[str]:
        """Find all record IDs matching all metadata criteria (AND logic).

        Args:
            metadata_filter: Dictionary of metadata key-value pairs.

        Returns:
            List of record IDs matching all metadata criteria.
        """
        client = await self._get_client()

        # Get record IDs for each metadata criterion
        metadata_sets: list[set[str]] = []
        for key, value in metadata_filter.items():
            value_str = str(value)
            metadata_key = self._metadata_key(key, value_str)
            members = await client.smembers(metadata_key)
            # Convert bytes to strings
            str_members = {
                m.decode("utf-8") if isinstance(m, bytes) else m for m in members
            }
            metadata_sets.append(str_members)

        # Compute intersection (AND logic)
        if not metadata_sets:
            return []

        result = metadata_sets[0]
        for s in metadata_sets[1:]:
            result &= s

        return list(result)

    async def _fetch_records_for_deletion(
        self, record_ids: list[str]
    ) -> dict[str, dict[str, Any]]:
        """Fetch record data needed for index cleanup.

        Args:
            record_ids: List of record IDs to fetch.

        Returns:
            Dictionary mapping record ID to parsed record data.
        """
        client = await self._get_client()
        records_data: dict[str, dict[str, Any]] = {}

        for record_id in record_ids:
            record_key = self._record_key(record_id)
            data = await client.hgetall(record_key)

            if data:
                # Convert bytes to strings (skip embedding which is binary)
                str_data: dict[str, str] = {}
                for key, value in data.items():
                    str_key = key.decode("utf-8") if isinstance(key, bytes) else key
                    # Skip embedding field - it's binary
                    if str_key == "embedding":
                        continue
                    # Handle other binary fields gracefully
                    if isinstance(value, bytes):
                        try:
                            str_value = value.decode("utf-8")
                        except UnicodeDecodeError:
                            continue  # Skip fields that can't be decoded
                    else:
                        str_value = value
                    str_data[str_key] = str_value

                # Parse categories and metadata for index cleanup
                try:
                    # Parse categories — handle both TAG (comma-separated) and JSON format
                    categories_str = str_data.get("categories", "")
                    if categories_str and categories_str.startswith("["):
                        categories = json.loads(categories_str)
                    elif categories_str:
                        categories = [
                            c.strip() for c in categories_str.split(",") if c.strip()
                        ]
                    else:
                        categories = []

                    parsed_data = {
                        "scope": str_data.get("scope", ""),
                        "categories": categories,
                        "metadata": json.loads(str_data.get("metadata", "{}"))
                        if str_data.get("metadata")
                        else {},
                    }
                    records_data[record_id] = parsed_data
                except (json.JSONDecodeError, ValueError) as e:
                    _logger.warning(
                        f"Could not parse record {record_id} for deletion: {e}"
                    )
                    # Still delete the record, just skip index cleanup
                    records_data[record_id] = {
                        "scope": "",
                        "categories": [],
                        "metadata": {},
                    }

        return records_data

    async def _vector_search(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        """Perform server-side vector search using Valkey Search.

        Uses FT.SEARCH command with KNN query for vector similarity.
        Applies filters for scope, categories, and metadata in the same query.

        Args:
            query_embedding: Embedding vector for the query.
            scope_prefix: Optional scope path prefix to filter results.
            categories: Optional list of categories (OR logic).
            metadata_filter: Optional metadata key-value pairs (AND logic).
            limit: Maximum number of results to return.
            min_score: Minimum similarity score threshold (0.0 to 1.0).

        Returns:
            List of (MemoryRecord, score) tuples ordered by descending score.

        Raises:
            RuntimeError: If Valkey Search module is not available.
        """
        client = await self._get_client()

        # Ensure vector index exists
        await self._ensure_vector_index()

        # Build query components
        query_parts: list[str] = []

        # Scope prefix filter
        # Format: @scope:{prefix*}
        if scope_prefix:
            # Escape special characters in scope prefix
            escaped_scope = self._escape_search_query(scope_prefix)
            # For root scope "/", match everything
            if scope_prefix == "/":
                query_parts.append("*")
            else:
                query_parts.append(f"@scope:{{{escaped_scope}*}}")

        # Category filter (OR logic)
        # Format: @categories:{cat1|cat2|cat3}
        if categories:
            # Escape each category and join with |
            escaped_categories = [self._escape_search_query(cat) for cat in categories]
            cat_query = "|".join(escaped_categories)
            query_parts.append(f"@categories:{{{cat_query}}}")

        # Metadata filters are NOT included in the FT.SEARCH query because the
        # `memory_index` schema only defines a fixed set of fields
        # (embedding/scope/categories/created_at/importance). Referencing
        # arbitrary metadata keys via @{key}:{value} would cause FT.SEARCH to
        # error or silently return wrong results. They are applied as a
        # post-filter on the deserialized records below.

        # Combine filters
        filter_query = " ".join(query_parts) if query_parts else "*"

        # When a post-filter (metadata) will be applied, oversample the KNN
        # results so we don't under-return after filtering. Mirrors the
        # approach used by `LanceDBStorage`.
        knn_limit = limit * 3 if metadata_filter else limit

        # Build KNN query with filters
        # Format: (filter)=>[KNN limit @field $BLOB AS score]
        # Note: Don't wrap single "*" in parentheses
        if filter_query == "*":
            query = f"{filter_query}=>[KNN {knn_limit} @embedding $BLOB AS score]"
        else:
            query = f"({filter_query})=>[KNN {knn_limit} @embedding $BLOB AS score]"

        # Prepare embedding blob for PARAMS
        embedding_blob = self._embedding_to_bytes(query_embedding)

        # Build FT.SEARCH options
        # Note: Vector search results are sorted by distance ascending (nearest first).
        # We convert distance to similarity in _parse_search_result and re-sort descending.
        return_fields = [
            ReturnField(field_identifier="id"),
            ReturnField(field_identifier="content"),
            ReturnField(field_identifier="scope"),
            ReturnField(field_identifier="categories"),
            ReturnField(field_identifier="metadata"),
            ReturnField(field_identifier="importance"),
            ReturnField(field_identifier="created_at"),
            ReturnField(field_identifier="last_accessed"),
            ReturnField(field_identifier="source"),
            ReturnField(field_identifier="private"),
            ReturnField(field_identifier="score"),
        ]

        search_options = FtSearchOptions(
            return_fields=return_fields,
            params={"BLOB": embedding_blob},
            limit=FtSearchLimit(0, knn_limit),
        )

        try:
            # Execute native ft.search
            result = await ft.search(client, "memory_index", query, search_options)

            # Native ft.search returns: [count, {key1: {fields...}, key2: {fields...}}]
            if not result or not isinstance(result, list) or len(result) < 1:
                return []

            # First element is total count
            total_count_raw = result[0]
            if isinstance(total_count_raw, (int, str)):
                total_count = int(total_count_raw) if total_count_raw else 0
            else:
                total_count = 0
            if total_count == 0:
                return []

            # Parse documents from dict format
            records: list[tuple[MemoryRecord, float]] = []
            if len(result) > 1 and isinstance(result[1], dict):
                docs_dict = result[1]
                for doc_fields in docs_dict.values():
                    field_dict = self._normalize_field_dict(doc_fields)
                    parsed = self._parse_search_result(field_dict, min_score)
                    if parsed is not None:
                        records.append(parsed)

            # Sort by score descending (should already be sorted, but ensure)
            records.sort(key=lambda x: x[1], reverse=True)

            # Post-filter for scope boundary correctness.
            # The FT.SEARCH tag query uses prefix matching (@scope:{prefix*})
            # which can match siblings (e.g., /crew/a matches /crew/ab).
            # Apply strict boundary check here.
            if scope_prefix and scope_prefix != "/":
                normalized = scope_prefix.rstrip("/")
                records = [
                    (rec, score)
                    for rec, score in records
                    if rec.scope == normalized or rec.scope.startswith(f"{normalized}/")
                ]

            # Post-filter on metadata. The FT index does not materialize
            # arbitrary metadata fields, so we apply the filter in Python
            # against the deserialized records. Compare directly first, then
            # fall back to a string comparison so callers can pass either the
            # native value (e.g. 42) or its string form ("42").
            if metadata_filter:
                records = [
                    (rec, score)
                    for rec, score in records
                    if self._metadata_matches(rec.metadata, metadata_filter)
                ]

            # Trim to the originally requested limit after post-filtering.
            if len(records) > limit:
                records = records[:limit]

            return records

        except Exception as e:
            error_msg = str(e).lower()
            if "unknown command" in error_msg or "ft.search" in error_msg:
                raise RuntimeError(
                    "Valkey Search module is not available. "
                    "Please ensure Valkey is running with the Search module loaded."
                ) from e
            _logger.error(f"Vector search failed: {e}")
            raise

    # ------------------------------------------------------------------
    # Search result parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_field_dict(raw: dict[Any, Any]) -> dict[str, Any]:
        """Convert a raw field dict (possibly bytes keys/values) to str keys.

        Embedding values are kept as bytes; all other bytes values are decoded
        to UTF-8 (falling back to raw bytes on decode errors).
        """
        out: dict[str, Any] = {}
        for key, value in raw.items():
            str_key = key.decode("utf-8") if isinstance(key, bytes) else str(key)
            if isinstance(value, bytes):
                if str_key == "embedding":
                    out[str_key] = value
                else:
                    try:
                        out[str_key] = value.decode("utf-8")
                    except UnicodeDecodeError:
                        out[str_key] = value
            else:
                out[str_key] = value
        return out

    def _parse_search_result(
        self,
        field_dict: dict[str, Any],
        min_score: float,
    ) -> tuple[MemoryRecord, float] | None:
        """Extract score, apply min_score filter, and deserialize a search hit.

        Score is converted from cosine distance ([0, 2]) to similarity ([0, 1])
        and clamped to that range.

        Returns:
            (MemoryRecord, score) or None if filtered out or deserialization fails.
        """
        # Extract score — Valkey Search returns cosine distance
        score = 0.0
        for score_key in ("__score", "score"):
            if score_key in field_dict:
                distance = float(field_dict[score_key])
                score = max(0.0, min(1.0, 1.0 - (distance / 2.0)))
                break

        if score < min_score:
            return None

        record = self._dict_to_record(field_dict)
        if record is None:
            return None
        return (record, score)

    @staticmethod
    def _metadata_matches(
        record_metadata: dict[str, Any],
        metadata_filter: dict[str, Any],
    ) -> bool:
        """Check whether a record's metadata satisfies all filter pairs.

        Performs an equality check on each key; values are compared directly
        first, then as strings to allow callers to pass either the native
        value or its string form (matching the behaviour of the metadata
        index used by ``_find_records_by_metadata``).

        Args:
            record_metadata: Metadata dict from the stored record.
            metadata_filter: Key-value pairs that must all match (AND logic).

        Returns:
            True if every key in ``metadata_filter`` is present in
            ``record_metadata`` with an equal value, False otherwise.
        """
        for key, expected in metadata_filter.items():
            if key not in record_metadata:
                return False
            actual = record_metadata[key]
            if actual == expected:
                continue
            if str(actual) == str(expected):
                continue
            return False
        return True

    def _escape_search_query(self, text: str) -> str:
        """Escape special characters in Valkey Search query.

        Valkey Search uses special characters: , . < > { } [ ] " ' : ; ! @ # $ % ^ & * ( ) - + = ~ |

        Args:
            text: Text to escape.

        Returns:
            Escaped text safe for use in search queries.
        """
        # Characters that need escaping in Valkey Search queries.
        # Note: both '=' and '>' are escaped individually, so the KNN
        # clause delimiter '=>' becomes '\=\>' and cannot be injected.
        special_chars = r",.<>{}[]\"':;!@#$%^&*()-+=~|"
        for char in special_chars:
            text = text.replace(char, f"\\{char}")
        return text

    async def asearch(
        self,
        query_embedding: list[float],
        scope_prefix: str | None = None,
        categories: list[str] | None = None,
        metadata_filter: dict[str, Any] | None = None,
        limit: int = 10,
        min_score: float = 0.0,
    ) -> list[tuple[MemoryRecord, float]]:
        """Search for memories by vector similarity (async).

        Uses Valkey Search module for server-side vector similarity computation.
        Applies filters for scope, categories, and metadata in the same query.

        Args:
            query_embedding: Embedding vector for the query.
            scope_prefix: Optional scope path prefix to filter results.
            categories: Optional list of categories (OR logic).
            metadata_filter: Optional metadata key-value pairs (AND logic).
            limit: Maximum number of results to return.
            min_score: Minimum similarity score threshold (0.0 to 1.0).

        Returns:
            List of (MemoryRecord, score) tuples ordered by relevance (descending score).

        Raises:
            RuntimeError: If Valkey Search module is not available.
        """
        return await self._vector_search(
            query_embedding,
            scope_prefix,
            categories,
            metadata_filter,
            limit,
            min_score,
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
        """Search for memories by vector similarity (sync wrapper).

        Uses Valkey Search module for server-side vector similarity computation.
        Applies filters for scope, categories, and metadata in the same query.

        Args:
            query_embedding: Embedding vector for the query.
            scope_prefix: Optional scope path prefix to filter results.
            categories: Optional list of categories (OR logic).
            metadata_filter: Optional metadata key-value pairs (AND logic).
            limit: Maximum number of results to return.
            min_score: Minimum similarity score threshold (0.0 to 1.0).

        Returns:
            List of (MemoryRecord, score) tuples ordered by relevance (descending score).

        Raises:
            RuntimeError: If Valkey Search module is not available or called from async context.
        """
        result: list[tuple[MemoryRecord, float]] = self._run_async(
            self.asearch(
                query_embedding,
                scope_prefix,
                categories,
                metadata_filter,
                limit,
                min_score,
            )
        )
        return result

    def list_records(
        self,
        scope_prefix: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        """List records in a scope, newest first.

        Uses scope sorted set ZRANGE with REV flag for newest-first ordering.
        Supports scope_prefix filtering and pagination via limit and offset.

        Args:
            scope_prefix: Optional scope path prefix to filter by.
            limit: Maximum number of records to return (default 200).
            offset: Number of records to skip for pagination (default 0).

        Returns:
            List of MemoryRecord, ordered by created_at descending (newest first).
        """
        result: list[MemoryRecord] = self._run_async(
            self._alist_records(scope_prefix, limit, offset)
        )
        return result

    async def _alist_records(
        self,
        scope_prefix: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[MemoryRecord]:
        """List records in a scope, newest first (async implementation).

        Args:
            scope_prefix: Optional scope path prefix to filter by.
            limit: Maximum number of records to return.
            offset: Number of records to skip for pagination.

        Returns:
            List of MemoryRecord, ordered by created_at descending.
        """
        client = await self._get_client()

        # Find all record IDs in scope(s)
        if scope_prefix is not None:
            # Get records from matching scopes
            record_ids = await self._find_records_by_scope(scope_prefix)
        else:
            # Get all records from all scopes
            record_ids = []
            cursor: str | bytes = "0"
            while True:
                result = await client.scan(cursor, match="scope:*", count=1000)
                cursor_new: str | bytes = result[0]  # type: ignore[assignment]
                keys: list[bytes] = result[1]  # type: ignore[assignment]

                for key_bytes in keys:
                    # Get all record IDs in this scope
                    scope_key = (
                        key_bytes.decode("utf-8")
                        if isinstance(key_bytes, bytes)
                        else key_bytes
                    )
                    members_result = await client.zrange(scope_key, RangeByIndex(0, -1))
                    record_ids.extend(
                        m.decode("utf-8") if isinstance(m, bytes) else str(m)
                        for m in members_result
                    )

                # Check if cursor is 0 (scan complete)
                cursor_str = (
                    cursor_new.decode("utf-8")
                    if isinstance(cursor_new, bytes)
                    else cursor_new
                )
                if cursor_str == "0":
                    break
                cursor = cursor_new

        # Fetch records and sort by created_at descending
        records: list[MemoryRecord] = []
        for record_id in record_ids:
            record = await self._aget_record(record_id)
            if record:
                records.append(record)

        # Sort by created_at descending (newest first)
        records.sort(key=lambda r: r.created_at, reverse=True)

        # Apply pagination
        return records[offset : offset + limit]

    def get_scope_info(self, scope: str) -> ScopeInfo:
        """Get information about a scope.

        Counts records in scope and subscopes using sorted set cardinality.
        Extracts categories used within scope.
        Finds oldest and newest record timestamps.
        Lists immediate child scope paths.

        Args:
            scope: The scope path.

        Returns:
            ScopeInfo with record count, categories, date range, child scopes.
        """
        result: ScopeInfo = self._run_async(self._aget_scope_info(scope))
        return result

    async def _aget_scope_info(self, scope: str) -> ScopeInfo:
        """Get information about a scope (async implementation).

        Args:
            scope: The scope path.

        Returns:
            ScopeInfo with record count, categories, date range, child scopes.
        """
        # Normalize scope path
        scope = scope.rstrip("/") or "/"
        prefix = scope if scope != "/" else ""

        # Find all record IDs in scope and subscopes
        record_ids = await self._find_records_by_scope(prefix or "/")

        if not record_ids:
            return ScopeInfo(
                path=scope,
                record_count=0,
                categories=[],
                oldest_record=None,
                newest_record=None,
                child_scopes=[],
            )

        # Fetch records to extract categories and timestamps
        categories_set: set[str] = set()
        oldest: datetime | None = None
        newest: datetime | None = None

        for record_id in record_ids:
            record = await self._aget_record(record_id)
            if record:
                # Collect categories
                categories_set.update(record.categories)

                # Track oldest and newest timestamps
                if oldest is None or record.created_at < oldest:
                    oldest = record.created_at
                if newest is None or record.created_at > newest:
                    newest = record.created_at

        # Find immediate child scopes
        child_scopes = await self._alist_scopes(scope)

        return ScopeInfo(
            path=scope,
            record_count=len(record_ids),
            categories=sorted(categories_set),
            oldest_record=oldest,
            newest_record=newest,
            child_scopes=child_scopes,
        )

    def list_scopes(self, parent: str = "/") -> list[str]:
        """List immediate child scopes under a parent path.

        Defaults to root scope "/" when no parent specified.
        Parses scope paths from scope sorted set keys.
        Returns only immediate children, not grandchildren.

        Args:
            parent: Parent scope path (default root "/").

        Returns:
            List of immediate child scope paths in sorted order.
        """
        result: list[str] = self._run_async(self._alist_scopes(parent))
        return result

    async def _alist_scopes(self, parent: str = "/") -> list[str]:
        """List immediate child scopes under a parent path (async implementation).

        Args:
            parent: Parent scope path (default root "/").

        Returns:
            List of immediate child scope paths in sorted order.
        """
        client = await self._get_client()

        # Normalize parent path
        parent = parent.rstrip("/") or ""
        prefix = (parent + "/") if parent else "/"

        # Scan for all scope keys
        children: set[str] = set()
        cursor: str | bytes = "0"
        while True:
            result = await client.scan(cursor, match="scope:*", count=1000)
            cursor_new: str | bytes = result[0]  # type: ignore[assignment]
            keys: list[bytes] = result[1]  # type: ignore[assignment]

            for key_bytes in keys:
                # Extract scope path from key
                key_str = (
                    key_bytes.decode("utf-8")
                    if isinstance(key_bytes, bytes)
                    else key_bytes
                )
                scope_path = key_str.split(":", 1)[1] if ":" in key_str else ""

                # Check if scope is a child of parent
                if scope_path.startswith(prefix) and scope_path != (
                    prefix.rstrip("/") or "/"
                ):
                    # Extract the immediate child component
                    rest = scope_path[len(prefix) :]
                    first_component = rest.split("/", 1)[0]
                    if first_component:
                        child_path = prefix + first_component
                        children.add(child_path)

            # Check if cursor is 0 (scan complete)
            cursor_str = (
                cursor_new.decode("utf-8")
                if isinstance(cursor_new, bytes)
                else cursor_new
            )
            if cursor_str == "0":
                break
            cursor = cursor_new

        return sorted(children)

    def list_categories(self, scope_prefix: str | None = None) -> dict[str, int]:
        """List categories and their counts within a scope.

        Supports filtering by scope_prefix.
        Computes counts by measuring category set cardinality.
        Returns global category counts when scope_prefix is None.

        Args:
            scope_prefix: Optional scope to limit to (None = global).

        Returns:
            Mapping of category name to record count.
        """
        result: dict[str, int] = self._run_async(self._alist_categories(scope_prefix))
        return result

    async def _alist_categories(
        self, scope_prefix: str | None = None
    ) -> dict[str, int]:
        """List categories and their counts within a scope (async implementation).

        Args:
            scope_prefix: Optional scope to limit to (None = global).

        Returns:
            Mapping of category name to record count.
        """
        client = await self._get_client()

        if scope_prefix is not None:
            # Get records in scope and count their categories
            record_ids = await self._find_records_by_scope(scope_prefix)
            counts: dict[str, int] = {}

            for record_id in record_ids:
                record = await self._aget_record(record_id)
                if record:
                    for category in record.categories:
                        counts[category] = counts.get(category, 0) + 1

            return counts
        # Global category counts - scan all category sets
        counts = {}
        cursor: str | bytes = "0"
        while True:
            result = await client.scan(cursor, match="category:*", count=1000)
            cursor_new: str | bytes = result[0]  # type: ignore[assignment]
            keys: list[bytes] = result[1]  # type: ignore[assignment]

            for key_bytes in keys:
                # Extract category name from key
                key_str = (
                    key_bytes.decode("utf-8")
                    if isinstance(key_bytes, bytes)
                    else key_bytes
                )
                category_name = key_str.split(":", 1)[1] if ":" in key_str else ""

                if category_name:
                    # Get cardinality of category set
                    category_key = self._category_key(category_name)
                    count = await client.scard(category_key)
                    counts[category_name] = int(count) if count else 0

            # Check if cursor is 0 (scan complete)
            cursor_str = (
                cursor_new.decode("utf-8")
                if isinstance(cursor_new, bytes)
                else cursor_new
            )
            if cursor_str == "0":
                break
            cursor = cursor_new

        return counts

    def count(self, scope_prefix: str | None = None) -> int:
        """Count records in scope (and subscopes).

        Uses scope sorted set cardinality for efficient counting.
        Supports scope_prefix filtering.
        Returns total count across all scopes when scope_prefix is None.

        Args:
            scope_prefix: Optional scope path (None = all).

        Returns:
            Number of records.
        """
        result: int = self._run_async(self._acount(scope_prefix))
        return result

    async def _acount(self, scope_prefix: str | None = None) -> int:
        """Count records in scope (and subscopes) (async implementation).

        Args:
            scope_prefix: Optional scope path (None = all).

        Returns:
            Number of records.
        """
        if scope_prefix is None or scope_prefix.strip("/") == "":
            # Count all records across all scopes
            record_ids = await self._find_records_by_scope("/")
            return len(set(record_ids))  # Use set to deduplicate
        # Count records in specific scope and subscopes
        record_ids = await self._find_records_by_scope(scope_prefix)
        return len(set(record_ids))  # Use set to deduplicate

    def reset(self, scope_prefix: str | None = None) -> None:
        """Reset (delete all) memories in scope.

        Deletes all records in scope and subscopes when scope_prefix provided.
        Deletes all records across all scopes when scope_prefix is None.
        Removes all index structures atomically.

        Args:
            scope_prefix: Optional scope path (None = reset all).
        """
        self._run_async(self._areset(scope_prefix))

    async def _areset(self, scope_prefix: str | None = None) -> None:
        """Reset (delete all) memories in scope (async implementation).

        Args:
            scope_prefix: Optional scope path (None = reset all).
        """
        # Use delete with scope_prefix to remove all records
        await self.adelete(scope_prefix=scope_prefix)
