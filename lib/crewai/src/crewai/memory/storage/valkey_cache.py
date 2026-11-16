"""Valkey-based cache implementation for CrewAI.

This module provides a simple cache interface using Valkey-GLIDE client
for caching operations with optional TTL support. It replaces Redis usage
in A2A communication, file uploads, and agent card caching.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from glide import GlideClient, GlideClientConfiguration, NodeAddress

from crewai.memory.storage.valkey_io_loop import on_io_loop


_logger = logging.getLogger(__name__)


class ValkeyCache:
    """Simple cache interface using Valkey-GLIDE client.

    Provides get/set/delete/exists operations for caching with optional TTL.
    Uses JSON serialization for complex values and lazy client initialization.

    Example:
        >>> cache = ValkeyCache(host="localhost", port=6379)
        >>> await cache.set("key", {"data": "value"}, ttl=3600)
        >>> value = await cache.get("key")
        >>> await cache.delete("key")
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        username: str | None = None,
        password: str | None = None,
        default_ttl: int | None = None,
        use_tls: bool = False,
    ) -> None:
        """Initialize Valkey cache.

        Args:
            host: Valkey server hostname.
            port: Valkey server port.
            db: Database number to use.
            username: Optional ACL username for authentication.
            password: Optional password for authentication.
            default_ttl: Default TTL in seconds (None = no expiration).
            use_tls: Enable TLS/SSL encryption for connections (rediss/valkeys).
        """
        self._host = host
        self._port = port
        self._db = db
        self._username = username
        self._password = password
        self._default_ttl = default_ttl
        self._use_tls = use_tls
        # The client and its creation lock are only ever touched on the shared
        # Valkey I/O loop (see valkey_io_loop), so they never need rebinding
        # when callers come from different event loops or threads.
        self._client: GlideClient | None = None
        self._client_lock: asyncio.Lock | None = None

    @on_io_loop
    async def _get_client(self) -> GlideClient:
        """Get or create the Valkey client (lazy initialization).

        Runs on the Valkey I/O loop. A double-check lock keeps concurrent
        callers from creating more than one client.

        Returns:
            Initialized GlideClient instance.

        Raises:
            RuntimeError: If connection to Valkey fails.
            TimeoutError: If connection attempt times out (10 seconds).
        """
        if self._client_lock is None:
            self._client_lock = asyncio.Lock()

        if self._client is None:
            async with self._client_lock:
                if self._client is None:
                    host = self._host
                    port = self._port
                    db = self._db
                    try:
                        from glide import ServerCredentials

                        config = GlideClientConfiguration(
                            addresses=[NodeAddress(host, port)],
                            client_name="crewai_valkey",
                            # Valkey library-name tag for CLIENT INFO attribution.
                            # Independent of client_name (CLIENT SETNAME); this sets
                            # the LIB-NAME suffix (GlidePy(crewai)) so operators can
                            # attribute Valkey usage to CrewAI. Metadata only.
                            client_info_tag="crewai",
                            use_tls=self._use_tls,
                            database_id=db,
                            credentials=(
                                ServerCredentials(
                                    username=self._username or None,
                                    password=self._password or "",
                                )
                                if (self._password or self._username)
                                else None
                            ),
                        )

                        # Add connection timeout (10 seconds)
                        try:
                            self._client = await asyncio.wait_for(
                                GlideClient.create(config), timeout=10.0
                            )
                        except asyncio.TimeoutError as e:
                            _logger.error("Connection timeout connecting to Valkey")
                            raise TimeoutError(
                                "Connection timeout to Valkey. "
                                "Ensure Valkey is running and accessible."
                            ) from e

                        _logger.info("Valkey cache client initialized")
                    except (TimeoutError, RuntimeError):
                        raise
                    except Exception as e:
                        _logger.error(
                            "Failed to create Valkey cache client: %s",
                            type(e).__name__,
                        )
                        raise RuntimeError(
                            "Cannot connect to Valkey. Check connection settings."
                        ) from e

        return self._client

    @on_io_loop
    async def get(self, key: str) -> Any | None:
        """Get value from cache.

        Args:
            key: Cache key.

        Returns:
            Cached value (deserialized from JSON) or None if not found.
        """
        client = await self._get_client()
        value = await client.get(key)

        if value is None:
            return None

        try:
            return json.loads(value)
        except json.JSONDecodeError:
            _logger.warning(f"Failed to deserialize cached value for key: {key}")
            return None

    @on_io_loop
    async def set(
        self,
        key: str,
        value: Any,
        ttl: int | None = None,
    ) -> None:
        """Set value in cache.

        Args:
            key: Cache key.
            value: Value to cache (will be serialized to JSON).
            ttl: TTL in seconds (None uses default_ttl, 0 = no expiration).

        Raises:
            TypeError: If value is not JSON-serializable.
            ValueError: If ttl is negative.
        """
        from glide import ExpirySet, ExpiryType

        # Validate input before opening a connection
        try:
            serialized = json.dumps(value)
        except (TypeError, ValueError) as e:
            _logger.error("Cannot serialize value for key %r: %s", key, e)
            raise TypeError(
                f"Value for cache key {key!r} is not JSON-serializable: {e}"
            ) from e

        ttl_to_use = ttl if ttl is not None else self._default_ttl
        if ttl_to_use is not None and ttl_to_use < 0:
            raise ValueError("TTL must be >= 0")

        client = await self._get_client()

        if ttl_to_use and ttl_to_use > 0:
            # Set with expiration using SET command with EX option
            await client.set(
                key,
                serialized,
                expiry=ExpirySet(ExpiryType.SEC, ttl_to_use),
            )
        else:
            await client.set(key, serialized)

    @on_io_loop
    async def delete(self, key: str) -> None:
        """Delete value from cache.

        Args:
            key: Cache key to delete.
        """
        client = await self._get_client()
        await client.delete([key])

    @on_io_loop
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache.

        Args:
            key: Cache key to check.

        Returns:
            True if key exists, False otherwise.
        """
        client = await self._get_client()
        result = await client.exists([key])
        return result > 0

    @on_io_loop
    async def clear(self, prefix: str = "") -> int:
        """Delete all keys matching a prefix (server-side, not just local state).

        Scans the keyspace so entries written by other processes or left from a
        previous run are also removed.

        Args:
            prefix: Key prefix to match. Empty string matches all keys.

        Returns:
            Number of keys deleted.
        """
        client = await self._get_client()
        match = f"{prefix}*" if prefix else "*"
        deleted = 0
        cursor: str | bytes = "0"
        while True:
            result = await client.scan(cursor, match=match, count=1000)
            cursor = result[0]
            keys = result[1]
            if keys:
                await client.delete(
                    [k.decode("utf-8") if isinstance(k, bytes) else k for k in keys]
                )
                deleted += len(keys)
            cursor_str = (
                cursor.decode("utf-8") if isinstance(cursor, bytes) else str(cursor)
            )
            if cursor_str == "0":
                break
        return deleted

    @on_io_loop
    async def close(self) -> None:
        """Close Valkey client connection."""
        if self._client:
            await self._client.close()
            self._client = None
            _logger.debug("Valkey cache client closed")
