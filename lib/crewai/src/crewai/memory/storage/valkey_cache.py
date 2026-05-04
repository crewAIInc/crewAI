"""Valkey-based cache implementation for CrewAI.

This module provides a simple cache interface using Valkey-GLIDE client
for caching operations with optional TTL support. It replaces Redis usage
in A2A communication, file uploads, and agent card caching.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from glide import GlideClient, GlideClientConfiguration, NodeAddress


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
        password: str | None = None,
        default_ttl: int | None = None,
    ) -> None:
        """Initialize Valkey cache.

        Args:
            host: Valkey server hostname.
            port: Valkey server port.
            db: Database number to use.
            password: Optional password for authentication.
            default_ttl: Default TTL in seconds (None = no expiration).
        """
        self._host = host
        self._port = port
        self._db = db
        self._password = password
        self._default_ttl = default_ttl
        self._client: GlideClient | None = None

    async def _get_client(self) -> GlideClient:
        """Get or create Valkey client (lazy initialization).

        Returns:
            Initialized GlideClient instance.

        Raises:
            RuntimeError: If connection to Valkey fails.
            TimeoutError: If connection attempt times out (10 seconds).
        """
        import asyncio

        if self._client is None:
            host = self._host
            port = self._port
            db = self._db
            try:
                from glide import ServerCredentials

                config = GlideClientConfiguration(
                    addresses=[NodeAddress(host, port)],
                    database_id=db,
                    credentials=(
                        ServerCredentials(password=self._password)
                        if self._password
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
                    "Failed to create Valkey cache client: %s", type(e).__name__
                )
                raise RuntimeError(
                    "Cannot connect to Valkey. Check connection settings."
                ) from e

        return self._client

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
        """
        from glide import ExpirySet, ExpiryType

        client = await self._get_client()
        try:
            serialized = json.dumps(value)
        except (TypeError, ValueError) as e:
            _logger.error("Cannot serialize value for key %r: %s", key, e)
            raise TypeError(
                f"Value for cache key {key!r} is not JSON-serializable: {e}"
            ) from e

        ttl_to_use = ttl if ttl is not None else self._default_ttl

        if ttl_to_use and ttl_to_use > 0:
            # Set with expiration using SET command with EX option
            await client.set(
                key,
                serialized,
                expiry=ExpirySet(ExpiryType.SEC, ttl_to_use),
            )
        else:
            await client.set(key, serialized)

    async def delete(self, key: str) -> None:
        """Delete value from cache.

        Args:
            key: Cache key to delete.
        """
        client = await self._get_client()
        await client.delete([key])

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

    async def close(self) -> None:
        """Close Valkey client connection."""
        if self._client:
            await self._client.close()
            self._client = None
            _logger.debug("Valkey cache client closed")
