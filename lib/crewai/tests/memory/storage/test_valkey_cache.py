"""Tests for ValkeyCache implementation."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai.memory.storage.valkey_cache import ValkeyCache


@pytest.fixture
def mock_glide_client() -> AsyncMock:
    """Create a mock GlideClient for testing."""
    client = AsyncMock()
    client.get = AsyncMock()
    client.set = AsyncMock()
    client.delete = AsyncMock()
    client.exists = AsyncMock()
    client.close = AsyncMock()
    return client


@pytest.fixture
def valkey_cache(mock_glide_client: AsyncMock) -> ValkeyCache:
    """Create a ValkeyCache instance with mocked client."""
    cache = ValkeyCache(host="localhost", port=6379, db=0)

    # Mock the client creation to return our mock
    async def mock_create_client() -> AsyncMock:
        cache._client = mock_glide_client
        return mock_glide_client

    cache._get_client = mock_create_client  # type: ignore[method-assign]
    return cache


class TestValkeyCacheBasicOperations:
    """Tests for basic ValkeyCache operations (get/set/delete/exists)."""

    @pytest.mark.asyncio
    async def test_set_and_get_string_value(
        self, valkey_cache: ValkeyCache, mock_glide_client: AsyncMock
    ) -> None:
        """Test setting and getting a string value."""
        # Mock get to return serialized string
        mock_glide_client.get.return_value = json.dumps("test_value")

        # Set value
        await valkey_cache.set("test_key", "test_value")

        # Verify set was called
        mock_glide_client.set.assert_called_once()
        call_args = mock_glide_client.set.call_args
        assert call_args[0][0] == "test_key"
        assert call_args[0][1] == json.dumps("test_value")

        # Get value
        result = await valkey_cache.get("test_key")

        # Verify get was called and result is correct
        mock_glide_client.get.assert_called_once_with("test_key")
        assert result == "test_value"

    @pytest.mark.asyncio
    async def test_set_and_get_dict_value(
        self, valkey_cache: ValkeyCache, mock_glide_client: AsyncMock
    ) -> None:
        """Test setting and getting a dictionary value."""
        test_dict = {"key1": "value1", "key2": 42, "key3": [1, 2, 3]}
        mock_glide_client.get.return_value = json.dumps(test_dict)

        # Set value
        await valkey_cache.set("dict_key", test_dict)

        # Verify set was called with serialized dict
        mock_glide_client.set.assert_called_once()
        call_args = mock_glide_client.set.call_args
        assert call_args[0][0] == "dict_key"
        assert call_args[0][1] == json.dumps(test_dict)

        # Get value
        result = await valkey_cache.get("dict_key")

        # Verify result matches original dict
        assert result == test_dict

    @pytest.mark.asyncio
    async def test_set_and_get_list_value(
        self, valkey_cache: ValkeyCache, mock_glide_client: AsyncMock
    ) -> None:
        """Test setting and getting a list value."""
        test_list = [1, "two", 3.0, {"nested": "dict"}]
        mock_glide_client.get.return_value = json.dumps(test_list)

        await valkey_cache.set("list_key", test_list)
        result = await valkey_cache.get("list_key")

        assert result == test_list

    @pytest.mark.asyncio
    async def test_get_nonexistent_key_returns_none(
        self, valkey_cache: ValkeyCache, mock_glide_client: AsyncMock
    ) -> None:
        """Test getting a non-existent key returns None."""
        mock_glide_client.get.return_value = None

        result = await valkey_cache.get("nonexistent_key")

        assert result is None
        mock_glide_client.get.assert_called_once_with("nonexistent_key")

    @pytest.mark.asyncio
    async def test_delete_key(
        self, valkey_cache: ValkeyCache, mock_glide_client: AsyncMock
    ) -> None:
        """Test deleting a key."""
        await valkey_cache.delete("test_key")

        mock_glide_client.delete.assert_called_once_with(["test_key"])

    @pytest.mark.asyncio
    async def test_exists_returns_true_for_existing_key(
        self, valkey_cache: ValkeyCache, mock_glide_client: AsyncMock
    ) -> None:
        """Test exists returns True for existing key."""
        mock_glide_client.exists.return_value = 1

        result = await valkey_cache.exists("existing_key")

        assert result is True
        mock_glide_client.exists.assert_called_once_with(["existing_key"])

    @pytest.mark.asyncio
    async def test_exists_returns_false_for_nonexistent_key(
        self, valkey_cache: ValkeyCache, mock_glide_client: AsyncMock
    ) -> None:
        """Test exists returns False for non-existent key."""
        mock_glide_client.exists.return_value = 0

        result = await valkey_cache.exists("nonexistent_key")

        assert result is False
        mock_glide_client.exists.assert_called_once_with(["nonexistent_key"])


class TestValkeyCacheTTL:
    """Tests for ValkeyCache TTL functionality."""

    @pytest.mark.asyncio
    async def test_set_with_explicit_ttl(
        self, valkey_cache: ValkeyCache, mock_glide_client: AsyncMock
    ) -> None:
        """Test setting a value with explicit TTL."""
        await valkey_cache.set("ttl_key", "value", ttl=3600)

        # Verify set was called with expiry
        mock_glide_client.set.assert_called_once()
        call_args = mock_glide_client.set.call_args
        assert call_args[0][0] == "ttl_key"
        assert call_args[0][1] == json.dumps("value")
        assert "expiry" in call_args[1]

    @pytest.mark.asyncio
    async def test_set_with_default_ttl(
        self, mock_glide_client: AsyncMock
    ) -> None:
        """Test setting a value with default TTL from constructor."""
        cache = ValkeyCache(host="localhost", port=6379, default_ttl=1800)

        async def mock_create_client() -> AsyncMock:
            cache._client = mock_glide_client
            return mock_glide_client

        cache._get_client = mock_create_client  # type: ignore[method-assign]

        await cache.set("default_ttl_key", "value")

        # Verify set was called with default TTL
        mock_glide_client.set.assert_called_once()
        call_args = mock_glide_client.set.call_args
        assert "expiry" in call_args[1]

    @pytest.mark.asyncio
    async def test_set_without_ttl(
        self, valkey_cache: ValkeyCache, mock_glide_client: AsyncMock
    ) -> None:
        """Test setting a value without TTL (no expiration)."""
        await valkey_cache.set("no_ttl_key", "value")

        # Verify set was called without expiry
        mock_glide_client.set.assert_called_once()
        call_args = mock_glide_client.set.call_args
        assert call_args[0][0] == "no_ttl_key"
        assert call_args[0][1] == json.dumps("value")
        # Should not have expiry parameter
        assert "expiry" not in call_args[1] or call_args[1].get("expiry") is None

    @pytest.mark.asyncio
    async def test_set_with_zero_ttl_no_expiration(
        self, valkey_cache: ValkeyCache, mock_glide_client: AsyncMock
    ) -> None:
        """Test setting a value with TTL=0 means no expiration."""
        await valkey_cache.set("zero_ttl_key", "value", ttl=0)

        # Verify set was called without expiry
        mock_glide_client.set.assert_called_once()
        call_args = mock_glide_client.set.call_args
        assert "expiry" not in call_args[1] or call_args[1].get("expiry") is None

    @pytest.mark.asyncio
    async def test_explicit_ttl_overrides_default(
        self, mock_glide_client: AsyncMock
    ) -> None:
        """Test explicit TTL overrides default TTL."""
        cache = ValkeyCache(host="localhost", port=6379, default_ttl=1800)

        async def mock_create_client() -> AsyncMock:
            cache._client = mock_glide_client
            return mock_glide_client

        cache._get_client = mock_create_client  # type: ignore[method-assign]

        await cache.set("override_key", "value", ttl=7200)

        # Verify set was called with explicit TTL (7200), not default (1800)
        mock_glide_client.set.assert_called_once()
        call_args = mock_glide_client.set.call_args
        assert "expiry" in call_args[1]


class TestValkeyCacheJSONSerialization:
    """Tests for ValkeyCache JSON serialization edge cases."""

    @pytest.mark.asyncio
    async def test_serialize_none_value(
        self, valkey_cache: ValkeyCache, mock_glide_client: AsyncMock
    ) -> None:
        """Test serializing None value."""
        mock_glide_client.get.return_value = json.dumps(None)

        await valkey_cache.set("none_key", None)
        result = await valkey_cache.get("none_key")

        assert result is None

    @pytest.mark.asyncio
    async def test_serialize_boolean_values(
        self, valkey_cache: ValkeyCache, mock_glide_client: AsyncMock
    ) -> None:
        """Test serializing boolean values."""
        mock_glide_client.get.side_effect = [
            json.dumps(True),
            json.dumps(False),
        ]

        await valkey_cache.set("true_key", True)
        await valkey_cache.set("false_key", False)

        result_true = await valkey_cache.get("true_key")
        result_false = await valkey_cache.get("false_key")

        assert result_true is True
        assert result_false is False

    @pytest.mark.asyncio
    async def test_serialize_numeric_values(
        self, valkey_cache: ValkeyCache, mock_glide_client: AsyncMock
    ) -> None:
        """Test serializing numeric values (int, float)."""
        mock_glide_client.get.side_effect = [
            json.dumps(42),
            json.dumps(3.14159),
            json.dumps(0),
            json.dumps(-100),
        ]

        await valkey_cache.set("int_key", 42)
        await valkey_cache.set("float_key", 3.14159)
        await valkey_cache.set("zero_key", 0)
        await valkey_cache.set("negative_key", -100)

        assert await valkey_cache.get("int_key") == 42
        assert await valkey_cache.get("float_key") == 3.14159
        assert await valkey_cache.get("zero_key") == 0
        assert await valkey_cache.get("negative_key") == -100

    @pytest.mark.asyncio
    async def test_serialize_empty_collections(
        self, valkey_cache: ValkeyCache, mock_glide_client: AsyncMock
    ) -> None:
        """Test serializing empty collections."""
        mock_glide_client.get.side_effect = [
            json.dumps([]),
            json.dumps({}),
            json.dumps(""),
        ]

        await valkey_cache.set("empty_list", [])
        await valkey_cache.set("empty_dict", {})
        await valkey_cache.set("empty_string", "")

        assert await valkey_cache.get("empty_list") == []
        assert await valkey_cache.get("empty_dict") == {}
        assert await valkey_cache.get("empty_string") == ""

    @pytest.mark.asyncio
    async def test_serialize_nested_structures(
        self, valkey_cache: ValkeyCache, mock_glide_client: AsyncMock
    ) -> None:
        """Test serializing deeply nested structures."""
        nested_data = {
            "level1": {
                "level2": {
                    "level3": [1, 2, {"level4": "deep"}]
                }
            },
            "list": [{"a": 1}, {"b": 2}]
        }
        mock_glide_client.get.return_value = json.dumps(nested_data)

        await valkey_cache.set("nested_key", nested_data)
        result = await valkey_cache.get("nested_key")

        assert result == nested_data

    @pytest.mark.asyncio
    async def test_deserialize_invalid_json_returns_none(
        self, valkey_cache: ValkeyCache, mock_glide_client: AsyncMock
    ) -> None:
        """Test deserializing invalid JSON returns None and logs warning."""
        mock_glide_client.get.return_value = "invalid json {{"

        with patch("crewai.memory.storage.valkey_cache._logger") as mock_logger:
            result = await valkey_cache.get("invalid_key")

            assert result is None
            mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_serialize_unicode_strings(
        self, valkey_cache: ValkeyCache, mock_glide_client: AsyncMock
    ) -> None:
        """Test serializing unicode strings."""
        unicode_data = "Hello 世界 🌍 Привет"
        mock_glide_client.get.return_value = json.dumps(unicode_data)

        await valkey_cache.set("unicode_key", unicode_data)
        result = await valkey_cache.get("unicode_key")

        assert result == unicode_data


class TestValkeyCacheConnectionManagement:
    """Tests for ValkeyCache connection management."""

    @pytest.mark.asyncio
    async def test_lazy_client_initialization(self) -> None:
        """Test client is initialized lazily on first use."""
        cache = ValkeyCache(host="localhost", port=6379)

        # Client should be None initially
        assert cache._client is None

        # Mock GlideClient.create
        with patch("crewai.memory.storage.valkey_cache.GlideClient") as mock_glide:
            mock_client = AsyncMock()
            mock_glide.create = AsyncMock(return_value=mock_client)
            mock_client.get = AsyncMock(return_value=None)

            # First operation should initialize client
            await cache.get("test_key")

            # Client should now be initialized
            assert cache._client is not None
            mock_glide.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_client_reuse_across_operations(
        self, valkey_cache: ValkeyCache, mock_glide_client: AsyncMock
    ) -> None:
        """Test client is reused across multiple operations."""
        mock_glide_client.get.return_value = json.dumps("value")
        mock_glide_client.exists.return_value = 1

        # Perform multiple operations
        await valkey_cache.get("key1")
        await valkey_cache.set("key2", "value2")
        await valkey_cache.exists("key3")
        await valkey_cache.delete("key4")

        # _get_client should return the same client instance
        client1 = await valkey_cache._get_client()
        client2 = await valkey_cache._get_client()
        assert client1 is client2

    @pytest.mark.asyncio
    async def test_close_connection(
        self, valkey_cache: ValkeyCache, mock_glide_client: AsyncMock
    ) -> None:
        """Test closing the client connection."""
        # Initialize client
        await valkey_cache._get_client()
        assert valkey_cache._client is not None

        # Close connection
        await valkey_cache.close()

        # Verify close was called and client is None
        mock_glide_client.close.assert_called_once()
        assert valkey_cache._client is None

    @pytest.mark.asyncio
    async def test_connection_error_raises_runtime_error(self) -> None:
        """Test connection error raises RuntimeError with descriptive message."""
        cache = ValkeyCache(host="invalid-host", port=9999)

        with patch("crewai.memory.storage.valkey_cache.GlideClient") as mock_glide:
            mock_glide.create = AsyncMock(side_effect=Exception("Connection refused"))

            with pytest.raises(RuntimeError) as exc_info:
                await cache._get_client()

            assert "Cannot connect to Valkey" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_authentication_with_password(self) -> None:
        """Test client initialization with password authentication."""
        cache = ValkeyCache(
            host="localhost",
            port=6379,
            password="secret_password"
        )

        with patch("crewai.memory.storage.valkey_cache.GlideClient") as mock_glide:
            mock_client = AsyncMock()
            mock_glide.create = AsyncMock(return_value=mock_client)

            await cache._get_client()

            # Verify GlideClient.create was called with credentials
            mock_glide.create.assert_called_once()
            config = mock_glide.create.call_args[0][0]
            assert hasattr(config, "credentials")


class TestValkeyCacheEdgeCases:
    """Tests for ValkeyCache edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_set_with_special_characters_in_key(
        self, valkey_cache: ValkeyCache, mock_glide_client: AsyncMock
    ) -> None:
        """Test setting values with special characters in key."""
        special_keys = [
            "key:with:colons",
            "key/with/slashes",
            "key-with-dashes",
            "key_with_underscores",
            "key.with.dots",
        ]

        for key in special_keys:
            await valkey_cache.set(key, "value")
            mock_glide_client.set.assert_called()

    @pytest.mark.asyncio
    async def test_large_value_serialization(
        self, valkey_cache: ValkeyCache, mock_glide_client: AsyncMock
    ) -> None:
        """Test serializing large values."""
        large_list = list(range(10000))
        mock_glide_client.get.return_value = json.dumps(large_list)

        await valkey_cache.set("large_key", large_list)
        result = await valkey_cache.get("large_key")

        assert result == large_list

    @pytest.mark.asyncio
    async def test_concurrent_operations(
        self, valkey_cache: ValkeyCache, mock_glide_client: AsyncMock
    ) -> None:
        """Test concurrent cache operations."""
        import asyncio

        mock_glide_client.get.return_value = json.dumps("value")

        # Perform concurrent operations
        tasks = [
            valkey_cache.set(f"key{i}", f"value{i}")
            for i in range(10)
        ]
        await asyncio.gather(*tasks)

        # Verify all operations completed
        assert mock_glide_client.set.call_count == 10
