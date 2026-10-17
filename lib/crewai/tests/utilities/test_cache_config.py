"""Tests for shared cache configuration helpers."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from crewai.utilities.cache_config import (
    get_aiocache_config,
    parse_cache_url,
    use_valkey_cache,
)


class TestParseCacheUrl:
    """Tests for parse_cache_url()."""

    def test_returns_none_when_no_env_vars(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert parse_cache_url() is None

    def test_parses_valkey_url(self) -> None:
        with patch.dict(
            os.environ, {"VALKEY_URL": "redis://myhost:6380/2"}, clear=True
        ):
            result = parse_cache_url()
            assert result is not None
            assert result["host"] == "myhost"
            assert result["port"] == 6380
            assert result["db"] == 2
            assert result["password"] is None

    def test_parses_redis_url(self) -> None:
        with patch.dict(
            os.environ, {"REDIS_URL": "redis://localhost:6379/0"}, clear=True
        ):
            result = parse_cache_url()
            assert result is not None
            assert result["host"] == "localhost"
            assert result["port"] == 6379
            assert result["db"] == 0

    def test_valkey_url_takes_priority_over_redis_url(self) -> None:
        with patch.dict(
            os.environ,
            {
                "VALKEY_URL": "redis://valkey-host:6380/1",
                "REDIS_URL": "redis://redis-host:6379/0",
            },
            clear=True,
        ):
            result = parse_cache_url()
            assert result is not None
            assert result["host"] == "valkey-host"
            assert result["port"] == 6380

    def test_parses_password(self) -> None:
        with patch.dict(
            os.environ,
            {"VALKEY_URL": "redis://:s3cret@myhost:6379/0"},
            clear=True,
        ):
            result = parse_cache_url()
            assert result is not None
            assert result["password"] == "s3cret"

    def test_defaults_for_minimal_url(self) -> None:
        with patch.dict(
            os.environ, {"VALKEY_URL": "redis://myhost"}, clear=True
        ):
            result = parse_cache_url()
            assert result is not None
            assert result["host"] == "myhost"
            assert result["port"] == 6379
            assert result["db"] == 0
            assert result["password"] is None

    def test_non_numeric_db_path_defaults_to_zero(self) -> None:
        with patch.dict(
            os.environ, {"VALKEY_URL": "redis://myhost:6379/mydb"}, clear=True
        ):
            result = parse_cache_url()
            assert result is not None
            assert result["db"] == 0


class TestGetAiocacheConfig:
    """Tests for get_aiocache_config()."""

    def test_returns_memory_cache_when_no_url(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = get_aiocache_config()
            assert config["default"]["cache"] == "aiocache.SimpleMemoryCache"

    def test_returns_redis_cache_when_url_set(self) -> None:
        with patch.dict(
            os.environ, {"VALKEY_URL": "redis://myhost:6380/2"}, clear=True
        ):
            config = get_aiocache_config()
            assert config["default"]["cache"] == "aiocache.RedisCache"
            assert config["default"]["endpoint"] == "myhost"
            assert config["default"]["port"] == 6380
            assert config["default"]["db"] == 2

    def test_forwards_tls_for_rediss_scheme(self) -> None:
        with patch.dict(
            os.environ, {"VALKEY_URL": "rediss://myhost:6380/2"}, clear=True
        ):
            config = get_aiocache_config()
            assert config["default"]["ssl"] is True

    def test_no_ssl_flag_for_plaintext_scheme(self) -> None:
        with patch.dict(
            os.environ, {"VALKEY_URL": "redis://myhost:6380/2"}, clear=True
        ):
            config = get_aiocache_config()
            assert "ssl" not in config["default"]

    def test_username_routed_through_connection_pool_kwargs(self) -> None:
        # aiocache's RedisCache forwards unknown top-level keys to
        # BaseCache.__init__, which rejects "username". The username must be
        # nested under connection_pool_kwargs so it reaches the redis pool.
        with patch.dict(
            os.environ,
            {"VALKEY_URL": "redis://acluser:pw@myhost:6379/0"},
            clear=True,
        ):
            config = get_aiocache_config()
            default = config["default"]
            assert "username" not in default
            assert default["connection_pool_kwargs"] == {"username": "acluser"}

    def test_password_only_url_omits_username_kwargs(self) -> None:
        with patch.dict(
            os.environ, {"VALKEY_URL": "redis://:pw@myhost:6379/0"}, clear=True
        ):
            config = get_aiocache_config()
            default = config["default"]
            assert "username" not in default
            assert "connection_pool_kwargs" not in default
            assert default["password"] == "pw"

    def test_username_config_constructs_without_typeerror(self) -> None:
        # Regression: a URL with userinfo used to inject a top-level "username"
        # key that crashed RedisCache construction with a TypeError.
        aiocache = pytest.importorskip("aiocache")
        with patch.dict(
            os.environ,
            {"VALKEY_URL": "redis://acluser:pw@localhost:6379/0"},
            clear=True,
        ):
            config = get_aiocache_config()
        aiocache.caches.set_config(config)
        cache = aiocache.caches.get("default")  # must not raise
        pool_kwargs = cache.client.connection_pool.connection_kwargs
        assert pool_kwargs.get("username") == "acluser"
        assert pool_kwargs.get("password") == "pw"


class TestParseCacheUrlUsername:
    """Username normalization in parse_cache_url()."""

    def test_password_only_url_normalizes_empty_username_to_none(self) -> None:
        with patch.dict(
            os.environ, {"VALKEY_URL": "redis://:pw@myhost:6379/0"}, clear=True
        ):
            result = parse_cache_url()
            assert result is not None
            # Empty string would authenticate as a blank ACL user under GLIDE.
            assert result["username"] is None
            assert result["password"] == "pw"

    def test_acl_username_preserved(self) -> None:
        with patch.dict(
            os.environ,
            {"VALKEY_URL": "redis://acluser:pw@myhost:6379/0"},
            clear=True,
        ):
            result = parse_cache_url()
            assert result is not None
            assert result["username"] == "acluser"


class TestUseValkeyCache:
    """Tests for use_valkey_cache()."""

    def test_returns_false_when_not_set(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            assert use_valkey_cache() is False

    def test_returns_true_when_set(self) -> None:
        with patch.dict(
            os.environ, {"VALKEY_URL": "redis://localhost:6379"}, clear=True
        ):
            assert use_valkey_cache() is True

    def test_returns_false_when_only_redis_url_set(self) -> None:
        with patch.dict(
            os.environ, {"REDIS_URL": "redis://localhost:6379"}, clear=True
        ):
            assert use_valkey_cache() is False


class TestCacheUrlUsername:
    """ACL username in VALKEY_URL/REDIS_URL must be parsed and forwarded."""

    def test_parse_cache_url_reads_username(self) -> None:
        with patch.dict(
            os.environ, {"VALKEY_URL": "redis://alice:s3cret@host:6379/0"}, clear=True
        ):
            conn = parse_cache_url()
            assert conn is not None
            assert conn["username"] == "alice"
            assert conn["password"] == "s3cret"

    def test_get_aiocache_config_forwards_username(self) -> None:
        with patch.dict(
            os.environ, {"VALKEY_URL": "redis://alice:s3cret@host:6379/0"}, clear=True
        ):
            config = get_aiocache_config()
            # Nested under connection_pool_kwargs, not a top-level key, because
            # aiocache's BaseCache rejects an unknown "username" argument.
            assert "username" not in config["default"]
            assert config["default"]["connection_pool_kwargs"] == {"username": "alice"}

    def test_no_username_key_when_absent(self) -> None:
        with patch.dict(
            os.environ, {"VALKEY_URL": "redis://:s3cret@host:6379/0"}, clear=True
        ):
            config = get_aiocache_config()
            assert "username" not in config["default"]
