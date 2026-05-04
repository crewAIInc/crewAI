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
