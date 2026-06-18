"""Tests for pluggable cache backends."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from crewai.agents.cache.cache_backend import CacheBackend
from crewai.agents.cache.cache_handler import CacheHandler
from crewai.agents.cache.in_memory_backend import InMemoryCacheBackend
from crewai.agents.cache.sqlite_backend import SQLiteCacheBackend


# -------------------------------------------------------------------
# Protocol conformance
# -------------------------------------------------------------------


class TestProtocolConformance:
    def test_in_memory_is_cache_backend(self):
        assert isinstance(InMemoryCacheBackend(), CacheBackend)

    def test_sqlite_is_cache_backend(self, tmp_path: Path):
        backend = SQLiteCacheBackend(str(tmp_path / "test.db"))
        assert isinstance(backend, CacheBackend)


# -------------------------------------------------------------------
# InMemoryCacheBackend
# -------------------------------------------------------------------


class TestInMemoryCacheBackend:
    def test_get_missing_key(self):
        backend = InMemoryCacheBackend()
        assert backend.get("no-such-key") is None

    def test_set_and_get(self):
        backend = InMemoryCacheBackend()
        backend.set("k", "v")
        assert backend.get("k") == "v"

    def test_set_overwrites(self):
        backend = InMemoryCacheBackend()
        backend.set("k", "v1")
        backend.set("k", "v2")
        assert backend.get("k") == "v2"

    def test_claim_if_absent_claims(self):
        backend = InMemoryCacheBackend()
        claimed, existing = backend.claim_if_absent("k", "sentinel")
        assert claimed is True
        assert existing is None
        assert backend.get("k") == "sentinel"

    def test_claim_if_absent_returns_existing(self):
        backend = InMemoryCacheBackend()
        backend.set("k", "real-value")
        claimed, existing = backend.claim_if_absent("k", "sentinel")
        assert claimed is False
        assert existing == "real-value"


# -------------------------------------------------------------------
# SQLiteCacheBackend
# -------------------------------------------------------------------


class TestSQLiteCacheBackend:
    @pytest.fixture()
    def backend(self, tmp_path: Path) -> SQLiteCacheBackend:
        return SQLiteCacheBackend(str(tmp_path / "test_cache.db"))

    def test_get_missing_key(self, backend: SQLiteCacheBackend):
        assert backend.get("no-such-key") is None

    def test_set_and_get(self, backend: SQLiteCacheBackend):
        backend.set("k", {"result": 42})
        assert backend.get("k") == {"result": 42}

    def test_set_overwrites(self, backend: SQLiteCacheBackend):
        backend.set("k", "v1")
        backend.set("k", "v2")
        assert backend.get("k") == "v2"

    def test_claim_if_absent_claims(self, backend: SQLiteCacheBackend):
        claimed, existing = backend.claim_if_absent("k", {"sentinel": True})
        assert claimed is True
        assert existing is None
        assert backend.get("k") == {"sentinel": True}

    def test_claim_if_absent_returns_existing(self, backend: SQLiteCacheBackend):
        backend.set("k", "real-value")
        claimed, existing = backend.claim_if_absent("k", {"sentinel": True})
        assert claimed is False
        assert existing == "real-value"

    def test_cross_instance_visibility(self, tmp_path: Path):
        """Two SQLiteCacheBackend instances sharing the same DB should see each other's writes."""
        db = str(tmp_path / "shared.db")
        a = SQLiteCacheBackend(db)
        b = SQLiteCacheBackend(db)
        a.set("key", "from-a")
        assert b.get("key") == "from-a"

    def test_cross_instance_claim(self, tmp_path: Path):
        db = str(tmp_path / "shared.db")
        a = SQLiteCacheBackend(db)
        b = SQLiteCacheBackend(db)
        claimed_a, _ = a.claim_if_absent("key", {"owner": "a"})
        claimed_b, existing = b.claim_if_absent("key", {"owner": "b"})
        assert claimed_a is True
        assert claimed_b is False
        assert existing == {"owner": "a"}


# -------------------------------------------------------------------
# CacheHandler with pluggable backend
# -------------------------------------------------------------------


class TestCacheHandlerWithBackend:
    def test_default_backend_is_in_memory(self):
        handler = CacheHandler()
        assert isinstance(handler._backend, InMemoryCacheBackend)

    def test_custom_backend_is_used(self, tmp_path: Path):
        backend = SQLiteCacheBackend(str(tmp_path / "test.db"))
        handler = CacheHandler(backend=backend)
        handler.add("tool", '{"arg": 1}', "result-1")
        assert handler.read("tool", '{"arg": 1}') == "result-1"

    def test_claim_if_absent_delegates(self):
        handler = CacheHandler()
        claimed, _ = handler.claim_if_absent("tool", "input", "sentinel")
        assert claimed is True
        claimed2, existing = handler.claim_if_absent("tool", "input", "sentinel2")
        assert claimed2 is False
        assert existing == "sentinel"
