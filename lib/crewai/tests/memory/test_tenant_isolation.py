"""Isolation invariant tests for per-tenant memory.

The contract:
    A recall scoped to tenant A NEVER returns a row written by tenant B.

If any test in this file fails or passes vacuously (e.g. because the
embeddings happen to differ), the per-tenant isolation feature is broken.

Tests in this file are split into three groups:

* ``TestScopedStorage`` -- exercise ScopedStorage directly. The wrapper is
  the enforcement chokepoint and these tests are the primary security
  contract.
* ``TestMemoryBackCompat`` -- pin that single-tenant deployments (no
  tenant_id passed anywhere) keep working unchanged via the '_default'
  fallback.
* ``TestMemoryIsolation`` -- exercise Memory.remember / Memory.recall /
  Memory.forget with explicit tenant_id kwargs end to end. These prove
  isolation works through the full public API, including the deep-recall
  LLM exploration path that goes through RecallFlow.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.memory.storage.scoped_storage import ScopedStorage
from crewai.memory.types import MemoryRecord


@pytest.fixture
def lance_path(tmp_path: Path) -> Path:
    return tmp_path / "isolation.lance"


@pytest.fixture
def lance_storage(lance_path: Path) -> LanceDBStorage:
    return LanceDBStorage(path=str(lance_path), vector_dim=4)


@pytest.fixture
def mock_embedder() -> MagicMock:
    """Embedder that returns DIFFERENT embeddings per text, never identical.

    A naive ``abs(hash(t)) % N`` mapping buckets distinct texts into N
    collisions, which would let two different inputs produce identical
    vectors and make the isolation assertions in this file pass vacuously
    (the foreign-tenant row would be filtered by tenant predicate, not by
    semantic distance). Using a SHA-256 digest of the text gives 2^32 worth
    of distinct values across four float buckets while staying deterministic
    across test runs.
    """
    import hashlib

    m = MagicMock()

    def embed(texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            digest = hashlib.sha256(t.encode("utf-8")).digest()
            # 4 buckets of 4 bytes each -> 4 floats in [0, 1).
            vec = [
                int.from_bytes(digest[i * 4 : (i + 1) * 4], "big") / 2**32
                for i in range(4)
            ]
            out.append(vec)
        return out

    m.side_effect = embed
    return m


# ----------------------------------------------------------------------
# ScopedStorage direct tests -- the enforcement primitive itself.
# These pass as of PR #3.
# ----------------------------------------------------------------------


class TestScopedStorage:
    def test_stamp_on_write_overrides_default_tenant(
        self, lance_storage: LanceDBStorage
    ) -> None:
        # A record with the implicit default tenant arrives at a ScopedStorage
        # bound to "alice"; it should be stamped as "alice" before persisting.
        scoped = ScopedStorage(lance_storage, tenant_id="alice")
        scoped.save([
            MemoryRecord(content="alice note", scope="/", embedding=[0.1] * 4)
        ])
        records = lance_storage.list_records(tenant_id="alice")
        assert len(records) == 1
        assert records[0].tenant_id == "alice"
        # And the default tenant sees nothing.
        assert lance_storage.list_records(tenant_id="_default") == []

    def test_save_rejects_cross_tenant_record(
        self, lance_storage: LanceDBStorage
    ) -> None:
        scoped = ScopedStorage(lance_storage, tenant_id="alice")
        bad = MemoryRecord(
            content="trojan", tenant_id="bob", scope="/", embedding=[0.1] * 4
        )
        with pytest.raises(PermissionError, match="alice.*bob|bob.*alice"):
            scoped.save([bad])

    def test_cross_tenant_search_returns_nothing(
        self, lance_storage: LanceDBStorage
    ) -> None:
        # Two tenants save near-identical content with the same embedding.
        # Alice's recall must never see Bob's row even though semantic
        # similarity would otherwise rank it first.
        alice = ScopedStorage(lance_storage, tenant_id="alice")
        bob = ScopedStorage(lance_storage, tenant_id="bob")

        embedding = [0.5, 0.5, 0.5, 0.5]
        alice.save([
            MemoryRecord(
                content="API key is alice-secret-123",
                scope="/credentials",
                embedding=embedding,
            )
        ])
        bob.save([
            MemoryRecord(
                content="API key is bob-secret-456",
                scope="/credentials",
                embedding=embedding,
            )
        ])

        alice_hits = alice.search(embedding, limit=10)
        bob_hits = bob.search(embedding, limit=10)

        assert len(alice_hits) == 1
        assert all(r.tenant_id == "alice" for r, _ in alice_hits)
        assert "alice-secret" in alice_hits[0][0].content
        assert not any("bob-secret" in r.content for r, _ in alice_hits)

        assert len(bob_hits) == 1
        assert all(r.tenant_id == "bob" for r, _ in bob_hits)
        assert "bob-secret" in bob_hits[0][0].content
        assert not any("alice-secret" in r.content for r, _ in bob_hits)

    def test_get_record_cross_tenant_returns_none(
        self, lance_storage: LanceDBStorage
    ) -> None:
        alice = ScopedStorage(lance_storage, tenant_id="alice")
        bob = ScopedStorage(lance_storage, tenant_id="bob")
        alice.save([
            MemoryRecord(
                id="rec-1", content="alice note", scope="/", embedding=[0.1] * 4
            )
        ])
        # Bob asking for Alice's record by id sees nothing.
        assert bob.get_record("rec-1") is None
        # Alice still sees her own.
        assert alice.get_record("rec-1") is not None

    def test_delete_is_scoped(self, lance_storage: LanceDBStorage) -> None:
        alice = ScopedStorage(lance_storage, tenant_id="alice")
        bob = ScopedStorage(lance_storage, tenant_id="bob")
        alice.save([
            MemoryRecord(content="alice note", scope="/", embedding=[0.1] * 4)
        ])
        bob.save([
            MemoryRecord(content="bob note", scope="/", embedding=[0.1] * 4)
        ])

        # Alice deletes her entire tenant; Bob's row must survive.
        deleted = alice.delete()
        assert deleted == 1
        bob_rows = bob.list_records()
        assert len(bob_rows) == 1
        assert "bob note" in bob_rows[0].content

    def test_reset_is_scoped(self, lance_storage: LanceDBStorage) -> None:
        alice = ScopedStorage(lance_storage, tenant_id="alice")
        bob = ScopedStorage(lance_storage, tenant_id="bob")
        alice.save([
            MemoryRecord(content="alice", scope="/", embedding=[0.1] * 4)
        ])
        bob.save([
            MemoryRecord(content="bob", scope="/", embedding=[0.1] * 4)
        ])
        alice.reset()
        assert alice.count() == 0
        assert bob.count() == 1

    def test_backend_leak_is_loud(self, lance_storage: LanceDBStorage) -> None:
        """If the backend filter ever leaks a foreign-tenant row, ScopedStorage
        raises RuntimeError instead of quietly filtering. Loud over silent.
        """
        alice = ScopedStorage(lance_storage, tenant_id="alice")
        bad = MemoryRecord(
            content="leak",
            tenant_id="bob",
            scope="/",
            embedding=[0.1] * 4,
        )
        # Monkeypatch the inner search to return a foreign-tenant row.
        alice._inner.search = MagicMock(return_value=[(bad, 0.99)])  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="cross-tenant"):
            alice.search([0.1] * 4)

    def test_user_id_filter_within_tenant(
        self, lance_storage: LanceDBStorage
    ) -> None:
        # Tenant 'acme' has two users; an instance bound to user_id='alice'
        # only sees alice's rows.
        alice = ScopedStorage(lance_storage, tenant_id="acme", user_id="alice")
        bob = ScopedStorage(lance_storage, tenant_id="acme", user_id="bob")
        alice.save([
            MemoryRecord(content="alice preferences", scope="/", embedding=[0.1] * 4)
        ])
        bob.save([
            MemoryRecord(content="bob preferences", scope="/", embedding=[0.1] * 4)
        ])

        alice_rows = alice.list_records()
        bob_rows = bob.list_records()
        assert len(alice_rows) == 1
        assert "alice" in alice_rows[0].content
        assert len(bob_rows) == 1
        assert "bob" in bob_rows[0].content

        # The tenant-admin view (no user_id) sees both.
        admin = ScopedStorage(lance_storage, tenant_id="acme")
        admin_rows = admin.list_records()
        assert len(admin_rows) == 2

    def test_constructor_rejects_empty_tenant(
        self, lance_storage: LanceDBStorage
    ) -> None:
        with pytest.raises(ValueError, match="tenant_id"):
            ScopedStorage(lance_storage, tenant_id="")


# ----------------------------------------------------------------------
# Memory-level isolation tests.
#
# These are XFAIL'd in PR #3 because Memory does not yet route through
# ScopedStorage -- every internal storage call hardcodes
# tenant_id="_default". PR #4 wires the resolved tenant through, and the
# XFAIL markers are removed there. The intent of having the failing tests
# in this PR is to keep the security contract visible in the test suite
# from the moment the wrapper lands.
# ----------------------------------------------------------------------


class TestMemoryBackCompat:
    """Single-tenant deployments (no tenant_id passed anywhere) must keep
    working unchanged. This already passes today via the '_default' fallback.
    """

    def test_default_tenant_backcompat(
        self, tmp_path: Path, mock_embedder: MagicMock
    ) -> None:
        from crewai.memory.unified_memory import Memory

        m = Memory(
            storage=str(tmp_path / "mem.lance"),
            llm=MagicMock(),
            embedder=mock_embedder,
        )
        m.remember("the meeting is at 3pm", scope="/")
        hits = m.recall("when is the meeting", depth="shallow")
        assert hits
        assert all(h.record.tenant_id == "_default" for h in hits)


class TestMemoryIsolation:
    def test_cross_tenant_recall_returns_nothing(
        self, tmp_path: Path, mock_embedder: MagicMock
    ) -> None:
        from crewai.memory.unified_memory import Memory

        m = Memory(
            storage=str(tmp_path / "mem.lance"),
            llm=MagicMock(),
            embedder=mock_embedder,
        )
        m.remember(
            "API key is alice-secret-123",
            tenant_id="alice",
            scope="/credentials",
        )
        m.remember(
            "API key is bob-secret-456",
            tenant_id="bob",
            scope="/credentials",
        )

        alice = m.recall("what is my api key", tenant_id="alice", depth="shallow")
        bob = m.recall("what is my api key", tenant_id="bob", depth="shallow")

        assert all(h.record.tenant_id == "alice" for h in alice)
        assert all(h.record.tenant_id == "bob" for h in bob)
        assert any("alice-secret" in h.record.content for h in alice)
        assert any("bob-secret" in h.record.content for h in bob)
        assert not any("bob-secret" in h.record.content for h in alice)
        assert not any("alice-secret" in h.record.content for h in bob)

    def test_forget_is_scoped(
        self, tmp_path: Path, mock_embedder: MagicMock
    ) -> None:
        from crewai.memory.unified_memory import Memory

        m = Memory(
            storage=str(tmp_path / "mem.lance"),
            llm=MagicMock(),
            embedder=mock_embedder,
        )
        m.remember("alice note", tenant_id="alice", scope="/")
        m.remember("bob note", tenant_id="bob", scope="/")
        deleted = m.forget(tenant_id="alice")
        assert deleted == 1
        bob_hits = m.recall("note", tenant_id="bob", depth="shallow")
        assert any("bob note" in h.record.content for h in bob_hits)

    def test_deep_recall_honors_tenant(
        self, tmp_path: Path, mock_embedder: MagicMock
    ) -> None:
        """depth='deep' goes through RecallFlow with LLM-driven exploration.

        The design doc claims RecallFlow is safe by construction because it
        holds a ScopedStorage, not the raw backend. This test exercises that
        path end to end so the claim has receipts. Without this test, every
        recall in this class would be the depth='shallow' path and the
        deep-recall code would be uncovered for isolation.
        """
        from crewai.memory.unified_memory import Memory

        # An LLM mock that always says "fall through to direct search" -- we
        # don't need the LLM's analysis to validate isolation; we need the
        # RecallFlow path to execute at all.
        llm = MagicMock()
        llm.call.return_value = "{}"

        m = Memory(
            storage=str(tmp_path / "mem.lance"),
            llm=llm,
            embedder=mock_embedder,
        )
        m.remember(
            "alice's birthday is March 4",
            tenant_id="alice",
            scope="/personal",
            categories=["birthday"],
            importance=0.5,
        )
        m.remember(
            "bob's birthday is March 4",
            tenant_id="bob",
            scope="/personal",
            categories=["birthday"],
            importance=0.5,
        )

        alice = m.recall("when is the birthday", tenant_id="alice", depth="deep")
        bob = m.recall("when is the birthday", tenant_id="bob", depth="deep")

        assert all(h.record.tenant_id == "alice" for h in alice)
        assert all(h.record.tenant_id == "bob" for h in bob)
        assert not any("bob" in h.record.content for h in alice)
        assert not any("alice" in h.record.content for h in bob)

    def test_instance_default_tenant_holds(
        self, tmp_path: Path, mock_embedder: MagicMock
    ) -> None:
        # Memory configured with a default tenant should auto-scope every call.
        from crewai.memory.unified_memory import Memory

        alice_mem = Memory(
            storage=str(tmp_path / "mem.lance"),
            llm=MagicMock(),
            embedder=mock_embedder,
            tenant_id="alice",
        )
        bob_mem = Memory(
            storage=str(tmp_path / "mem.lance"),
            llm=MagicMock(),
            embedder=mock_embedder,
            tenant_id="bob",
        )
        alice_mem.remember("alice's calendar", scope="/")
        bob_mem.remember("bob's calendar", scope="/")

        alice = alice_mem.recall("calendar", depth="shallow")
        bob = bob_mem.recall("calendar", depth="shallow")
        assert all(h.record.tenant_id == "alice" for h in alice)
        assert all(h.record.tenant_id == "bob" for h in bob)
        assert not any("bob" in h.record.content for h in alice)
        assert not any("alice" in h.record.content for h in bob)
