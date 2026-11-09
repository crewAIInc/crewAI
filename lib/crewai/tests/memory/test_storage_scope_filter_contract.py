"""Shared StorageBackend contract tests for scope boundaries and filter combinations.

Every test runs against each memory storage backend with fixed vectors, so no
embedder is called. A fixed corpus spans nested scopes, sibling scopes (one of
which shares a string prefix with its neighbour), a root-scope record, two
categories, metadata, and two ages. Each test asserts both what a search or
delete reaches and which control records it leaves untouched.

Known backend gaps that already have an open issue are marked as strict
per-backend xfails, so the marker must be removed when the referenced fix lands.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta
import importlib.util
from pathlib import Path
from typing import Any
from uuid import uuid4

from crewai.memory.storage.backend import StorageBackend
from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.memory.types import MemoryRecord
import pytest


VECTOR_DIM = 4
QUERY_VECTOR = [1.0, 1.0, 1.0, 1.0]
SEARCH_LIMIT = 50
OLD = datetime(2026, 1, 1, 12, 0, 0)
CUTOFF = OLD + timedelta(days=1)
NEW = OLD + timedelta(days=2)

LANCEDB = pytest.param("lancedb", id="lancedb")
QDRANT_EDGE = pytest.param(
    "qdrant_edge",
    id="qdrant_edge",
    marks=pytest.mark.skipif(
        importlib.util.find_spec("qdrant_edge") is None,
        reason="qdrant-edge-py not installed",
    ),
)

LANCEDB_ROOT_SCOPE_DELETE = (
    "LanceDB scoped delete also removes root-scope records "
    "(#8404, #8867; fixes proposed in #8405, #8870)"
)
LANCEDB_STRING_PREFIX_SCOPE = (
    "LanceDB scope filter is a string prefix, so /team/alpha matches "
    "/team/alphabet (#7543, #8867; fixes proposed in #8559, #8870)"
)
LANCEDB_RECORD_IDS_COMBINED = (
    "LanceDB delete drops record_ids when combined with categories/metadata "
    "and drops older_than when combined with record_ids (#7419)"
)

# (scope, categories, metadata, created_at). Keys name each record's role.
CORPUS_SPEC: dict[str, tuple[str, list[str], dict[str, Any], datetime]] = {
    "alpha_old_decision": (
        "/team/alpha",
        ["decision"],
        {"status": "open", "owner": "ana"},
        OLD,
    ),
    "alpha_new_note": (
        "/team/alpha",
        ["note"],
        {"status": "open", "owner": "ana"},
        NEW,
    ),
    "alpha_child_decision": (
        "/team/alpha/sub",
        ["decision", "finance"],
        {"status": "closed", "owner": "ana"},
        NEW,
    ),
    "alphabet_decision": (
        "/team/alphabet",
        ["decision"],
        {"status": "open", "owner": "bo"},
        OLD,
    ),
    "beta_old_decision": (
        "/team/beta",
        ["decision"],
        {"status": "open", "owner": "bo"},
        OLD,
    ),
    "beta_new_note": (
        "/team/beta",
        ["note"],
        {"status": "closed", "owner": "bo"},
        NEW,
    ),
    "root_decision": ("/", ["decision"], {"status": "open"}, OLD),
}
ALL_KEYS = frozenset(CORPUS_SPEC)


@pytest.fixture(params=[LANCEDB, QDRANT_EDGE])
def backend(request: pytest.FixtureRequest) -> str:
    """Name of the storage backend under test."""
    return str(request.param)


@pytest.fixture
def storage(backend: str, tmp_path: Path) -> Iterator[StorageBackend]:
    """Yield an empty storage backend rooted in a temporary directory."""
    if backend == "lancedb":
        yield LanceDBStorage(path=str(tmp_path / "lancedb"), vector_dim=VECTOR_DIM)
        return

    from crewai.memory.storage.qdrant_edge_storage import QdrantEdgeStorage

    qdrant = QdrantEdgeStorage(path=str(tmp_path / "qdrant"), vector_dim=VECTOR_DIM)
    yield qdrant
    qdrant.close()


@pytest.fixture
def corpus(storage: StorageBackend) -> dict[str, MemoryRecord]:
    """Save the fixed corpus and return its records by key."""
    records: dict[str, MemoryRecord] = {}
    for index, (key, (scope, categories, metadata, created_at)) in enumerate(
        CORPUS_SPEC.items()
    ):
        records[key] = MemoryRecord(
            content=key,
            scope=scope,
            categories=categories,
            metadata=metadata,
            embedding=[1.0, float(index + 1), 0.5, 1.0],
            created_at=created_at + timedelta(minutes=index),
            last_accessed=created_at + timedelta(minutes=index),
        )
    storage.save(list(records.values()))
    return records


def _known_gap(
    request: pytest.FixtureRequest, backend: str, gaps: dict[str, str]
) -> None:
    """Mark the running test as a strict xfail when ``backend`` has a known gap."""
    reason = gaps.get(backend)
    if reason is not None:
        request.applymarker(pytest.mark.xfail(reason=reason, strict=True))


def _keys(corpus: dict[str, MemoryRecord], ids: list[str]) -> set[str]:
    by_id = {record.id: key for key, record in corpus.items()}
    return {by_id[record_id] for record_id in ids}


def _search(
    storage: StorageBackend, corpus: dict[str, MemoryRecord], **filters: Any
) -> set[str]:
    results = storage.search(QUERY_VECTOR, limit=SEARCH_LIMIT, **filters)
    return _keys(corpus, [record.id for record, _ in results])


def _surviving(storage: StorageBackend, corpus: dict[str, MemoryRecord]) -> set[str]:
    return {
        key
        for key, record in corpus.items()
        if storage.get_record(record.id) is not None
    }


def _assert_deleted(
    storage: StorageBackend,
    corpus: dict[str, MemoryRecord],
    deleted: int,
    expected: set[str],
) -> None:
    assert _surviving(storage, corpus) == ALL_KEYS - expected
    assert deleted == len(expected)
    assert storage.count() == len(ALL_KEYS) - len(expected)


class TestSearchScope:
    @pytest.mark.parametrize("scope_prefix", [None, "/"])
    def test_unscoped_search_reaches_every_record(
        self,
        storage: StorageBackend,
        corpus: dict[str, MemoryRecord],
        scope_prefix: str | None,
    ) -> None:
        assert _search(storage, corpus, scope_prefix=scope_prefix) == ALL_KEYS

    def test_scope_includes_descendants_and_excludes_root(
        self, storage: StorageBackend, corpus: dict[str, MemoryRecord]
    ) -> None:
        assert _search(storage, corpus, scope_prefix="/team") == ALL_KEYS - {
            "root_decision"
        }

    @pytest.mark.parametrize("scope_prefix", ["/team/beta", "/team/beta/"])
    def test_scope_excludes_sibling_scopes(
        self,
        storage: StorageBackend,
        corpus: dict[str, MemoryRecord],
        scope_prefix: str,
    ) -> None:
        assert _search(storage, corpus, scope_prefix=scope_prefix) == {
            "beta_old_decision",
            "beta_new_note",
        }

    def test_scope_excludes_sibling_sharing_a_string_prefix(
        self,
        request: pytest.FixtureRequest,
        backend: str,
        storage: StorageBackend,
        corpus: dict[str, MemoryRecord],
    ) -> None:
        _known_gap(request, backend, {"lancedb": LANCEDB_STRING_PREFIX_SCOPE})

        assert _search(storage, corpus, scope_prefix="/team/alpha") == {
            "alpha_old_decision",
            "alpha_new_note",
            "alpha_child_decision",
        }

    def test_missing_scope_returns_nothing(
        self, storage: StorageBackend, corpus: dict[str, MemoryRecord]
    ) -> None:
        assert _search(storage, corpus, scope_prefix="/team/gamma") == set()


class TestSearchFilters:
    def test_categories_match_any_listed_category(
        self, storage: StorageBackend, corpus: dict[str, MemoryRecord]
    ) -> None:
        assert _search(storage, corpus, categories=["finance", "note"]) == {
            "alpha_new_note",
            "alpha_child_decision",
            "beta_new_note",
        }

    def test_categories_combine_with_scope(
        self, storage: StorageBackend, corpus: dict[str, MemoryRecord]
    ) -> None:
        assert _search(
            storage, corpus, scope_prefix="/team/beta", categories=["decision"]
        ) == {"beta_old_decision"}

    def test_metadata_filter_requires_every_key(
        self, storage: StorageBackend, corpus: dict[str, MemoryRecord]
    ) -> None:
        assert _search(storage, corpus, metadata_filter={"status": "closed"}) == {
            "alpha_child_decision",
            "beta_new_note",
        }
        assert _search(
            storage, corpus, metadata_filter={"status": "open", "owner": "bo"}
        ) == {"alphabet_decision", "beta_old_decision"}

    def test_scope_categories_and_metadata_combine(
        self, storage: StorageBackend, corpus: dict[str, MemoryRecord]
    ) -> None:
        assert _search(
            storage,
            corpus,
            scope_prefix="/team",
            categories=["decision"],
            metadata_filter={"owner": "ana"},
        ) == {"alpha_old_decision", "alpha_child_decision"}

    def test_unmatched_filters_return_nothing(
        self, storage: StorageBackend, corpus: dict[str, MemoryRecord]
    ) -> None:
        assert _search(storage, corpus, categories=["missing"]) == set()
        assert _search(storage, corpus, metadata_filter={"owner": "zoe"}) == set()


class TestDeleteScope:
    def test_scope_deletes_subtree_and_spares_siblings_and_root(
        self,
        request: pytest.FixtureRequest,
        backend: str,
        storage: StorageBackend,
        corpus: dict[str, MemoryRecord],
    ) -> None:
        _known_gap(request, backend, {"lancedb": LANCEDB_ROOT_SCOPE_DELETE})

        deleted = storage.delete(scope_prefix="/team/beta")

        _assert_deleted(
            storage, corpus, deleted, {"beta_old_decision", "beta_new_note"}
        )

    def test_scope_spares_sibling_sharing_a_string_prefix(
        self,
        request: pytest.FixtureRequest,
        backend: str,
        storage: StorageBackend,
        corpus: dict[str, MemoryRecord],
    ) -> None:
        _known_gap(
            request,
            backend,
            {"lancedb": f"{LANCEDB_STRING_PREFIX_SCOPE}; {LANCEDB_ROOT_SCOPE_DELETE}"},
        )

        deleted = storage.delete(scope_prefix="/team/alpha")

        _assert_deleted(
            storage,
            corpus,
            deleted,
            {"alpha_old_decision", "alpha_new_note", "alpha_child_decision"},
        )

    def test_scope_and_older_than_both_apply(
        self,
        request: pytest.FixtureRequest,
        backend: str,
        storage: StorageBackend,
        corpus: dict[str, MemoryRecord],
    ) -> None:
        _known_gap(request, backend, {"lancedb": LANCEDB_ROOT_SCOPE_DELETE})

        deleted = storage.delete(scope_prefix="/team/beta", older_than=CUTOFF)

        _assert_deleted(storage, corpus, deleted, {"beta_old_decision"})


class TestDeleteFilters:
    def test_older_than_alone_deletes_only_older_records(
        self, storage: StorageBackend, corpus: dict[str, MemoryRecord]
    ) -> None:
        deleted = storage.delete(older_than=CUTOFF)

        _assert_deleted(
            storage,
            corpus,
            deleted,
            {
                "alpha_old_decision",
                "alphabet_decision",
                "beta_old_decision",
                "root_decision",
            },
        )

    def test_categories_combine_with_scope(
        self, storage: StorageBackend, corpus: dict[str, MemoryRecord]
    ) -> None:
        deleted = storage.delete(scope_prefix="/team/beta", categories=["decision"])

        _assert_deleted(storage, corpus, deleted, {"beta_old_decision"})

    def test_categories_combine_with_metadata(
        self, storage: StorageBackend, corpus: dict[str, MemoryRecord]
    ) -> None:
        deleted = storage.delete(
            categories=["decision"], metadata_filter={"status": "open"}
        )

        _assert_deleted(
            storage,
            corpus,
            deleted,
            {
                "alpha_old_decision",
                "alphabet_decision",
                "beta_old_decision",
                "root_decision",
            },
        )

    def test_scope_metadata_and_older_than_combine(
        self, storage: StorageBackend, corpus: dict[str, MemoryRecord]
    ) -> None:
        deleted = storage.delete(
            scope_prefix="/team",
            metadata_filter={"status": "open"},
            older_than=CUTOFF,
        )

        _assert_deleted(
            storage,
            corpus,
            deleted,
            {"alpha_old_decision", "alphabet_decision", "beta_old_decision"},
        )

    def test_unmatched_filters_delete_nothing(
        self, storage: StorageBackend, corpus: dict[str, MemoryRecord]
    ) -> None:
        assert storage.delete(categories=["missing"]) == 0
        assert storage.delete(metadata_filter={"owner": "zoe"}) == 0
        assert storage.delete(record_ids=[str(uuid4())]) == 0
        assert _surviving(storage, corpus) == ALL_KEYS


class TestDeleteRecordIds:
    def test_record_ids_alone_delete_only_those_records(
        self, storage: StorageBackend, corpus: dict[str, MemoryRecord]
    ) -> None:
        targets = {"alpha_new_note", "root_decision"}

        deleted = storage.delete(record_ids=[corpus[key].id for key in targets])

        _assert_deleted(storage, corpus, deleted, targets)

    def test_record_ids_combine_with_categories(
        self,
        request: pytest.FixtureRequest,
        backend: str,
        storage: StorageBackend,
        corpus: dict[str, MemoryRecord],
    ) -> None:
        _known_gap(request, backend, {"lancedb": LANCEDB_RECORD_IDS_COMBINED})

        deleted = storage.delete(
            record_ids=[
                corpus["alpha_old_decision"].id,
                corpus["alpha_new_note"].id,
            ],
            categories=["decision"],
        )

        _assert_deleted(storage, corpus, deleted, {"alpha_old_decision"})

    def test_record_ids_combine_with_older_than(
        self,
        request: pytest.FixtureRequest,
        backend: str,
        storage: StorageBackend,
        corpus: dict[str, MemoryRecord],
    ) -> None:
        _known_gap(request, backend, {"lancedb": LANCEDB_RECORD_IDS_COMBINED})

        deleted = storage.delete(
            record_ids=[
                corpus["alpha_old_decision"].id,
                corpus["alpha_new_note"].id,
            ],
            older_than=CUTOFF,
        )

        _assert_deleted(storage, corpus, deleted, {"alpha_old_decision"})

    def test_record_ids_combine_with_scope(
        self,
        request: pytest.FixtureRequest,
        backend: str,
        storage: StorageBackend,
        corpus: dict[str, MemoryRecord],
    ) -> None:
        _known_gap(
            request,
            backend,
            {
                "lancedb": "LanceDB delete by record_ids ignores scope_prefix "
                "(#8653; fix proposed in #8654)",
                "qdrant_edge": "Qdrant Edge delete by record_ids ignores "
                "scope_prefix (#8462, #8653; fixes proposed in #8467, #8654)",
            },
        )

        deleted = storage.delete(
            record_ids=[
                corpus["alpha_old_decision"].id,
                corpus["beta_old_decision"].id,
            ],
            scope_prefix="/team/beta",
        )

        _assert_deleted(storage, corpus, deleted, {"beta_old_decision"})
