"""A bare scope name addresses the same records as its anchored form.

Scope paths are stored anchored at the root (``"/sales"``), which is the form
:func:`crewai.memory.utils.normalize_scope_path` produces. ``LanceDBStorage``
matches scopes with a raw string prefix, so ``"sales"`` and ``"/sales"`` only
address the same rows if every filter site anchors the prefix the same way.
Read paths did not anchor it while ``delete()``/``get_scope_info()`` did, so
``Memory.forget(scope="sales")`` deleted nothing while ``Memory.recall(
scope="sales")`` returned the record, and ``Memory.reset(scope="sales")``
matched a different row set again. These tests pin one scope to one row set
across search, list, count, reset, forget, and both built-in backends.
"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from unittest.mock import MagicMock

from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.memory.types import MemoryRecord
from crewai.memory.unified_memory import Memory
import pytest


DIM = 4
SALES_RECORDS = {"sales-1", "sales-2"}

# Every spelling of one scope: unanchored, anchored, and the trailing-slash
# variants `str.rstrip("/")` is applied to before matching.
SCOPE_SPELLINGS = ["sales", "/sales", "sales/", "/sales/"]


def _record(record_id: str, scope: str) -> MemoryRecord:
    return MemoryRecord(
        id=record_id,
        content=f"content of {record_id}",
        scope=scope,
        categories=["note"],
        embedding=[0.1] * DIM,
    )


def _new_storage(path: Path) -> LanceDBStorage:
    storage = LanceDBStorage(path=path, vector_dim=DIM, compact_every=0)
    storage.save(
        [
            _record("sales-1", "/sales"),
            _record("sales-2", "/sales/eu"),
            _record("support-1", "/support"),
        ]
    )
    return storage


def _all_ids(storage: LanceDBStorage) -> set[str]:
    return {r.id for r in storage.list_records(limit=1000)}


def _search_ids(storage: LanceDBStorage, scope_prefix: str) -> set[str]:
    return {
        r.id
        for r, _ in storage.search([0.1] * DIM, scope_prefix=scope_prefix, limit=100)
    }


@pytest.fixture
def storage(tmp_path: Path) -> LanceDBStorage:
    return _new_storage(tmp_path / "mem")


@pytest.fixture
def embedder() -> MagicMock:
    mock = MagicMock()
    mock.side_effect = lambda texts: [[0.1] * DIM for _ in texts]
    return mock


@pytest.mark.parametrize("prefix", SCOPE_SPELLINGS)
def test_every_spelling_of_a_scope_reads_the_same_records(
    storage: LanceDBStorage, prefix: str
) -> None:
    assert _search_ids(storage, prefix) == SALES_RECORDS
    assert {r.id for r in storage.list_records(prefix)} == SALES_RECORDS
    assert storage.count(prefix) == 2
    assert storage.get_scope_info(prefix).record_count == 2
    assert storage.get_scope_info(prefix).path == "/sales"
    assert storage.list_categories(prefix) == {"note": 2}
    assert storage.list_scopes(prefix) == ["/sales/eu"]


@pytest.mark.parametrize("prefix", SCOPE_SPELLINGS)
def test_every_spelling_of_a_scope_deletes_the_same_records(
    tmp_path: Path, prefix: str
) -> None:
    storage = _new_storage(tmp_path / "mem")
    deleted = storage.delete(scope_prefix=prefix)
    assert deleted == 2
    assert _all_ids(storage) == {"support-1"}


@pytest.mark.parametrize("prefix", SCOPE_SPELLINGS)
def test_every_spelling_of_a_scope_resets_the_same_records(
    tmp_path: Path, prefix: str
) -> None:
    storage = _new_storage(tmp_path / "mem")
    storage.reset(scope_prefix=prefix)
    assert _all_ids(storage) == {"support-1"}


@pytest.mark.parametrize("prefix", SCOPE_SPELLINGS)
def test_canonical_prefixes_still_do_not_cross_scope_boundaries(
    storage: LanceDBStorage, prefix: str
) -> None:
    """Anchoring must not widen a prefix into an unrelated scope's rows.

    Prefixes are compared as raw strings, so ``"/sales"`` legitimately also
    selects ``"/salesforce"``; what must not happen is ``"/sales"`` reaching
    ``"/support"``.
    """
    assert _search_ids(storage, prefix) == SALES_RECORDS
    assert _search_ids(storage, "support") == {"support-1"}
    assert storage.count("support") == 1
    assert storage.count("/support") == 1


@pytest.mark.parametrize("scope", SCOPE_SPELLINGS)
def test_remember_persists_the_anchored_scope(
    tmp_path: Path, embedder: MagicMock, scope: str
) -> None:
    memory = Memory(
        storage=str(tmp_path / "db"), llm=MagicMock(), embedder=embedder
    )
    record = memory.remember(
        "We chose Python for the sales pipeline.",
        scope=scope,
        categories=[],
        importance=0.5,
        metadata={},
    )
    assert record is not None
    assert record.scope == "/sales"


@pytest.mark.parametrize("spelling", SCOPE_SPELLINGS)
def test_recall_count_and_forget_agree_on_one_scope(
    tmp_path: Path, embedder: MagicMock, spelling: str
) -> None:
    """The reported symptom: recall found the record, forget deleted nothing."""
    memory = Memory(
        storage=str(tmp_path / "db"), llm=MagicMock(), embedder=embedder
    )
    record = memory.remember(
        "We chose Python for the sales pipeline.",
        scope="sales",
        categories=[],
        importance=0.5,
        metadata={},
    )
    assert record is not None

    found = memory.recall("sales pipeline", scope=spelling, limit=5, depth="shallow")
    assert {m.record.id for m in found} == {record.id}
    assert memory._storage.count(spelling) == 1

    assert memory.forget(scope=spelling) == 1
    assert memory._storage.count() == 0


@pytest.mark.parametrize(
    "spelling",
    SCOPE_SPELLINGS,
)
def test_reset_clears_the_same_scope_recall_reports(
    tmp_path: Path, embedder: MagicMock, spelling: str
) -> None:
    memory = Memory(
        storage=str(tmp_path / "db"), llm=MagicMock(), embedder=embedder
    )
    memory.remember(
        "A sales note.",
        scope="sales",
        categories=[],
        importance=0.5,
        metadata={},
    )
    assert memory.info(spelling).record_count == 1
    memory.reset(scope=spelling)
    assert memory._storage.count() == 0


def test_update_persists_the_anchored_scope(
    tmp_path: Path, embedder: MagicMock
) -> None:
    memory = Memory(
        storage=str(tmp_path / "db"), llm=MagicMock(), embedder=embedder
    )
    record = memory.remember(
        "Move me to another scope.",
        scope="/sales",
        categories=[],
        importance=0.5,
        metadata={},
    )
    assert record is not None
    updated = memory.update(record.id, scope="support")
    assert updated.scope == "/support"
    stored = memory._storage.get_record(record.id)
    assert stored is not None
    assert stored.scope == "/support"


# Spellings that carry repeated slashes. The write path collapses them via
# ``normalize_scope_path``, so ``"//sales"`` is persisted as ``"/sales"``; a
# filter that keeps the doubled slash builds ``LIKE '//sales%'`` and silently
# matches nothing.
SLASH_SPELLINGS = ["//sales", "///sales", "//sales/", "/sales//"]


@pytest.mark.parametrize("spelling", SLASH_SPELLINGS)
def test_repeated_slashes_address_the_same_scope_as_the_collapsed_form(
    tmp_path: Path, embedder: MagicMock, spelling: str
) -> None:
    """A scope written and read with repeated slashes must round-trip."""
    memory = Memory(
        storage=str(tmp_path / "db"), llm=MagicMock(), embedder=embedder
    )
    record = memory.remember(
        "We chose Python for the sales pipeline.",
        scope="//sales",
        categories=[],
        importance=0.5,
        metadata={},
    )
    assert record is not None
    assert record.scope == "/sales"

    found = memory.recall("sales pipeline", scope=spelling, limit=5, depth="shallow")
    assert {m.record.id for m in found} == {record.id}
    assert memory._storage.count(spelling) == 1

    assert memory.forget(scope=spelling) == 1
    assert memory._storage.count() == 0


def _memory_with_backend(tmp_path: Path, backend: str, embedder: MagicMock) -> Memory:
    spec = "qdrant-edge" if backend == "qdrant-edge" else str(tmp_path / "lancedb")
    return Memory(storage=spec, llm=MagicMock(), embedder=embedder)


@pytest.mark.parametrize("backend", ["lancedb", "qdrant-edge"])
def test_built_in_backends_agree_on_a_bare_scope(
    tmp_path: Path, embedder: MagicMock, backend: str
) -> None:
    if backend == "qdrant-edge" and find_spec("qdrant_edge") is None:
        pytest.skip("qdrant-edge-py not installed")

    memory = _memory_with_backend(tmp_path, backend, embedder)
    try:
        record = memory.remember(
            "We chose Python for the sales pipeline.",
            scope="sales",
            categories=[],
            importance=0.5,
            metadata={},
        )
        assert record is not None
        assert record.scope == "/sales"

        for spelling in SCOPE_SPELLINGS:
            found = memory.recall(
                "sales pipeline", scope=spelling, limit=5, depth="shallow"
            )
            assert {m.record.id for m in found} == {record.id}
            assert memory._storage.count(spelling) == 1

        assert memory.forget(scope="sales") == 1
        assert memory._storage.count() == 0
    finally:
        close = getattr(memory._storage, "close", None)
        if close is not None:
            close()


@pytest.fixture(autouse=True)
def isolated_storage_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the spec-based backends out of the developer's data directory.

    ``Memory(storage="qdrant-edge")`` resolves to the platform data directory,
    so without this the parametrized cases share one store between runs and read
    (and write) whatever the developer has stored for real.
    """
    monkeypatch.setenv("CREWAI_STORAGE_DIR", str(tmp_path / "storage"))


@pytest.mark.parametrize("backend", ["lancedb", "qdrant-edge"])
@pytest.mark.parametrize("spelling", SCOPE_SPELLINGS)
def test_child_scopes_are_listed_under_every_spelling(
    tmp_path: Path, embedder: MagicMock, backend: str, spelling: str
) -> None:
    """``list_scopes``/``child_scopes`` must anchor the parent like the filters do."""
    if backend == "qdrant-edge" and find_spec("qdrant_edge") is None:
        pytest.skip("qdrant-edge-py not installed")

    memory = _memory_with_backend(tmp_path, backend, embedder)
    try:
        memory.remember("Parent", scope="/sales", categories=[], importance=0.5)
        memory.remember("Child", scope="/sales/eu", categories=[], importance=0.5)

        storage = memory._storage
        assert storage.list_scopes(spelling) == ["/sales/eu"]
        info = storage.get_scope_info(spelling)
        assert info.path == "/sales"
        assert info.record_count == 2
        assert info.child_scopes == ["/sales/eu"]
    finally:
        close = getattr(memory._storage, "close", None)
        if close is not None:
            close()


def test_legacy_unanchored_rows_stay_reachable(tmp_path: Path) -> None:
    """Rows stored before scopes were anchored stay addressable, not orphaned.

    Releases before this change persisted ``scope="sales"`` exactly as typed, so
    an existing database can hold both spellings. Canonicalizing the query must
    not make those rows disappear from scoped reads while unscoped reads still
    return them.
    """
    storage = _new_storage(tmp_path / "mem")
    storage.save([
        _record("legacy-1", "sales"),
        _record("legacy-2", "sales/eu"),
        _record("legacy-3", "salesforce"),
        _record("legacy-4", "sales-eu"),
    ])
    siblings = {"legacy-3", "legacy-4"}
    try:
        # Both spellings address the anchored and the legacy rows, exactly and
        # by descendant.
        for spelling in SCOPE_SPELLINGS:
            assert _search_ids(storage, spelling) == SALES_RECORDS | {
                "legacy-1",
                "legacy-2",
            }
            assert storage.count(spelling) == 4

        # The legacy half is not `LIKE 'sales%'`: a hyphenated or suffix sibling
        # in the old spelling is a different scope, and stays out.
        assert storage.count("salesforce") == 1
        assert storage.count("sales-eu") == 1

        info = storage.get_scope_info("/sales")
        assert info.record_count == 4
        assert info.child_scopes == ["/sales/eu"]

        # Deletion reaches the legacy rows and leaves both siblings alone:
        # deleting rows that no scoped read can see is the failure this guards.
        assert storage.delete(scope_prefix="/sales") == 4
        assert _all_ids(storage) == {"support-1"} | siblings
    finally:
        close = getattr(storage, "close", None)
        if close is not None:
            close()


def test_reset_matches_the_same_predicate_as_delete(tmp_path: Path) -> None:
    """``reset`` must own exactly the rows the other scoped calls address.

    It used to build a lexicographic range of its own, which both missed the
    legacy spelling and sorted siblings such as ``sales-eu`` into the interval.
    """
    storage = _new_storage(tmp_path / "mem")
    storage.save([
        _record("legacy-1", "sales"),
        _record("legacy-2", "sales/eu"),
        _record("legacy-3", "salesforce"),
        _record("legacy-4", "sales-eu"),
    ])
    try:
        assert storage.count("/sales") == 4
        storage.reset("/sales")
        # What `count` counted is what `reset` removed, and no sibling went with
        # it.
        assert storage.count("/sales") == 0
        assert _all_ids(storage) == {"support-1", "legacy-3", "legacy-4"}
    finally:
        close = getattr(storage, "close", None)
        if close is not None:
            close()


def test_qdrant_legacy_payload_scope_is_still_a_child(tmp_path: Path) -> None:
    """A legacy ``sales/eu`` payload must compare as a child of ``/sales``."""
    if find_spec("qdrant_edge") is None:
        pytest.skip("qdrant-edge-py not installed")

    memory = _memory_with_backend(tmp_path, "qdrant-edge", MagicMock())
    try:
        storage = memory._storage
        storage.save([_record("legacy-eu", "sales/eu")])

        for spelling in SCOPE_SPELLINGS:
            # The ancestor filter already resolves the legacy scope; the
            # comparison used to look at the raw payload and drop it.
            assert storage.list_scopes(spelling) == ["/sales/eu"]
            info = storage.get_scope_info(spelling)
            assert info.record_count == 1
            assert info.child_scopes == ["/sales/eu"]
            assert storage.count(spelling) == 1

        # `_build_scope_filter` gets the same canonicalization, so a doubled
        # slash queries `/sales` instead of the literal `//sales`.
        assert storage.count("//sales") == 1
    finally:
        close = getattr(memory._storage, "close", None)
        if close is not None:
            close()

