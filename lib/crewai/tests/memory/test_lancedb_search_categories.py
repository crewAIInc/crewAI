"""Regression tests for server-side category filtering in LanceDBStorage.search().

Categories were applied only AFTER the ANN fetch (``limit * 3`` oversample), so
matching records ranked below the oversampled fetch were silently missed -- the
same defect already fixed for the qdrant backend. Categories now ride the same
``where()`` clause the scope filter uses, and are matched through their
``json.dumps`` rendering so the delimiting quotes keep the match exact and any
LIKE metacharacter in a category name still matches literally.
``metadata_filter`` keeps the client-side post-filter with oversampling.
"""

from __future__ import annotations

from pathlib import Path

from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.memory.types import MemoryRecord

DIM = 4
NEAR = [1.0, 0.0, 0.0, 0.0]
FAR = [0.0, 0.0, 0.0, 1.0]


def _record(
    record_id: str,
    categories: list[str],
    *,
    scope: str = "/proj",
    embedding: list[float] | None = None,
    metadata: dict | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        id=record_id,
        content=f"content of {record_id}",
        scope=scope,
        categories=categories,
        metadata=metadata or {},
        embedding=embedding if embedding is not None else FAR,
    )


def _new_storage(path: Path, records: list[MemoryRecord]) -> LanceDBStorage:
    storage = LanceDBStorage(path=path, vector_dim=DIM, compact_every=0)
    storage.save(records)
    return storage


def _search_ids(
    storage: LanceDBStorage,
    categories: list[str],
    scope_prefix: str | None = None,
) -> set[str]:
    return {
        r.id
        for r, _ in storage.search(
            NEAR, scope_prefix=scope_prefix, categories=categories, limit=100
        )
    }


def test_matches_below_fetch_limit_are_found(tmp_path: Path) -> None:
    # Eight near neighbours that do not match the category outrank the two
    # that do, so the old limit*3=6 fetch never saw a single match.
    records = [_record(f"noise-{i}", ["other"], embedding=NEAR) for i in range(8)]
    records += [_record(f"target-{i}", ["target"]) for i in range(2)]
    storage = _new_storage(tmp_path / "mem", records)

    hits = storage.search(NEAR, categories=["target"], limit=2)

    assert sorted(r.id for r, _ in hits) == ["target-0", "target-1"]


def test_categories_limit_respected(tmp_path: Path) -> None:
    storage = _new_storage(
        tmp_path / "mem", [_record(f"target-{i}", ["target"]) for i in range(5)]
    )

    hits = storage.search(FAR, categories=["target"], limit=2)

    assert len(hits) == 2
    assert all(r.id.startswith("target-") for r, _ in hits)


def test_metadata_filter_with_categories_still_applies(tmp_path: Path) -> None:
    storage = _new_storage(
        tmp_path / "mem",
        [
            _record("target-keep", ["target"], metadata={"kind": "keep"}),
            _record("target-drop", ["target"], metadata={"kind": "drop"}),
            _record("noise-keep", ["other"], metadata={"kind": "keep"}),
        ],
    )

    hits = storage.search(
        FAR, categories=["target"], metadata_filter={"kind": "keep"}, limit=5
    )

    assert [r.id for r, _ in hits] == ["target-keep"]


def test_category_is_matched_exactly_not_as_substring(tmp_path: Path) -> None:
    storage = _new_storage(
        tmp_path / "mem",
        [_record("exact", ["note"]), _record("longer", ["notebook"])],
    )

    assert _search_ids(storage, categories=["note"]) == {"exact"}
    assert _search_ids(storage, categories=["notebook"]) == {"longer"}


def test_any_of_several_categories_matches(tmp_path: Path) -> None:
    storage = _new_storage(
        tmp_path / "mem",
        [
            _record("first", ["alpha"]),
            _record("second", ["beta"]),
            _record("neither", ["gamma"]),
        ],
    )

    assert _search_ids(storage, categories=["alpha", "beta"]) == {"first", "second"}


def test_like_metacharacters_in_category_match_literally(tmp_path: Path) -> None:
    storage = _new_storage(
        tmp_path / "mem",
        [
            _record("pct", ["100%"]),
            _record("digits", ["1000"]),
            _record("underscore", ["a_b"]),
            _record("lookalike", ["axb"]),
        ],
    )

    assert _search_ids(storage, categories=["100%"]) == {"pct"}
    assert _search_ids(storage, categories=["a_b"]) == {"underscore"}


def test_quotes_and_backslashes_in_category_match_literally(
    tmp_path: Path,
) -> None:
    storage = _new_storage(
        tmp_path / "mem",
        [
            _record("quote", ["o'brien"]),
            _record("double-quote", ['say"hi']),
            _record("backslash", ["share\\docs"]),
            _record("other", ["plain"]),
        ],
    )

    assert _search_ids(storage, categories=["o'brien"]) == {"quote"}
    assert _search_ids(storage, categories=['say"hi']) == {"double-quote"}
    assert _search_ids(storage, categories=["share\\docs"]) == {"backslash"}


def test_non_ascii_category_matches(tmp_path: Path) -> None:
    # json.dumps escapes non-ASCII on the way in, so the filter has to be built
    # from the same rendering rather than from the raw category text.
    storage = _new_storage(
        tmp_path / "mem",
        [_record("cjk", ["目标"]), _record("latin", ["target"])],
    )

    assert _search_ids(storage, categories=["目标"]) == {"cjk"}


def test_scope_prefix_and_categories_combine(tmp_path: Path) -> None:
    storage = _new_storage(
        tmp_path / "mem",
        [
            _record("mine", ["target"], scope="/tenant-a/notes"),
            _record("theirs", ["target"], scope="/tenant-b/secrets"),
            _record("mine-other", ["other"], scope="/tenant-a/notes"),
        ],
    )

    assert _search_ids(storage, scope_prefix="/tenant-a", categories=["target"]) == {
        "mine"
    }
