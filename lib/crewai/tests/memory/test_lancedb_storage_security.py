"""Regression tests for SQL-injection hardening of ``LanceDBStorage``.

Issue: GH #5728

LanceDB's ``where()`` accepts a raw Apache DataFusion SQL expression and does
not support parameterized queries.  Earlier versions of ``LanceDBStorage``
interpolated caller-supplied scope paths and record IDs directly into the
WHERE clause via f-strings, which allowed:

* an unprivileged caller to escape the configured ``scope`` sandbox and
  read / delete records belonging to any other scope, and
* legitimate strings containing single quotes (e.g. ``"O'Brien"``) to crash
  with a SQL parse error.

These tests pin the hardened behaviour so the bug can never silently
regress.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.memory.types import MemoryRecord


@pytest.fixture
def storage(tmp_path: Path) -> LanceDBStorage:
    return LanceDBStorage(path=str(tmp_path / "mem"), vector_dim=4)


def _seed(storage: LanceDBStorage) -> None:
    storage.save(
        [
            MemoryRecord(
                id="alpha-1",
                content="alpha",
                scope="/alpha",
                embedding=[0.1, 0.2, 0.3, 0.4],
            ),
            MemoryRecord(
                id="bravo-1",
                content="bravo",
                scope="/bravo",
                embedding=[0.5, 0.6, 0.7, 0.8],
            ),
        ]
    )


# ----------------------------------------------------------------------
# Helper unit tests
# ----------------------------------------------------------------------


def test_escape_sql_str_doubles_single_quotes() -> None:
    assert LanceDBStorage._escape_sql_str("O'Brien") == "O''Brien"
    assert LanceDBStorage._escape_sql_str("a'; DROP TABLE t;--") == "a''; DROP TABLE t;--"
    # Non-string inputs are coerced.
    assert LanceDBStorage._escape_sql_str(42) == "42"


def test_escape_like_escapes_metacharacters() -> None:
    # Backslash is escaped first so subsequent escapes don't double-escape.
    assert LanceDBStorage._escape_like("a\\b") == "a\\\\b"
    assert LanceDBStorage._escape_like("a%b") == "a\\%b"
    assert LanceDBStorage._escape_like("a_b") == "a\\_b"
    assert LanceDBStorage._escape_like("O'Brien") == "O''Brien"
    # All metacharacters together.
    assert (
        LanceDBStorage._escape_like("100%_off'\\")
        == "100\\%\\_off''\\\\"
    )


# ----------------------------------------------------------------------
# Sink 1: search() must not leak across scopes via injected scope_prefix
# ----------------------------------------------------------------------


def test_search_scope_prefix_injection_returns_no_match(
    storage: LanceDBStorage,
) -> None:
    _seed(storage)
    # Classic ``' OR '1'='1`` style payload aimed at widening the LIKE.
    payload = "/alpha' OR '1'='1"
    results = storage.search([0.1, 0.2, 0.3, 0.4], scope_prefix=payload, limit=10)
    assert results == []


def test_search_scope_prefix_with_apostrophe_does_not_crash(
    storage: LanceDBStorage,
) -> None:
    storage.save(
        [
            MemoryRecord(
                id="x-1",
                content="x",
                scope="/O'Brien",
                embedding=[0.1, 0.2, 0.3, 0.4],
            )
        ]
    )
    # Must round-trip a legitimate scope containing an apostrophe.
    results = storage.search(
        [0.1, 0.2, 0.3, 0.4], scope_prefix="/O'Brien", limit=10
    )
    assert len(results) == 1
    assert results[0][0].scope == "/O'Brien"


def test_search_scope_prefix_percent_is_literal_not_wildcard(
    storage: LanceDBStorage,
) -> None:
    """A ``%`` in the user-supplied prefix must be treated as a literal,
    not as a SQL ``LIKE`` wildcard that would match unrelated scopes."""
    _seed(storage)
    # ``/%`` would, without escaping, match every scope that starts with ``/``.
    results = storage.search([0.1, 0.2, 0.3, 0.4], scope_prefix="/%", limit=10)
    assert results == []


# ----------------------------------------------------------------------
# Sink 2: delete(scope_prefix=...) must not let an attacker wipe other scopes
# ----------------------------------------------------------------------


def test_delete_scope_prefix_injection_does_not_bypass_isolation(
    storage: LanceDBStorage,
) -> None:
    """The most damaging payload from issue #5728: a malicious scope_prefix
    that, before the fix, deleted every record in the table by appending
    ``OR scope LIKE '/%`` to the WHERE clause."""
    _seed(storage)
    assert storage.count() == 2

    # Pre-fix, this WHERE evaluated to:
    #   scope LIKE '/alpha' OR scope LIKE '/%' OR scope = '/'
    # which deletes /alpha AND /bravo.  Post-fix the entire payload is
    # treated as a literal prefix and matches nothing.
    payload = "/alpha' OR scope LIKE '/%"
    n = storage.delete(scope_prefix=payload)

    assert n == 0
    assert storage.count() == 2

    # And the legitimate scoped delete must still work.
    n = storage.delete(scope_prefix="/alpha")
    assert n == 1
    assert storage.count() == 1
    remaining = storage.list_records()
    assert [r.scope for r in remaining] == ["/bravo"]


# ----------------------------------------------------------------------
# Sink 3: delete(record_ids=[...]) must escape IDs
# ----------------------------------------------------------------------


def test_delete_record_ids_injection_does_not_match_real_rows(
    storage: LanceDBStorage,
) -> None:
    _seed(storage)
    # An attacker-controlled "id" containing a quote used to either
    # crash the SQL tokenizer or, worse, evaluate a tautology.
    n = storage.delete(record_ids=["' OR '1'='1"])
    assert n == 0
    assert storage.count() == 2


def test_delete_record_ids_with_apostrophe_round_trips(
    storage: LanceDBStorage,
) -> None:
    storage.save(
        [
            MemoryRecord(
                id="O'Reilly-42",
                content="ok",
                scope="/team",
                embedding=[0.0] * 4,
            )
        ]
    )
    assert storage.count() == 1
    n = storage.delete(record_ids=["O'Reilly-42"])
    assert n == 1
    assert storage.count() == 0


# ----------------------------------------------------------------------
# Sink 4: reset(scope_prefix=...) must not crash on apostrophes
# ----------------------------------------------------------------------


def test_reset_scope_prefix_with_apostrophe_does_not_crash(
    storage: LanceDBStorage,
) -> None:
    storage.save(
        [
            MemoryRecord(
                id="r-1",
                content="x",
                scope="/O'Brien/team",
                embedding=[0.0] * 4,
            ),
            MemoryRecord(
                id="r-2",
                content="y",
                scope="/other",
                embedding=[0.0] * 4,
            ),
        ]
    )
    # Must not raise and must scope the reset correctly.
    storage.reset(scope_prefix="/O'Brien")
    remaining = storage.list_records()
    assert [r.scope for r in remaining] == ["/other"]


def test_reset_scope_prefix_injection_does_not_drop_unrelated_scopes(
    storage: LanceDBStorage,
) -> None:
    _seed(storage)
    # ``' OR scope >= ''`` would, without escaping, broaden the range
    # comparison to every row.
    storage.reset(scope_prefix="/alpha' OR scope >= '")
    assert storage.count() == 2  # nothing should have been deleted


# ----------------------------------------------------------------------
# Scan-based readers: list_records / list_scopes / get_scope_info /
# list_categories / count all flow through ``_scan_rows`` and used to
# crash on apostrophes and leak across scopes via ``%``/``_`` wildcards.
# ----------------------------------------------------------------------


def test_scan_methods_with_apostrophe_in_scope(
    storage: LanceDBStorage,
) -> None:
    storage.save(
        [
            MemoryRecord(
                id="s-1",
                content="x",
                scope="/O'Brien",
                categories=["c1"],
                embedding=[0.0] * 4,
            )
        ]
    )

    assert [r.id for r in storage.list_records(scope_prefix="/O'Brien")] == ["s-1"]
    info = storage.get_scope_info("/O'Brien")
    assert info.record_count == 1
    assert info.path == "/O'Brien"
    assert storage.list_scopes("/O'Brien") == []
    assert storage.list_categories(scope_prefix="/O'Brien") == {"c1": 1}
    assert storage.count(scope_prefix="/O'Brien") == 1


def test_scan_methods_treat_percent_as_literal(
    storage: LanceDBStorage,
) -> None:
    _seed(storage)
    # ``/%`` should NOT match every scope rooted at ``/``; it should match
    # only literal ``/<percent>...`` prefixes (of which there are none).
    assert storage.list_records(scope_prefix="/%") == []
    assert storage.count(scope_prefix="/%") == 0
    assert storage.list_categories(scope_prefix="/%") == {}
