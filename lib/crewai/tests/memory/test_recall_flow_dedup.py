"""Tests for ``RecallFlow.synthesize_results`` deduplication.

A record reachable from more than one ``(sub-query, scope)`` task is scored once
per task, and the scores differ. ``_do_search`` appends findings in
``as_completed`` order, so a dedup that kept the first occurrence let thread
completion timing decide which score survived — the same query could return the
same record at a different rank on consecutive runs.

Each test drives ``synthesize_results`` on a hand-built ``chunk_findings`` list
rather than through a thread pool, so the assertions pin the dedup rule itself
instead of racing the scheduler.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from crewai.memory.recall_flow import RecallFlow, RecallState
from crewai.memory.types import MemoryConfig, MemoryRecord


_FIXED_CREATED_AT = datetime(2026, 1, 1, 12, 0, 0)


def _record(record_id: str) -> MemoryRecord:
    """A record with everything but ``id`` held constant.

    ``compute_composite_score`` also reads ``created_at`` and ``importance``, so
    pinning both keeps the composite a function of the semantic score alone.
    """
    return MemoryRecord(
        id=record_id,
        content=f"content for {record_id}",
        created_at=_FIXED_CREATED_AT,
        importance=0.5,
    )


def _flow_with_findings(findings: list[dict]) -> RecallFlow:
    flow = RecallFlow(
        storage=object(),
        llm=object(),
        embedder=object(),
        config=MemoryConfig(),
    )
    flow._state = RecallState(limit=10, chunk_findings=findings)
    return flow


def test_duplicate_record_keeps_the_best_score_not_the_first_seen() -> None:
    """The regression: the surviving score must not depend on finding order."""
    record = _record("R")
    high = {"scope": "/a", "results": [(record, 0.9)], "top_score": 0.9}
    low = {"scope": "/a/b", "results": [(record, 0.1)], "top_score": 0.1}

    high_first = _flow_with_findings([high, low]).synthesize_results()
    low_first = _flow_with_findings([low, high]).synthesize_results()
    # What the 0.9 and the 0.1 semantic scores are worth on their own. The
    # composite also carries a recency term read from the wall clock at scoring
    # time, so these are compared with a tolerance well below the 0.9/0.1 gap
    # rather than for exact equality.
    only_high = _flow_with_findings([high]).synthesize_results()[0].score
    only_low = _flow_with_findings([low]).synthesize_results()[0].score

    assert len(high_first) == 1
    assert len(low_first) == 1
    # Both orderings resolve to the higher score, not to whichever came first.
    assert high_first[0].score == pytest.approx(only_high, abs=1e-6)
    assert low_first[0].score == pytest.approx(only_high, abs=1e-6)
    assert only_high - only_low > 0.1


def test_record_reachable_from_several_scopes_appears_once() -> None:
    """Dedup still collapses duplicates; keeping the best score is not keeping both."""
    record = _record("R")
    findings = [
        {"scope": f"/s{i}", "results": [(record, 0.1 * i)], "top_score": 0.1 * i}
        for i in range(1, 5)
    ]

    matches = _flow_with_findings(findings).synthesize_results()

    assert [m.record.id for m in matches] == ["R"]


def test_equal_scores_rank_by_record_id_rather_than_arrival() -> None:
    """Ties must not fall back to whichever task finished first."""
    a, b = _record("aaa"), _record("bbb")
    forward = [
        {"scope": "/x", "results": [(a, 0.5)], "top_score": 0.5},
        {"scope": "/y", "results": [(b, 0.5)], "top_score": 0.5},
    ]
    reverse = list(reversed(forward))

    assert [m.record.id for m in _flow_with_findings(forward).synthesize_results()] == [
        "aaa",
        "bbb",
    ]
    assert [m.record.id for m in _flow_with_findings(reverse).synthesize_results()] == [
        "aaa",
        "bbb",
    ]


def test_distinct_records_are_all_kept_and_ranked_by_score() -> None:
    """Ranking behaviour that already worked cannot regress."""
    low, mid, high = _record("low"), _record("mid"), _record("high")
    findings = [
        {"scope": "/a", "results": [(low, 0.1), (high, 0.9)], "top_score": 0.9},
        {"scope": "/b", "results": [(mid, 0.5)], "top_score": 0.5},
    ]

    matches = _flow_with_findings(findings).synthesize_results()

    assert [m.record.id for m in matches] == ["high", "mid", "low"]


def test_limit_is_applied_after_deduplication() -> None:
    """A duplicate must not consume one of the caller's ``limit`` slots."""
    dup = _record("dup")
    others = [_record(f"r{i}") for i in range(3)]
    findings = [
        {"scope": "/a", "results": [(dup, 0.9)], "top_score": 0.9},
        {"scope": "/a/b", "results": [(dup, 0.8)], "top_score": 0.8},
        {"scope": "/c", "results": [(r, 0.5) for r in others], "top_score": 0.5},
    ]
    flow = _flow_with_findings(findings)
    flow._state.limit = 3

    matches = flow.synthesize_results()

    assert len(matches) == 3
    assert [m.record.id for m in matches].count("dup") == 1


def test_malformed_findings_are_skipped_without_raising() -> None:
    """Defensive shapes the original loop tolerated are still tolerated."""
    record = _record("R")
    findings = [
        "not a dict",
        {"scope": "/a", "results": "not a list"},
        {"scope": "/b", "results": [("not a record", 0.5), (record,), (record, 0.7)]},
    ]

    matches = _flow_with_findings(findings).synthesize_results()  # type: ignore[arg-type]

    assert [m.record.id for m in matches] == ["R"]
