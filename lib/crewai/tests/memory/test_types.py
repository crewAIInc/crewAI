"""Tests for crewai.memory.types scoring helpers."""

from datetime import datetime, timezone

from crewai.memory.types import (
    MemoryConfig,
    MemoryRecord,
    compute_composite_score,
)


def test_compute_composite_score_with_timezone_aware_created_at():
    """A timezone-aware created_at must not crash.

    Previously the age was computed as `datetime.utcnow() - record.created_at`;
    subtracting an aware datetime from a naive one raises TypeError.
    """
    record = MemoryRecord(content="hello", created_at=datetime.now(timezone.utc))

    score, reasons = compute_composite_score(
        record, semantic_score=0.8, config=MemoryConfig()
    )

    assert 0.0 <= score <= 1.0
    assert "semantic" in reasons


def test_compute_composite_score_with_naive_created_at():
    """A naive created_at (the default factory) still works."""
    record = MemoryRecord(content="hello", created_at=datetime(2020, 1, 1))

    score, _reasons = compute_composite_score(
        record, semantic_score=0.5, config=MemoryConfig()
    )

    assert 0.0 <= score <= 1.0
