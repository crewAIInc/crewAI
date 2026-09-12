"""Regression tests for time-filter handling in the recall flow."""

import json
from datetime import datetime

from crewai.memory.recall_flow import RecallFlow
from crewai.memory.types import MemoryRecord


class _FakeStorage:
    def __init__(self, record: MemoryRecord) -> None:
        self._record = record

    def list_scopes(self, _prefix: str) -> list[str]:
        return ["/"]

    def get_scope_info(self, _prefix: str) -> None:
        return None

    def search(
        self, *_args: object, **_kwargs: object
    ) -> list[tuple[MemoryRecord, float]]:
        return [(self._record, 0.9)]


class _FakeLLM:
    def call(self, *_args: object, **_kwargs: object) -> str:
        return json.dumps(
            {
                "keywords": [],
                "suggested_scopes": [],
                "complexity": "simple",
                "recall_queries": ["dummy query"],
                "time_filter": "2026-01-01T00:00:00Z",
            }
        )


def _recall_flow(record: MemoryRecord) -> RecallFlow:
    return RecallFlow(
        storage=_FakeStorage(record),
        llm=_FakeLLM(),
        embedder=lambda texts: [[0.1] for _ in texts],
    )


def test_tz_aware_time_filter_is_normalized_to_naive_utc():
    record = MemoryRecord(content="remembered fact", created_at=datetime(2026, 2, 1))
    flow = _recall_flow(record)

    results = flow.kickoff(inputs={"query": "what happened since january? " * 10})

    assert flow.state.time_cutoff is not None
    assert flow.state.time_cutoff.tzinfo is None
    assert [match.record.content for match in results] == ["remembered fact"]
