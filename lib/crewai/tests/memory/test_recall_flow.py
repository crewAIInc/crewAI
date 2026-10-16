"""Behavioral tests for progress-aware deep recall termination."""

from datetime import datetime
from threading import Lock
from typing import Any
from unittest.mock import MagicMock

from crewai.hooks.dispatch import HookAborted
from crewai.memory.analyze import QueryAnalysis
from crewai.memory.recall_flow import RecallFlow
from crewai.memory.types import MemoryConfig, MemoryRecord
from crewai.memory.unified_memory import Memory
import pytest


RecallFixture = tuple[RecallFlow, MagicMock, MagicMock, MemoryRecord]


@pytest.fixture
def recall() -> RecallFixture:
    record = MemoryRecord(content="The project uses Python.")
    storage = MagicMock()
    storage.list_scopes.return_value = ["/"]
    storage.search.return_value = [(record, 0.1)]
    llm = MagicMock()
    llm.call.return_value = {
        "evidence_gaps": ["The project's Python version."],
        "follow_up_queries": [],
    }
    flow = RecallFlow(
        storage=storage,
        llm=llm,
        embedder=lambda texts: [[0.1 * (i + 1)] for i, _ in enumerate(texts)],
        config=MemoryConfig(
            exploration_budget=3,
            semantic_weight=1.0,
            recency_weight=0.0,
            importance_weight=0.0,
        ),
    )
    return flow, storage, llm, record


@pytest.mark.parametrize("budget", [0, 1, 3])
@pytest.mark.parametrize("scope_count", [1, 2])
@pytest.mark.parametrize("query_count", [1, 2])
def test_unchanged_plan_searches_once(
    recall: RecallFixture,
    monkeypatch: pytest.MonkeyPatch,
    budget: int,
    scope_count: int,
    query_count: int,
) -> None:
    flow, storage, llm, record = recall
    flow._config.exploration_budget = budget
    flow._config.query_analysis_threshold = 0
    storage.list_scopes.return_value = [f"/scope{i}" for i in range(scope_count)]
    monkeypatch.setattr(
        "crewai.memory.recall_flow.analyze_query",
        lambda *args: QueryAnalysis(
            recall_queries=[f"query{i}" for i in range(query_count)],
            complexity="simple",
        ),
    )

    results = flow.kickoff(inputs={"query": "project"})

    assert storage.search.call_count == query_count * scope_count
    assert llm.call.call_count == (1 if budget else 0)
    assert [match.record.id for match in results] == [record.id]
    assert results[0].score == pytest.approx(0.1)
    assert results[0].evidence_gaps == (
        llm.call.return_value["evidence_gaps"] if budget else []
    )


def test_successful_empty_search_stops(recall: RecallFixture) -> None:
    flow, storage, llm, _ = recall
    storage.search.return_value = []

    assert flow.kickoff(inputs={"query": "project"}) == []
    storage.search.assert_called_once()
    llm.call.assert_not_called()


@pytest.mark.parametrize(
    "complexity,score,expected_llm_calls",
    [
        ("simple", 0.9, 0),
        ("simple", 0.6, 0),
        ("complex", 0.6, 1),
    ],
)
def test_confidence_routing(
    recall: RecallFixture,
    monkeypatch: pytest.MonkeyPatch,
    complexity: str,
    score: float,
    expected_llm_calls: int,
) -> None:
    flow, storage, llm, record = recall
    flow._config.query_analysis_threshold = 0
    storage.search.return_value = [(record, score)]
    monkeypatch.setattr(
        "crewai.memory.recall_flow.analyze_query",
        lambda *args: QueryAnalysis(recall_queries=["project"], complexity=complexity),
    )

    results = flow.kickoff(inputs={"query": "project"})

    storage.search.assert_called_once()
    assert llm.call.call_count == expected_llm_calls
    assert results[0].score == pytest.approx(score)


@pytest.mark.parametrize(
    "field",
    [
        "embedding",
        "scope",
        "categories",
        "time_cutoff",
        "source",
        "include_private",
        "limit",
    ],
)
def test_changed_plan_searches_again_even_with_same_record_ids(
    recall: RecallFixture, field: str
) -> None:
    flow, storage, llm, record = recall
    flow.state.categories = ["engineering"]

    def explore(*args: Any, **kwargs: Any) -> dict[str, list[str]]:
        if llm.call.call_count == 1:
            if field == "embedding":
                flow.state.query_embeddings[0][1][0] = 0.2
            elif field == "scope":
                flow.state.candidate_scopes.append("/engineering")
            elif field == "categories":
                assert flow.state.categories is not None
                flow.state.categories.append("python")
            elif field == "time_cutoff":
                flow.state.time_cutoff = datetime(2000, 1, 1)
            elif field == "source":
                flow.state.source = "user"
            elif field == "include_private":
                flow.state.include_private = True
            elif field == "limit":
                flow.state.limit = 5
        return {"evidence_gaps": ["Version."], "follow_up_queries": []}

    llm.call.side_effect = explore
    results = flow.kickoff(inputs={"query": "project"})

    assert storage.search.call_count == (3 if field == "scope" else 2)
    assert llm.call.call_count == 2
    assert [match.record.id for match in results] == [record.id]


@pytest.mark.parametrize("scope_count", [1, 2])
@pytest.mark.parametrize("empty", [False, True])
def test_failed_search_retries_until_a_complete_success(
    recall: RecallFixture, scope_count: int, empty: bool
) -> None:
    flow, storage, llm, record = recall
    storage.list_scopes.return_value = [f"/scope{i}" for i in range(scope_count)]
    calls = 0
    lock = Lock()

    def search(*args: Any, **kwargs: Any) -> list[tuple[MemoryRecord, float]]:
        nonlocal calls
        with lock:
            calls += 1
            first = calls == 1
        if first:
            raise RuntimeError("Transient storage failure")
        return [] if empty else [(record, 0.1)]

    storage.search.side_effect = search
    results = flow.kickoff(inputs={"query": "project"})

    assert storage.search.call_count == scope_count * 2
    assert llm.call.call_count == (0 if empty else scope_count)
    assert [match.record.id for match in results] == ([] if empty else [record.id])


def test_persistent_failure_is_bounded_by_budget(recall: RecallFixture) -> None:
    flow, storage, llm, _ = recall
    storage.search.side_effect = RuntimeError("Unavailable")

    assert flow.kickoff(inputs={"query": "project"}) == []
    assert storage.search.call_count == 4
    llm.call.assert_not_called()


def test_reused_flow_still_searches_on_each_kickoff(recall: RecallFixture) -> None:
    flow, storage, llm, _ = recall
    flow.kickoff(inputs={"query": "project"})
    replacement = MemoryRecord(content="Updated project facts.")
    storage.search.return_value = [(replacement, 0.1)]

    results = flow.kickoff(inputs={"query": "project"})

    assert storage.search.call_count == 2
    assert llm.call.call_count == 2
    assert [match.record.id for match in results] == [replacement.id]


def test_deep_memory_recall_does_not_cache_between_requests(
    recall: RecallFixture,
) -> None:
    _, storage, llm, record = recall
    memory = Memory(
        storage=storage,
        llm=llm,
        embedder=lambda texts: [[0.1] for _ in texts],
        exploration_budget=3,
        semantic_weight=1.0,
        recency_weight=0.0,
        importance_weight=0.0,
        read_only=True,
    )
    try:
        first = memory.recall("project", depth="deep")
        replacement = MemoryRecord(content="Updated project facts.")
        storage.search.return_value = [(replacement, 0.1)]
        second = memory.recall("project", depth="deep")
    finally:
        memory.close()

    assert storage.search.call_count == 2
    assert llm.call.call_count == 2
    assert first[0].record.id == record.id
    assert second[0].record.id == replacement.id


def test_exploration_abort_is_not_swallowed(recall: RecallFixture) -> None:
    flow, storage, llm, _ = recall
    llm.call.side_effect = HookAborted(reason="Blocked by policy")

    with pytest.raises(HookAborted):
        flow.kickoff(inputs={"query": "project"})
    storage.search.assert_called_once()


def test_exploration_failure_keeps_search_results(recall: RecallFixture) -> None:
    flow, storage, llm, record = recall
    llm.call.side_effect = RuntimeError("LLM unavailable")

    results = flow.kickoff(inputs={"query": "project"})

    storage.search.assert_called_once()
    llm.call.assert_called_once()
    assert [match.record.id for match in results] == [record.id]


def test_early_stop_preserves_filters_ranking_and_limit(recall: RecallFixture) -> None:
    flow, storage, _, _ = recall
    older = MemoryRecord(content="Old", created_at=datetime(1999, 1, 1))
    private = MemoryRecord(content="Private", private=True, source="another-user")
    low = MemoryRecord(content="Low", private=True, source="user")
    high = MemoryRecord(content="High")
    storage.search.return_value = [
        (older, 0.9),
        (private, 0.8),
        (low, 0.1),
        (high, 0.2),
    ]

    results = flow.kickoff(
        inputs={
            "query": "project",
            "time_cutoff": datetime(2000, 1, 1),
            "source": "user",
            "categories": ["engineering"],
            "limit": 1,
        }
    )

    storage.search.assert_called_once_with(
        [0.1],
        scope_prefix="/",
        categories=["engineering"],
        limit=2,
        min_score=0.0,
    )
    assert [match.record.id for match in results] == [high.id]
    assert results[0].score == pytest.approx(0.2)
    assert results[0].evidence_gaps == ["The project's Python version."]
