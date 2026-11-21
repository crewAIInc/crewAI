"""Deep recall must turn retrieved clues into bounded, grounded searches."""

from datetime import datetime
import json
from typing import Any
from unittest.mock import MagicMock

from crewai.hooks.dispatch import HookAborted
from crewai.memory.recall_flow import RecallFlow, _RecallRefinement
from crewai.memory.types import MemoryConfig, MemoryRecord
from crewai.memory.unified_memory import Memory
import pytest


@pytest.fixture
def recall() -> tuple[RecallFlow, MagicMock, MagicMock, MagicMock]:
    storage = MagicMock()
    storage.list_scopes.return_value = ["/project"]
    storage.search.side_effect = [
        [(MemoryRecord(id="clue", content="The project codename is ORION."), 0.1)],
        [(MemoryRecord(id="answer", content="ORION launches on October 10."), 0.2)],
    ]
    llm = MagicMock()
    llm.supports_function_calling.return_value = True
    llm.call.side_effect = [
        {
            "evidence_gaps": ["缺少发布日期"],
            "follow_up_queries": ["ORION release date"],
        },
        {"evidence_gaps": [], "follow_up_queries": []},
    ]
    embedder = MagicMock(
        side_effect=lambda texts: [[0.2 if "ORION" in text else 0.1] for text in texts]
    )
    flow = RecallFlow(
        storage,
        llm,
        embedder,
        MemoryConfig(
            exploration_budget=3,
            semantic_weight=1,
            recency_weight=0,
            importance_weight=0,
        ),
    )
    return flow, storage, llm, embedder


RecallFixture = tuple[RecallFlow, MagicMock, MagicMock, MagicMock]


def test_two_hop_recall_keeps_both_records_and_resolves_gaps(
    recall: RecallFixture,
) -> None:
    flow, storage, llm, embedder = recall
    results = flow.kickoff(inputs={"query": "When does our project launch?"})

    assert [call.args[0] for call in storage.search.call_args_list] == [[0.1], [0.2]]
    assert embedder.call_args_list[1].args[0] == ["ORION release date"]
    assert [m.record.id for m in results] == ["answer", "clue"]
    assert results[0].evidence_gaps == []
    assert llm.call.call_count == 2
    assert "ORION" in llm.call.call_args_list[0].args[0][1]["content"]
    assert "October 10" in llm.call.call_args_list[1].args[0][1]["content"]


@pytest.mark.parametrize("budget,searches", [(0, 1), (1, 2), (2, 3)])
def test_refinement_obeys_budget(
    recall: RecallFixture, budget: int, searches: int
) -> None:
    flow, storage, llm, embedder = recall
    flow._config.exploration_budget = budget
    record = MemoryRecord(content="Clue")
    storage.search.side_effect = None
    storage.search.return_value = [(record, 0.1)]
    llm.call.side_effect = lambda *args, **kwargs: {
        "evidence_gaps": ["Need more detail"],
        "follow_up_queries": [f"Detail {llm.call.call_count}"],
    }
    embedder.side_effect = lambda texts: [[float(embedder.call_count)] for _ in texts]

    assert len(flow.kickoff(inputs={"query": "project"})) == 1
    assert storage.search.call_count == searches
    assert llm.call.call_count == budget


@pytest.mark.parametrize(
    "output",
    [
        "Nothing is missing.",
        "not JSON",
        None,
        {"evidence_gaps": "wrong type", "follow_up_queries": ["ORION"]},
        {"evidence_gaps": [], "follow_up_queries": ["ORION"]},
        {"evidence_gaps": ["Date"], "follow_up_queries": ["  PROJECT  ", "", "   "]},
        {"evidence_gaps": ["Date"], "follow_up_queries": ["x" * 501]},
        {"evidence_gaps": ["Date"], "follow_up_queries": ["a", "b", "c", "d"]},
        {"evidence_gaps": ["Date"], "follow_up_queries": ["ORION"], "scope": "/secret"},
    ],
)
def test_invalid_or_unactionable_refinement_stops(
    recall: RecallFixture, output: Any
) -> None:
    flow, storage, llm, embedder = recall
    llm.call.side_effect = None
    llm.call.return_value = output

    results = flow.kickoff(inputs={"query": "project"})

    assert [m.record.id for m in results] == ["clue"]
    storage.search.assert_called_once()
    embedder.assert_called_once()


def test_json_fallback_and_non_english_gap(recall: RecallFixture) -> None:
    flow, _, llm, _ = recall
    flow._config.exploration_budget = 1
    llm.supports_function_calling.return_value = False
    llm.call.side_effect = [
        json.dumps(
            {
                "evidence_gaps": ["缺少发布日期"],
                "follow_up_queries": ["ORION release date"],
            }
        )
    ]

    results = flow.kickoff(inputs={"query": "project"})

    assert {m.record.id for m in results} == {"clue", "answer"}
    assert results[0].evidence_gaps == ["缺少发布日期"]
    assert "response_model" not in llm.call.call_args.kwargs


@pytest.mark.parametrize("failure", [[], RuntimeError("Storage unavailable")])
def test_follow_up_failure_keeps_first_hop(recall: RecallFixture, failure: Any) -> None:
    flow, storage, _, _ = recall
    flow._config.exploration_budget = 1
    clue = MemoryRecord(id="clue", content="Codename: ORION")
    storage.search.side_effect = [[(clue, 0.1)], failure]

    results = flow.kickoff(inputs={"query": "project"})

    assert [m.record.id for m in results] == ["clue"]
    assert storage.search.call_count == 2


@pytest.mark.parametrize(
    "vectors", [[], [[]], [[0.1]], [[float("nan")]], [[float("inf")]], [[1, 2]]]
)
def test_unusable_or_repeated_vectors_stop(recall: RecallFixture, vectors: Any) -> None:
    flow, storage, _, embedder = recall
    embedder.side_effect = [[[0.1]], vectors]

    results = flow.kickoff(inputs={"query": "project"})

    assert [m.record.id for m in results] == ["clue"]
    storage.search.assert_called_once()


def test_query_deduplication_across_rounds(recall: RecallFixture) -> None:
    flow, storage, llm, embedder = recall
    llm.call.side_effect = [
        {
            "evidence_gaps": ["Date"],
            "follow_up_queries": [" ORION  date ", "orion date", "project"],
        },
        {"evidence_gaps": ["Date"], "follow_up_queries": ["ORION DATE"]},
    ]

    flow.kickoff(inputs={"query": "project"})

    assert embedder.call_args_list[1].args[0] == ["ORION date"]
    assert storage.search.call_count == 2
    assert llm.call.call_count == 2


def test_filters_apply_to_follow_up_results(recall: RecallFixture) -> None:
    flow, storage, _, _ = recall
    flow._config.exploration_budget = 1
    clue = MemoryRecord(
        id="clue", content="Codename: ORION", private=True, source="user"
    )
    answer = MemoryRecord(id="answer", content="October 10")
    hidden = MemoryRecord(content="Secret", private=True, source="other")
    old = MemoryRecord(content="Old", created_at=datetime(1999, 1, 1))
    storage.search.side_effect = [
        [(clue, 0.1)],
        [(hidden, 0.9), (old, 0.9), (answer, 0.2)],
    ]

    results = flow.kickoff(
        inputs={
            "query": "project",
            "scope": "/project",
            "categories": ["release"],
            "time_cutoff": datetime(2000, 1, 1),
            "source": "user",
            "limit": 2,
        }
    )

    assert {m.record.id for m in results} == {"clue", "answer"}
    assert (
        storage.search.call_args_list[0].kwargs
        == storage.search.call_args_list[1].kwargs
    )
    assert storage.search.call_args.kwargs == {
        "scope_prefix": "/project",
        "categories": ["release"],
        "limit": 4,
        "min_score": 0.0,
    }


@pytest.mark.parametrize("target", ["llm", "embedder"])
@pytest.mark.parametrize("abort", [False, True])
def test_refinement_exceptions(recall: RecallFixture, target: str, abort: bool) -> None:
    flow, storage, llm, embedder = recall
    error = HookAborted(reason="policy") if abort else RuntimeError("Unavailable")
    if target == "llm":
        llm.call.side_effect = error
    else:
        embedder.side_effect = [[[0.1]], error]

    if abort:
        with pytest.raises(HookAborted):
            flow.kickoff(inputs={"query": "project"})
    else:
        assert flow.kickoff(inputs={"query": "project"})[0].record.id == "clue"
    storage.search.assert_called_once()


def test_native_response_batches_queries_and_keeps_best_duplicate_score(
    recall: RecallFixture,
) -> None:
    flow, storage, llm, embedder = recall
    flow._config.exploration_budget = 1
    clue = MemoryRecord(id="clue", content="Codename: ORION")
    storage.search.side_effect = lambda vector, **kwargs: [(clue, vector[0])]
    llm.call.side_effect = [
        _RecallRefinement(
            evidence_gaps=["Date"],
            follow_up_queries=["ORION date", "ORION schedule", "ORION launch"],
        )
    ]
    embedder.side_effect = [[[0.1]], [[0.2], [0.3], [0.3]]]

    results = flow.kickoff(inputs={"query": "project"})

    assert embedder.call_args_list[1].args[0] == [
        "ORION date",
        "ORION schedule",
        "ORION launch",
    ]
    assert storage.search.call_count == 3
    assert len(results) == 1
    assert results[0].score == pytest.approx(0.3)
    assert llm.call.call_args.kwargs["response_model"] is _RecallRefinement


def test_refinement_context_is_bounded_and_deduplicated(recall: RecallFixture) -> None:
    flow, storage, llm, _ = recall
    record = MemoryRecord(id="clue", content="ORION " * 1000)
    storage.list_scopes.return_value = ["/a", "/b"]
    storage.search.side_effect = None
    storage.search.return_value = [(record, 0.1)] * 2 + [
        (MemoryRecord(content=str(i) * 1000), 0.1) for i in range(20)
    ]
    llm.call.side_effect = [{"evidence_gaps": [], "follow_up_queries": []}]

    flow.kickoff(inputs={"query": "project"})

    llm.call.assert_called_once()
    context = json.loads(llm.call.call_args.args[0][1]["content"])
    assert len(context["memory_excerpts"]) == 10
    assert all(len(text) <= 500 for text in context["memory_excerpts"])
    assert len([text for text in context["memory_excerpts"] if "ORION" in text]) == 1


def test_reused_flow_resets_evidence_and_query_history(recall: RecallFixture) -> None:
    flow, storage, llm, embedder = recall
    flow._config.exploration_budget = 1
    storage.search.side_effect = None
    storage.search.return_value = [
        (MemoryRecord(id="old", content="Codename: ORION"), 0.1)
    ]
    llm.call.side_effect = None
    llm.call.return_value = {
        "evidence_gaps": ["Date"],
        "follow_up_queries": ["ORION date"],
    }
    flow.kickoff(inputs={"query": "project"})
    storage.search.return_value = [
        (MemoryRecord(id="new", content="Codename: ORION"), 0.1)
    ]

    results = flow.kickoff(inputs={"query": "project"})

    assert [m.record.id for m in results] == ["new"]
    assert storage.search.call_count == 4
    assert embedder.call_count == 4
    assert results[0].evidence_gaps == ["Date"]


@pytest.mark.parametrize(
    "budget,expected_ids", [(0, {"clue"}), (1, {"clue", "answer"})]
)
def test_public_deep_recall_retrieves_follow_up_evidence(
    recall: RecallFixture, budget: int, expected_ids: set[str]
) -> None:
    _, storage, llm, embedder = recall
    memory = Memory(
        storage=storage,
        llm=llm,
        embedder=embedder,
        exploration_budget=budget,
        semantic_weight=1,
        recency_weight=0,
        importance_weight=0,
        read_only=True,
    )
    try:
        results = memory.recall("project", depth="deep")
    finally:
        memory.close()

    assert {m.record.id for m in results} == expected_ids
    assert storage.search.call_count == 1 + budget
    assert llm.call.call_count == budget
