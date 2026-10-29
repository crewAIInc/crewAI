"""Deep recall must keep LLM-selected scopes within the caller's scope."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from crewai.memory.analyze import QueryAnalysis
from crewai.memory.recall_flow import RecallFlow
from crewai.memory.unified_memory import Memory


@pytest.mark.parametrize(
    ("requested_scope", "allowed_scope", "other_scope", "suggested_scopes"),
    [
        (None, "/crew/a/inner", "/crew/b/inner", ["/crew/b/inner"]),
        (None, "/crew/a/inner", "/crew/a-neighbor", ["/crew/a-neighbor"]),
        (None, "/crew/a/inner", "/crew/b/inner", ["/"]),
        ("/inner", "/crew/a/inner", "/crew/a/other", ["/crew/a/other"]),
        (
            None,
            "/crew/a/inner",
            "/crew/b/inner",
            ["/crew/b/inner", "/crew/a/inner"],
        ),
    ],
)
def test_deep_recall_confines_llm_suggested_scopes(
    tmp_path: Path,
    requested_scope: str | None,
    allowed_scope: str,
    other_scope: str,
    suggested_scopes: list[str],
) -> None:
    """A model suggestion must not widen root_scope or an explicit scope."""
    embedder = MagicMock(side_effect=lambda texts: [[0.1] * 4 for _ in texts])
    storage = str(tmp_path / "memory")
    global_memory = Memory(storage=storage, llm=MagicMock(), embedder=embedder)
    global_memory.remember(
        "Allowed record", scope=allowed_scope, categories=["test"], importance=0.5
    )
    global_memory.remember(
        "Other record", scope=other_scope, categories=["test"], importance=0.5
    )

    llm = MagicMock()
    llm.supports_function_calling.return_value = False
    llm.call.return_value = QueryAnalysis(
        suggested_scopes=suggested_scopes,
        recall_queries=["record"],
    )
    scoped_memory = Memory(
        storage=storage,
        llm=llm,
        embedder=embedder,
        root_scope="/crew/a",
        query_analysis_threshold=0,
        exploration_budget=0,
    )

    matches = scoped_memory.recall("record", scope=requested_scope, depth="deep")

    assert [match.record.scope for match in matches] == [allowed_scope]


def test_deep_recall_confines_storage_listed_scopes() -> None:
    """An inconsistent backend scope listing cannot widen the caller's scope."""
    storage = MagicMock()
    storage.list_scopes.return_value = ["/crew/b", "/crew/a/inner"]
    flow = RecallFlow(storage=storage, llm=MagicMock(), embedder=MagicMock())
    flow.state.scope = "/crew/a"
    flow.state.query_analysis = QueryAnalysis()

    assert flow.filter_and_chunk() == ["/crew/a/inner"]

    storage.list_scopes.return_value = ["/crew/b"]
    assert flow.filter_and_chunk() == ["/crew/a"]
