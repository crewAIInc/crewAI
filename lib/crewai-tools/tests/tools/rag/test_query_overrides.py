"""Per-query RAG options must override configured defaults, including zero."""

from typing import Any

import pytest
from crewai_tools.tools.rag.rag_tool import Adapter, RagTool
from pydantic import Field


class RecordingAdapter(Adapter):
    calls: list[tuple[str, float | None, int | None]] = Field(default_factory=list)

    def query(
        self,
        question: str,
        similarity_threshold: float | None = None,
        limit: int | None = None,
    ) -> str:
        self.calls.append((question, similarity_threshold, limit))
        return "matching document"

    def add(self, *args: Any, **kwargs: Any) -> None:
        pass


@pytest.mark.parametrize("threshold", [None, 0.0, 0.2])
@pytest.mark.parametrize("limit", [None, 2])
def test_query_forwards_explicit_options_and_falls_back_only_for_none(
    threshold: float | None, limit: int | None
) -> None:
    adapter = RecordingAdapter()
    tool = RagTool(adapter=adapter, similarity_threshold=0.8, limit=7)

    result = tool._run("question", similarity_threshold=threshold, limit=limit)

    assert result == "Relevant Content:\nmatching document"
    assert adapter.calls == [
        (
            "question",
            0.8 if threshold is None else threshold,
            7 if limit is None else limit,
        )
    ]
