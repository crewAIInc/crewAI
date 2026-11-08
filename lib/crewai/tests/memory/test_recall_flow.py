"""Tests for adaptive memory recall search planning."""

from typing import Any

from crewai.memory.recall_flow import RecallFlow


class _Storage:
    def __init__(self, result: list[Any] | None = None, fail: bool = False) -> None:
        self.calls = 0
        self.result = result or []
        self.fail = fail

    def search(self, *_args: Any, **_kwargs: Any) -> list[Any]:
        self.calls += 1
        if self.fail:
            raise RuntimeError("storage unavailable")
        return self.result


def _flow(storage: _Storage) -> RecallFlow:
    flow = RecallFlow(storage=storage, llm=None, embedder=None)
    flow.state.query_embeddings = [("query", [1.0, 0.0])]
    flow.state.candidate_scopes = ["/"]
    return flow


def test_re_search_skips_an_unchanged_successful_plan() -> None:
    storage = _Storage()
    flow = _flow(storage)

    flow._do_search()
    flow.state.chunk_findings = [
        {"scope": "/", "results": [], "extraction": "preserve this"}
    ]

    assert flow.re_search() == flow.state.chunk_findings
    assert storage.calls == 1


def test_re_search_retries_when_the_previous_plan_failed() -> None:
    storage = _Storage(fail=True)
    flow = _flow(storage)

    flow._do_search()
    flow.re_search()

    assert storage.calls == 2
