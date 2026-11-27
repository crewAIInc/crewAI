"""A closer private record must not hide visible shallow matches."""

from unittest.mock import MagicMock

from crewai.memory.unified_memory import Memory, MemoryRecord


def _record(content: str, private: bool, source: str | None) -> MemoryRecord:
    return MemoryRecord(
        content=content,
        scope="/project",
        categories=[],
        private=private,
        source=source,
        embedding=[0.1] * 1536,
    )


def test_shallow_recall_skips_private_match():
    embedder = MagicMock()
    embedder.side_effect = lambda texts: [[0.1] * 1536 for _ in texts]
    storage = MagicMock()
    private = _record("private", True, "owner")
    public = _record("public", False, None)
    storage.search.return_value = [(private, 0.9), (public, 0.8)]

    m = Memory(storage=storage, llm=MagicMock(), embedder=embedder)
    matches = m.recall("query", scope="/project", limit=1, depth="shallow", source="other")

    assert [match.record.content for match in matches] == ["public"]
    assert storage.search.call_args.kwargs["limit"] == 2
