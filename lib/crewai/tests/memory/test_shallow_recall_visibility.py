from pathlib import Path
from unittest.mock import MagicMock

from crewai.memory.types import MemoryRecord
from crewai.memory.unified_memory import Memory
import pytest


@pytest.mark.parametrize(
    ("source", "include_private", "expected"),
    [
        (None, False, ["public"]),
        ("owner", False, ["private"]),
        (None, True, ["private"]),
    ],
)
def test_shallow_recall_counts_only_visible_records(
    tmp_path: Path, source: str | None, include_private: bool, expected: list[str]
) -> None:
    memory = Memory(
        storage=str(tmp_path),
        llm=MagicMock(),
        embedder=lambda texts: [[1.0, 0.0] for _ in texts],
    )
    memory._storage.save(
        [
            MemoryRecord(
                content="private", embedding=[1.0, 0.0], private=True, source="owner"
            ),
            MemoryRecord(content="public", embedding=[0.9, 0.1]),
        ]
    )

    matches = memory.recall(
        "travel",
        depth="shallow",
        limit=1,
        source=source,
        include_private=include_private,
    )

    assert [match.record.content for match in matches] == expected


def test_shallow_recall_stops_when_no_records_are_visible(tmp_path: Path) -> None:
    memory = Memory(
        storage=str(tmp_path),
        llm=MagicMock(),
        embedder=lambda texts: [[1.0, 0.0] for _ in texts],
    )
    memory._storage.save(
        [MemoryRecord(content="private", embedding=[1.0, 0.0], private=True)]
    )

    assert memory.recall("travel", depth="shallow", limit=1) == []


def test_deep_recall_counts_only_visible_records(tmp_path: Path) -> None:
    memory = Memory(
        storage=str(tmp_path),
        llm=MagicMock(),
        embedder=lambda texts: [[1.0, 0.0] for _ in texts],
    )
    memory._storage.save(
        [
            MemoryRecord(
                content=f"private {i}",
                embedding=[1.0, 0.0],
                private=True,
                source="owner",
            )
            for i in range(3)
        ]
        + [MemoryRecord(content="public", embedding=[0.9, 0.1])]
    )

    matches = memory.recall("travel", limit=1)

    assert [match.record.content for match in matches] == ["public"]
