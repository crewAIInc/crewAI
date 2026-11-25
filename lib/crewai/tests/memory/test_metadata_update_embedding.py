"""Metadata edits must not regenerate or discard persisted embeddings."""

from pathlib import Path
from unittest.mock import Mock

import pytest
from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.memory.types import MemoryRecord
from crewai.memory.unified_memory import Memory


@pytest.mark.parametrize(
    "change_content", [False, True], ids=["metadata-only", "content-control"]
)
def test_update_embeds_only_when_content_changes(
    tmp_path: Path, change_content: bool
) -> None:
    storage = LanceDBStorage(path=tmp_path / "memory", vector_dim=4, compact_every=0)
    original_vector = [0.1, 0.2, 0.3, 0.4]
    new_vector = [0.4, 0.3, 0.2, 0.1]
    record = MemoryRecord(
        content="original fact",
        scope="/",
        embedding=original_vector,
        metadata={"version": 1},
    )
    storage.save([record])
    embedder = Mock(return_value=[new_vector])
    memory = Memory(storage=storage, llm=Mock(), embedder=embedder)

    updated = memory.update(
        record.id,
        content="revised fact" if change_content else None,
        metadata={"version": 2},
    )
    stored = storage.get_record(record.id)

    assert stored is not None
    assert updated.id == stored.id == record.id
    assert (
        updated.content
        == stored.content
        == ("revised fact" if change_content else "original fact")
    )
    assert updated.metadata == stored.metadata == {"version": 2}
    assert stored.embedding == pytest.approx(
        new_vector if change_content else original_vector
    )
    if change_content:
        embedder.assert_called_once_with(["revised fact"])
    else:
        embedder.assert_not_called()
