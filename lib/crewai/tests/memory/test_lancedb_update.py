"""Regression tests for preserving stored memories when an update is invalid."""

from datetime import datetime
from pathlib import Path

from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.memory.types import MemoryRecord
import pytest


@pytest.mark.parametrize(
    "invalid_value",
    [datetime(2026, 1, 1), {"tag"}],
    ids=["datetime", "set"],
)
def test_update_serialization_failure_preserves_existing_record(
    tmp_path: Path, invalid_value: object
) -> None:
    storage = LanceDBStorage(path=tmp_path / "memory", vector_dim=2, compact_every=0)
    original = MemoryRecord(
        content="original content",
        scope="/notes",
        metadata={"version": 1},
        embedding=[1.0, 0.0],
    )
    storage.save([original])
    replacement = original.model_copy(
        update={"content": "replacement content", "metadata": {"value": invalid_value}}
    )

    with pytest.raises(TypeError, match="not JSON serializable"):
        storage.update(replacement)

    assert storage.get_record(original.id) == original
    assert storage.count() == 1


def test_update_replaces_existing_record(tmp_path: Path) -> None:
    storage = LanceDBStorage(path=tmp_path / "memory", vector_dim=2, compact_every=0)
    original = MemoryRecord(content="original content", embedding=[1.0, 0.0])
    storage.save([original])
    replacement = original.model_copy(
        update={"content": "replacement content", "metadata": {"version": 2}}
    )

    storage.update(replacement)

    assert storage.get_record(original.id) == replacement
    assert storage.count() == 1
