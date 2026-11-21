"""Consolidation must not strip a record's privacy flag or provenance.

``EncodingFlow.execute_plans()`` rebuilds the updated ``MemoryRecord`` field by
field and omitted ``private`` and ``source``, so both fell back to their
defaults (``False`` and ``None``). A private memory that consolidation merged
into came back public, and the record lost the source that identified where it
came from. The insert branch a few lines above sets both explicitly from the
resolved item, so the omission was an asymmetry rather than a design choice.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

from crewai.memory.analyze import ConsolidationAction, ConsolidationPlan
from crewai.memory.encoding_flow import EncodingFlow, ItemState
from crewai.memory.types import MemoryConfig, MemoryRecord

DIM = 4
CREATED = datetime(2026, 1, 1, 12, 0, 0)


def _existing(**overrides: object) -> MemoryRecord:
    fields: dict[str, object] = {
        "id": "rec-1",
        "content": "original secret",
        "scope": "/t1/agent-a",
        "categories": ["decision"],
        "metadata": {"kind": "note"},
        "importance": 0.7,
        "created_at": CREATED,
        "embedding": [0.1] * DIM,
        "source": "agent-a",
        "private": True,
    }
    fields.update(overrides)
    return MemoryRecord(**fields)  # type: ignore[arg-type]


def _flow_with_update(existing: MemoryRecord) -> tuple[EncodingFlow, MagicMock]:
    storage = MagicMock()
    storage.search.return_value = []
    flow = EncodingFlow(
        storage=storage,
        llm=MagicMock(),
        embedder=lambda texts: [[0.2] * DIM for _ in texts],
        config=MemoryConfig(),
    )
    item = ItemState(content="more secret info", embedding=[0.2] * DIM)
    item.similar_records = [existing]
    item.plan = ConsolidationPlan(
        actions=[
            ConsolidationAction(
                action="update", record_id=existing.id, new_content="merged secret"
            )
        ],
        insert_new=False,
    )
    flow.state.items = [item]
    return flow, storage


def _updated_record(storage: MagicMock) -> MemoryRecord:
    storage.update.assert_called_once()
    return storage.update.call_args[0][0]


def test_private_record_stays_private_after_consolidation() -> None:
    flow, storage = _flow_with_update(_existing(private=True))

    flow.execute_plans()

    assert _updated_record(storage).private is True


def test_source_is_preserved_after_consolidation() -> None:
    flow, storage = _flow_with_update(_existing(source="agent-a"))

    flow.execute_plans()

    assert _updated_record(storage).source == "agent-a"


def test_public_record_stays_public() -> None:
    flow, storage = _flow_with_update(_existing(private=False, source=None))

    flow.execute_plans()

    updated = _updated_record(storage)
    assert updated.private is False
    assert updated.source is None


def test_remaining_fields_are_still_preserved() -> None:
    flow, storage = _flow_with_update(_existing())

    flow.execute_plans()

    updated = _updated_record(storage)
    assert updated.id == "rec-1"
    assert updated.content == "merged secret"
    assert updated.scope == "/t1/agent-a"
    assert updated.categories == ["decision"]
    assert updated.metadata == {"kind": "note"}
    assert updated.importance == 0.7
    assert updated.created_at == CREATED
    assert updated.embedding == [0.2] * DIM
    assert updated.last_accessed > CREATED + timedelta(days=1)


def test_result_record_carries_the_same_fields() -> None:
    flow, storage = _flow_with_update(_existing())

    flow.execute_plans()

    result = flow.state.items[0].result_record
    assert result is not None
    assert result.private is True
    assert result.source == "agent-a"
