"""Exercise consolidation scope and per-item authorization."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from crewai.memory.analyze import ConsolidationAction, ConsolidationPlan
from crewai.memory.encoding_flow import EncodingFlow, ItemState
from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.memory.types import MemoryRecord


def _embed(texts: list[str]) -> list[list[float]]:
    return [[1.0, 0.0] for _ in texts]


@pytest.mark.parametrize(
    ("root_scope", "scope"), [(None, "/team/alpha"), ("/team", "alpha")]
)
def test_consolidation_does_not_send_sibling_records_to_the_model(
    tmp_path: Path, root_scope: str | None, scope: str
) -> None:
    storage = LanceDBStorage(path=tmp_path, compact_every=0)
    descendant = MemoryRecord(
        content="allowed", scope="/team/alpha/notes", embedding=[1.0, 0.0]
    )
    sibling = MemoryRecord(
        content="outside", scope="/team/alphabet", embedding=[1.0, 0.0]
    )
    storage.save([descendant, sibling])
    flow = EncodingFlow(storage, Mock(), _embed)

    with patch(
        "crewai.memory.encoding_flow.analyze_for_consolidation",
        return_value=ConsolidationPlan(insert_new=False),
    ) as analyze:
        flow.kickoff(
            inputs={
                "items": [
                    {
                        "content": "another fact",
                        "scope": scope,
                        "root_scope": root_scope,
                        "categories": [],
                        "importance": 0.5,
                    }
                ]
            }
        )

    analyze.assert_called_once()
    assert [record.id for record in analyze.call_args.args[1]] == [descendant.id]
    result = flow.state.items[0].result_record
    assert result is not None and result.id == descendant.id


def test_execute_plans_rejects_ids_outside_each_items_candidates(tmp_path: Path) -> None:
    storage = LanceDBStorage(path=tmp_path, compact_every=0)
    permitted = MemoryRecord(
        content="allowed update", scope="/team/alpha", embedding=[1.0, 0.0]
    )
    sibling = MemoryRecord(
        content="outside", scope="/team/alphabet", embedding=[1.0, 0.0]
    )
    unlisted = MemoryRecord(
        content="not retrieved", scope="/team/alpha", embedding=[1.0, 0.0]
    )
    storage.save([permitted, sibling, unlisted])
    flow = EncodingFlow(storage, Mock(), _embed)
    owner = ItemState(
        scope="/team/alpha",
        similar_records=[permitted, sibling],
        plan=ConsolidationPlan(
            insert_new=False,
            actions=[
                ConsolidationAction(action="delete", record_id=sibling.id),
                ConsolidationAction(action="delete", record_id=unlisted.id),
                ConsolidationAction(
                    action="update", record_id=permitted.id, new_content="updated"
                ),
            ],
        ),
    )
    other = ItemState(
        scope="/team/alphabet",
        similar_records=[sibling],
        plan=ConsolidationPlan(
            insert_new=False,
            actions=[
                ConsolidationAction(
                    action="update", record_id=permitted.id, new_content="unexpected"
                )
            ],
        ),
    )
    flow.state.items = [owner, other]

    flow.execute_plans()

    assert storage.get_record(sibling.id) is not None
    assert storage.get_record(unlisted.id) is not None
    updated = storage.get_record(permitted.id)
    assert updated is not None and updated.content == "updated"
    assert owner.result_record is not None and owner.result_record.id == permitted.id
    assert other.result_record is not None and other.result_record.id == sibling.id
