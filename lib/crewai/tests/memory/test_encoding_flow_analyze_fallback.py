"""Regression tests for storage-listing failures in EncodingFlow.parallel_analyze.

The scope/category listings only feed LLM *suggestion* lists. The sibling step
``parallel_find_similar`` already treats storage failures as "treat item as
new"; ``parallel_analyze`` must likewise degrade gracefully instead of crashing
the whole save pipeline.
"""

from pathlib import Path
from unittest.mock import MagicMock

from crewai.memory.encoding_flow import EncodingFlow, ItemState
from crewai.memory.types import MemoryConfig


def _flow(storage: MagicMock) -> EncodingFlow:
    return EncodingFlow(
        storage=storage,
        llm=MagicMock(),
        embedder=lambda texts: [[0.1] * 4 for _ in texts],
        config=MemoryConfig(),
    )


def _item_needing_fields() -> ItemState:
    return ItemState(
        content="user prefers dark mode",
        scope=None,
        categories=None,
        importance=None,
        embedding=[0.1, 0.2, 0.3, 0.4],
    )


def test_scope_listing_failure_still_analyzes(tmp_path: Path) -> None:
    storage = MagicMock()
    storage.list_scopes.side_effect = RuntimeError("transient storage read failure")
    storage.list_categories.return_value = {}
    flow = _flow(storage)
    item = _item_needing_fields()
    flow.state.items = [item]

    flow.parallel_analyze()

    assert item.plan is not None and item.plan.insert_new
    storage.list_categories.assert_called_once()


def test_category_listing_failure_still_analyzes(tmp_path: Path) -> None:
    storage = MagicMock()
    storage.list_scopes.return_value = ["/projects"]
    storage.list_categories.side_effect = RuntimeError("transient storage read failure")
    flow = _flow(storage)
    item = _item_needing_fields()
    flow.state.items = [item]

    flow.parallel_analyze()

    assert item.plan is not None and item.plan.insert_new


def test_both_listing_failures_still_analyze(tmp_path: Path) -> None:
    storage = MagicMock()
    storage.list_scopes.side_effect = RuntimeError("transient storage read failure")
    storage.list_categories.side_effect = RuntimeError("transient storage read failure")
    flow = _flow(storage)
    item = _item_needing_fields()
    flow.state.items = [item]

    flow.parallel_analyze()

    assert item.plan is not None and item.plan.insert_new
    assert item.resolved_scope  # defaults applied, item still persisted-able


def test_kickoff_survives_listing_failure(tmp_path: Path) -> None:
    """End-to-end: the whole save pipeline completes despite the listing failure."""
    storage = MagicMock()
    storage.search.return_value = []
    storage.list_scopes.side_effect = RuntimeError("transient storage read failure")
    storage.list_categories.side_effect = RuntimeError("transient storage read failure")
    flow = _flow(storage)

    flow.kickoff(inputs={"items": [{"content": "user prefers dark mode"}]})

    assert flow.state.items[0].result_record is not None
    storage.save.assert_called_once()
