"""Tests for EncodingFlow error handling and thread pool management."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crewai.memory.encoding_flow import EncodingFlow, EncodingState, ItemState
from crewai.memory.storage.lancedb_storage import LanceDBStorage
from crewai.memory.types import MemoryConfig


@pytest.fixture
def mock_llm():
    """Mock LLM that can be configured to fail."""
    llm = MagicMock()
    llm.call.return_value = '{"suggested_scope": "/test", "categories": ["test"], "importance": 0.5}'
    return llm


@pytest.fixture
def mock_embedder():
    """Mock embedder that returns fixed vectors."""
    embedder = MagicMock()
    embedder.embed_texts = MagicMock(return_value=[[0.1, 0.2, 0.3, 0.4]])
    return embedder


@pytest.fixture
def storage(tmp_path: Path):
    """Create a temporary LanceDB storage."""
    return LanceDBStorage(path=str(tmp_path / "test_mem"), vector_dim=4)


def _set_flow_state(flow, state):
    """Set flow state directly, bypassing the read-only property."""
    flow._state = state


def test_encoding_flow_handles_llm_failure(storage, mock_llm, mock_embedder):
    """Test that encoding flow continues if one LLM call fails."""
    mock_llm.call.side_effect = [
        Exception("LLM timeout"),
        '{"suggested_scope": "/test", "categories": ["test"], "importance": 0.5}',
    ]

    flow = EncodingFlow(
        storage=storage,
        llm=mock_llm,
        embedder=mock_embedder,
        config=MemoryConfig(),
    )

    _set_flow_state(flow, EncodingState(
        items=[
            ItemState(content="item 1"),
            ItemState(content="item 2"),
        ]
    ))

    with patch("crewai.memory.encoding_flow.embed_texts") as mock_embed:
        mock_embed.return_value = [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]]

        flow.batch_embed()
        flow.intra_batch_dedup()
        flow.parallel_find_similar()

        # This should not raise even though first LLM call fails
        flow.parallel_analyze()

    # First item should have fallen back to defaults
    assert flow.state.items[0].resolved_scope == "/"
    assert flow.state.items[0].plan is not None
    assert flow.state.items[0].plan.insert_new is True


def test_encoding_flow_thread_pool_waits_for_completion(storage, mock_llm, mock_embedder):
    """Test that thread pool waits for all futures before shutdown."""
    import time

    def slow_llm_call(*args, **kwargs):
        time.sleep(0.1)
        return '{"suggested_scope": "/test", "categories": ["test"], "importance": 0.5}'

    mock_llm.call.side_effect = slow_llm_call

    flow = EncodingFlow(
        storage=storage,
        llm=mock_llm,
        embedder=mock_embedder,
        config=MemoryConfig(),
    )

    _set_flow_state(flow, EncodingState(
        items=[ItemState(content="test item")]
    ))

    with patch("crewai.memory.encoding_flow.embed_texts") as mock_embed:
        mock_embed.return_value = [[0.1, 0.2, 0.3, 0.4]]

        flow.batch_embed()
        flow.intra_batch_dedup()
        flow.parallel_find_similar()

        start = time.time()
        flow.parallel_analyze()
        duration = time.time() - start

        # Should have waited at least 0.1 seconds for the slow call
        assert duration >= 0.1

        assert flow.state.items[0].plan is not None


def test_encoding_flow_intra_batch_dedup(storage, mock_llm, mock_embedder):
    """Test that near-duplicate items within a batch are dropped."""
    flow = EncodingFlow(
        storage=storage,
        llm=mock_llm,
        embedder=mock_embedder,
        config=MemoryConfig(batch_dedup_threshold=0.99),
    )

    _set_flow_state(flow, EncodingState(
        items=[
            ItemState(content="item A"),
            ItemState(content="item B (identical embedding)"),
        ]
    ))

    with patch("crewai.memory.encoding_flow.embed_texts") as mock_embed:
        # Return identical embeddings so dedup kicks in
        mock_embed.return_value = [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]

        flow.batch_embed()
        flow.intra_batch_dedup()

    # Second item should be dropped as a near-duplicate
    assert flow.state.items[0].dropped is False
    assert flow.state.items[1].dropped is True
    assert flow.state.items_dropped_dedup == 1


def test_encoding_flow_no_dedup_for_different_items(storage, mock_llm, mock_embedder):
    """Test that sufficiently different items are NOT dropped."""
    flow = EncodingFlow(
        storage=storage,
        llm=mock_llm,
        embedder=mock_embedder,
        config=MemoryConfig(batch_dedup_threshold=0.99),
    )

    _set_flow_state(flow, EncodingState(
        items=[
            ItemState(content="item A"),
            ItemState(content="item B"),
        ]
    ))

    with patch("crewai.memory.encoding_flow.embed_texts") as mock_embed:
        # Return orthogonal embeddings
        mock_embed.return_value = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]

        flow.batch_embed()
        flow.intra_batch_dedup()

    assert flow.state.items[0].dropped is False
    assert flow.state.items[1].dropped is False
    assert flow.state.items_dropped_dedup == 0


def test_encoding_flow_group_a_fast_path(storage, mock_llm, mock_embedder):
    """Test Group A: all fields provided + no similar records = fast insert, 0 LLM calls."""
    flow = EncodingFlow(
        storage=storage,
        llm=mock_llm,
        embedder=mock_embedder,
        config=MemoryConfig(),
    )

    _set_flow_state(flow, EncodingState(
        items=[
            ItemState(
                content="fully specified item",
                scope="/project",
                categories=["decision"],
                importance=0.8,
            )
        ]
    ))

    with patch("crewai.memory.encoding_flow.embed_texts") as mock_embed:
        mock_embed.return_value = [[0.1, 0.2, 0.3, 0.4]]

        flow.batch_embed()
        flow.intra_batch_dedup()
        flow.parallel_find_similar()
        flow.parallel_analyze()

    item = flow.state.items[0]
    assert item.resolved_scope == "/project"
    assert item.resolved_categories == ["decision"]
    assert item.resolved_importance == 0.8
    assert item.plan is not None
    assert item.plan.insert_new is True
    # No LLM calls should have been made for Group A
    mock_llm.call.assert_not_called()
