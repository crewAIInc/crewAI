"""Tests for EncodingFlow.intra_batch_dedup (vectorized cosine dedup).

The vectorized implementation must reproduce the exact drop decisions of the
original scalar O(n^2) algorithm, which is retained as ``_dedup_scalar``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from crewai.memory.encoding_flow import EncodingFlow, ItemState
from crewai.memory.types import MemoryConfig


def _make_flow(threshold: float = 0.98) -> EncodingFlow:
    config = MemoryConfig()
    config.batch_dedup_threshold = threshold
    return EncodingFlow(
        storage=MagicMock(), llm=MagicMock(), embedder=MagicMock(), config=config
    )


def _run_vectorized(items: list[ItemState], threshold: float = 0.98) -> EncodingFlow:
    flow = _make_flow(threshold)
    flow.state.items = items
    flow.intra_batch_dedup()
    return flow


def test_drops_identical_keeps_distinct() -> None:
    items = [
        ItemState(content="a", embedding=[0.5] * 8),
        ItemState(content="b", embedding=[0.5] * 8),  # identical to a -> dropped
        ItemState(content="c", embedding=[1.0] + [0.0] * 7),  # orthogonal -> kept
    ]
    flow = _run_vectorized(items)
    assert [it.dropped for it in items] == [False, True, False]
    assert flow.state.items_dropped_dedup == 1


def test_first_occurrence_wins() -> None:
    items = [ItemState(content=str(i), embedding=[0.5] * 8) for i in range(4)]
    flow = _run_vectorized(items)
    # Only the first survives; the rest are dropped against it.
    assert [it.dropped for it in items] == [False, True, True, True]
    assert flow.state.items_dropped_dedup == 3


def test_items_without_embeddings_never_dropped() -> None:
    items = [
        ItemState(content="a", embedding=[0.5] * 8),
        ItemState(content="no-emb", embedding=[]),  # never participates
        ItemState(content="b", embedding=[0.5] * 8),  # dup of a
    ]
    flow = _run_vectorized(items)
    assert [it.dropped for it in items] == [False, False, True]
    assert flow.state.items_dropped_dedup == 1


def test_pre_dropped_item_is_skipped() -> None:
    """A pre-dropped item must neither be re-counted nor suppress others."""
    a = ItemState(content="a", embedding=[0.5] * 8)
    a.dropped = True  # already dropped upstream
    items = [
        a,
        ItemState(content="b", embedding=[0.5] * 8),  # same vector as the dropped a
        ItemState(content="c", embedding=[0.5] * 8),  # dup of b
    ]
    flow = _run_vectorized(items)
    # 'a' stays dropped (not re-counted); 'b' becomes the surviving original;
    # 'c' is dropped against 'b'.
    assert items[0].dropped is True
    assert items[1].dropped is False
    assert items[2].dropped is True
    assert flow.state.items_dropped_dedup == 1  # only 'c' is a new drop


def test_matches_scalar_reference_on_clustered_data() -> None:
    """Vectorized dedup must match the scalar reference exactly on clustered
    embeddings (intra-cluster ~1.0, inter-cluster low), across many trials.
    """
    rng = np.random.default_rng(0)
    dim = 32
    threshold = 0.98

    for _ in range(25):
        n_clusters = int(rng.integers(1, 5))
        centers = rng.normal(size=(n_clusters, dim))
        embeddings: list[list[float]] = []
        for _ in range(int(rng.integers(2, 12))):
            c = centers[int(rng.integers(0, n_clusters))]
            vec = c + rng.normal(scale=1e-3, size=dim)  # tiny noise -> sim ~1.0
            embeddings.append(vec.tolist())

        vec_items = [
            ItemState(content=str(i), embedding=e) for i, e in enumerate(embeddings)
        ]
        scalar_items = [
            ItemState(content=str(i), embedding=e) for i, e in enumerate(embeddings)
        ]

        _run_vectorized(vec_items, threshold)

        scalar_flow = _make_flow(threshold)
        scalar_flow._dedup_scalar(scalar_items, threshold)

        assert [it.dropped for it in vec_items] == [
            it.dropped for it in scalar_items
        ]
