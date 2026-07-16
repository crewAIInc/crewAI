"""Tests for Milvus utility functions."""

import importlib.metadata
from typing import Any

import pytest


pytest.importorskip("pymilvus")

from crewai.rag.milvus import utils
from crewai.rag.milvus.utils import (
    _milvus_lite_uses_cosine_distance,
    _normalize_milvus_score,
    _process_search_results,
)


def test_normalize_milvus_score_keeps_cosine_and_l2_mappings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test existing COSINE and L2 score normalization behavior."""
    monkeypatch.setattr(utils, "_milvus_lite_uses_cosine_distance", lambda: False)

    assert _normalize_milvus_score(raw_score=1.0, metric_type="COSINE") == 1.0
    assert _normalize_milvus_score(raw_score=0.0, metric_type="COSINE") == 0.5
    assert _normalize_milvus_score(raw_score=0.0, metric_type="L2") == 1.0
    assert _normalize_milvus_score(raw_score=2.0, metric_type="L2") == pytest.approx(
        1.0 / 3.0
    )


def test_normalize_milvus_score_keeps_inner_product_raw() -> None:
    """Test that IP scores preserve Milvus threshold and ordering semantics."""
    assert _normalize_milvus_score(raw_score=2.0, metric_type="IP") == 2.0
    assert _normalize_milvus_score(raw_score=0.4, metric_type="IP") == 0.4
    assert _normalize_milvus_score(raw_score=-0.5, metric_type="IP") == -0.5


def test_process_search_results_keeps_inner_product_threshold_and_order() -> None:
    """Test that IP processing does not apply cosine-style normalization."""
    response: list[list[dict[str, Any]]] = [
        [
            {
                "id": "low",
                "distance": 1.5,
                "entity": {"content": "low", "metadata": {}},
            },
            {
                "id": "below-threshold",
                "distance": 0.4,
                "entity": {"content": "below", "metadata": {}},
            },
            {
                "id": "high",
                "distance": 2.0,
                "entity": {"content": "high", "metadata": {}},
            },
        ]
    ]

    results = _process_search_results(
        response=response,
        metric_type="IP",
        score_threshold=0.6,
    )

    assert [result["id"] for result in results] == ["high", "low"]
    assert [result["score"] for result in results] == [2.0, 1.5]


def test_milvus_lite_cosine_distance_probe_is_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test that Milvus Lite version lookup is not repeated per search hit."""
    calls = 0

    def fake_version(package_name: str) -> str:
        nonlocal calls
        calls += 1
        assert package_name == "milvus-lite"
        return "3.0.0"

    _milvus_lite_uses_cosine_distance.cache_clear()
    monkeypatch.setattr(importlib.metadata, "version", fake_version)
    try:
        assert _milvus_lite_uses_cosine_distance() is True
        assert _milvus_lite_uses_cosine_distance() is True
        assert calls == 1
    finally:
        _milvus_lite_uses_cosine_distance.cache_clear()
