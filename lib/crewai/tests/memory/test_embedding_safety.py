"""Tests for embedding safety: bytes→float validators and async-safe embed_texts."""

from __future__ import annotations

import asyncio
import concurrent.futures
from unittest.mock import MagicMock

import numpy as np
import pytest

from crewai.memory.types import MemoryRecord, embed_text, embed_texts


class TestMemoryRecordEmbeddingValidator:
    """Tests for MemoryRecord.validate_embedding (bytes→list[float])."""

    def test_none_embedding_stays_none(self) -> None:
        r = MemoryRecord(content="test", embedding=None)
        assert r.embedding is None

    def test_list_of_floats_passes_through(self) -> None:
        r = MemoryRecord(content="test", embedding=[0.1, 0.2, 0.3])
        assert r.embedding == [0.1, 0.2, 0.3]

    def test_bytes_converted_to_list_float(self) -> None:
        arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        raw_bytes = arr.tobytes()
        r = MemoryRecord(content="test", embedding=raw_bytes)
        assert r.embedding is not None
        assert len(r.embedding) == 3
        assert all(isinstance(x, float) for x in r.embedding)
        np.testing.assert_allclose(r.embedding, [0.1, 0.2, 0.3], atol=1e-6)

    def test_empty_bytes_becomes_none(self) -> None:
        r = MemoryRecord(content="test", embedding=b"")
        assert r.embedding is None

    def test_list_of_ints_converted_to_floats(self) -> None:
        r = MemoryRecord(content="test", embedding=[1, 2, 3])
        assert r.embedding == [1.0, 2.0, 3.0]
        assert all(isinstance(x, float) for x in r.embedding)

    def test_numpy_array_converted_to_list(self) -> None:
        arr = np.array([0.5, 0.6], dtype=np.float32)
        r = MemoryRecord(content="test", embedding=arr)
        assert r.embedding is not None
        assert isinstance(r.embedding, list)
        assert len(r.embedding) == 2


class TestEmbedTextsAsyncSafety:
    """Tests for embed_texts running safely in async context."""

    def test_embed_texts_sync_context(self) -> None:
        """embed_texts works in a normal sync context."""
        embedder = MagicMock(return_value=[[0.1, 0.2], [0.3, 0.4]])
        result = embed_texts(embedder, ["hello", "world"])
        assert len(result) == 2
        assert result[0] == [0.1, 0.2]
        embedder.assert_called_once()

    def test_embed_texts_empty_input(self) -> None:
        embedder = MagicMock()
        assert embed_texts(embedder, []) == []
        embedder.assert_not_called()

    def test_embed_texts_all_empty_strings(self) -> None:
        embedder = MagicMock()
        result = embed_texts(embedder, ["", "  ", ""])
        assert result == [[], [], []]
        embedder.assert_not_called()

    def test_embed_texts_skips_empty_preserves_positions(self) -> None:
        embedder = MagicMock(return_value=[[0.1, 0.2]])
        result = embed_texts(embedder, ["", "hello", ""])
        assert result == [[], [0.1, 0.2], []]
        embedder.assert_called_once_with(["hello"])

    def test_embed_texts_in_async_context(self) -> None:
        """embed_texts uses thread pool when called from async context."""
        embedder = MagicMock(return_value=[[0.1, 0.2]])

        async def run() -> list[list[float]]:
            return embed_texts(embedder, ["hello"])

        result = asyncio.run(run())
        assert result == [[0.1, 0.2]]
        embedder.assert_called_once()


class TestEmbedText:
    """Tests for embed_text (single text)."""

    def test_empty_string_returns_empty(self) -> None:
        embedder = MagicMock()
        assert embed_text(embedder, "") == []
        embedder.assert_not_called()

    def test_whitespace_only_returns_empty(self) -> None:
        embedder = MagicMock()
        assert embed_text(embedder, "   ") == []
        embedder.assert_not_called()

    def test_normal_text_returns_embedding(self) -> None:
        embedder = MagicMock(return_value=[[0.1, 0.2, 0.3]])
        result = embed_text(embedder, "hello")
        assert result == [0.1, 0.2, 0.3]

    def test_numpy_array_result_converted(self) -> None:
        arr = np.array([0.1, 0.2], dtype=np.float32)
        embedder = MagicMock(return_value=[arr])
        result = embed_text(embedder, "hello")
        assert isinstance(result, list)
        assert len(result) == 2
