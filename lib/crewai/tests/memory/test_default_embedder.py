"""Memory's default embedder goes through the native OpenAI function."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from crewai.memory.unified_memory import Memory, _default_embedder
from crewai.rag.embeddings.providers.openai.embedding_callable import (
    OpenAIEmbeddingFunction,
)


@pytest.fixture
def openai_mock(monkeypatch: pytest.MonkeyPatch):
    for name in (
        "EMBEDDINGS_OPENAI_MODEL_NAME",
        "EMBEDDINGS_OPENAI_API_BASE",
        "OPENAI_API_BASE",
        "EMBEDDINGS_OPENAI_API_TYPE",
        "OPENAI_API_TYPE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    with patch("openai.OpenAI") as client_cls:
        client_cls.return_value.embeddings.create.side_effect = lambda **kw: MagicMock(
            data=[MagicMock(embedding=[0.1] * 8) for _ in kw["input"]]
        )
        yield client_cls


def test_default_embedder_is_native_openai_function(openai_mock) -> None:
    embedder = _default_embedder()

    assert isinstance(embedder, OpenAIEmbeddingFunction)
    assert embedder.model_name == "text-embedding-3-large"
    assert openai_mock.call_args.kwargs["api_key"] == "test-key"


def test_memory_remember_and_recall_with_default_embedder(
    tmp_path: Path, openai_mock
) -> None:
    memory = Memory(storage=str(tmp_path / "db"), llm=MagicMock())

    memory.remember(
        "We decided to use Python.",
        scope="/project",
        categories=["decision"],
        importance=0.7,
    )
    matches = memory.recall("Python", scope="/project", limit=5, depth="shallow")

    assert len(matches) == 1
    assert matches[0].record.content == "We decided to use Python."
    create = openai_mock.return_value.embeddings.create
    assert create.call_args.kwargs["model"] == "text-embedding-3-large"
