"""RagTool picks Qdrant by default; chromadb is opt-in via crewai[chromadb]."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from crewai_tools.rag import RAG
from crewai_tools.tools.rag.rag_tool import RagTool


class RecordingEmbedder:
    """CrewAI-style embedder (list in, vectors out) that records its inputs."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, input: list[str]) -> list[np.ndarray]:
        self.calls.append(list(input))
        return [np.array([0.1, 0.2, 0.3]) for _ in input]


@pytest.fixture
def created_config():
    """Yield a getter for the config RagTool hands to the client factory."""
    with patch("crewai_tools.adapters.crewai_rag_adapter.create_client") as create:
        create.return_value = MagicMock()
        yield lambda: create.call_args.args[0]


def test_default_provider_is_qdrant_with_the_default_embedder(created_config):
    embedder = RecordingEmbedder()
    with patch(
        "crewai_tools.tools.rag.rag_tool.default_embedding_function",
        return_value=embedder,
    ):
        RagTool()

    config = created_config()
    assert config.provider == "qdrant"
    assert config.embedding_function("hello") == pytest.approx([0.1, 0.2, 0.3])
    assert embedder.calls == [["hello"]]


def test_embedding_model_without_provider_still_defaults_to_qdrant(created_config):
    embedder = RecordingEmbedder()
    with patch("crewai_tools.tools.rag.rag_tool.build_embedder", return_value=embedder):
        RagTool(
            config={
                "embedding_model": {
                    "provider": "openai",
                    "config": {"model": "text-embedding-3-small", "api_key": "k"},
                }
            }
        )

    config = created_config()
    assert config.provider == "qdrant"
    config.embedding_function("hello")
    assert embedder.calls == [["hello"]]


def test_explicit_qdrant_embedder_is_wrapped_for_the_qdrant_client(created_config):
    embedder = RecordingEmbedder()
    with patch("crewai_tools.tools.rag.rag_tool.build_embedder", return_value=embedder):
        RagTool(
            config={
                "vectordb": {"provider": "qdrant", "config": {}},
                "embedding_model": {
                    "provider": "openai",
                    "config": {"model": "text-embedding-3-small", "api_key": "k"},
                },
            }
        )

    # Qdrant embeds one string at a time; the CrewAI embedder expects a list.
    created_config().embedding_function("hello")
    assert embedder.calls == [["hello"]]


def test_chromadb_adapter_explains_how_to_install_it():
    with patch.dict(sys.modules, {"chromadb": None}):
        with pytest.raises(ImportError, match=r"pip install crewai\[chromadb\]"):
            RAG()
