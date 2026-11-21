"""Notebook ingestion through the real loader and chunker, with no external services."""

import json
import math
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

from crewai_tools.rag.core import RAG
from crewai_tools.rag.data_types import DataType
import pytest


VOCABULARY = ("revenue", "pandas", "matplotlib")


class FakeEmbeddings:
    """Embed text as vocabulary word counts and capture what was embedded."""

    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed_text(self, text: str) -> list[float]:
        return [float(text.lower().count(word)) for word in VOCABULARY]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [self.embed_text(text) for text in texts]


class FakeCollection:
    """Store documents in memory and rank them by cosine similarity."""

    def __init__(self) -> None:
        self.documents: list[str] = []
        self.ids: list[str] = []
        self.embeddings: list[list[float]] = []
        self.metadatas: list[dict[str, Any]] = []

    def get(self, *, where: dict[str, Any], limit: int) -> dict[str, Any]:
        return {"metadatas": []}

    def add(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        self.ids.extend(ids)
        self.embeddings.extend(embeddings)
        self.documents.extend(documents)
        self.metadatas.extend(metadatas)

    def query(
        self,
        *,
        query_embeddings: list[list[float]],
        n_results: int,
        where: dict[str, Any] | None,
        include: list[str],
    ) -> dict[str, Any]:
        query = query_embeddings[0]

        def distance(embedding: list[float]) -> float:
            norm = math.hypot(*query) * math.hypot(*embedding)
            dot = sum(a * b for a, b in zip(query, embedding, strict=True))
            return 1 - dot / norm if norm else 1.0

        ranked = sorted(
            range(len(self.documents)), key=lambda i: distance(self.embeddings[i])
        )[:n_results]
        return {
            "documents": [[self.documents[i] for i in ranked]],
            "metadatas": [[self.metadatas[i] for i in ranked]],
            "distances": [[distance(self.embeddings[i]) for i in ranked]],
        }


@pytest.fixture
def rag() -> tuple[RAG, FakeCollection, FakeEmbeddings]:
    """Use real RAG initialization with fake storage and embeddings."""
    collection = FakeCollection()
    embeddings = FakeEmbeddings()
    client = Mock()
    client.get_or_create_collection.return_value = collection
    with (
        patch("crewai_tools.rag.core.chromadb.Client", return_value=client),
        patch("crewai_tools.rag.core.EmbeddingService", return_value=embeddings),
    ):
        instance = RAG()
    return instance, collection, embeddings


def write_notebook(path: Path, cells: list[dict[str, Any]], nbformat: int = 4) -> Path:
    """Serialize cells into a notebook file."""
    path.write_text(
        json.dumps({"nbformat": nbformat, "cells": cells}, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def stored_lines(collection: FakeCollection) -> list[str]:
    """Return non-blank stored lines; the splitter may pad blank lines."""
    return [
        line
        for document in collection.documents
        for line in document.splitlines()
        if line.strip()
    ]


@pytest.fixture
def long_notebook(tmp_path: Path) -> Path:
    """Build a notebook whose cells are too large to share a single chunk."""
    return write_notebook(
        tmp_path / "report.ipynb",
        [
            {
                "cell_type": "markdown",
                "source": "# Revenue report\n"
                + "\n".join(f"Region {i} revenue grew this quarter." for i in range(20)),
            },
            {
                "cell_type": "code",
                "source": "import pandas\n"
                + "\n".join(f'pandas.read_csv("region_{i}.csv")' for i in range(20)),
            },
            {
                "cell_type": "code",
                "source": "import matplotlib.pyplot as plt\n"
                + "\n".join(f"matplotlib.pyplot.plot(series_{i})" for i in range(20)),
            },
        ],
    )


@pytest.mark.parametrize(
    ("extension", "data_type"),
    [
        (".ipynb", None),
        (".txt", "notebook"),
        (".txt", DataType.NOTEBOOK),
    ],
)
def test_add_multi_cell_notebook_stores_markers_and_source(
    rag: tuple[RAG, FakeCollection, FakeEmbeddings],
    tmp_path: Path,
    extension: str,
    data_type: str | DataType | None,
) -> None:
    """Explicit selection and discovery keep cell markers, source, and order."""
    instance, collection, embeddings = rag
    path = write_notebook(
        tmp_path / f"analysis{extension}",
        [
            {"cell_type": "markdown", "source": "# Café sales\n\nSummary of 你好."},
            {
                "cell_type": "code",
                "source": ["import pandas as pd\n", "\n", "print(pd.__version__)"],
                "outputs": [{"output_type": "stream", "text": "OUTPUT_SENTINEL"}],
            },
            {"cell_type": "raw", "source": "raw notes"},
        ],
    )

    instance.add(path, data_type=data_type, metadata={"project": "example"})

    assert stored_lines(collection) == [
        "[cell 1: markdown]",
        "# Café sales",
        "Summary of 你好.",
        "[cell 2: code]",
        "import pandas as pd",
        "print(pd.__version__)",
        "[cell 3: raw]",
        "raw notes",
    ]
    assert "OUTPUT_SENTINEL" not in "".join(collection.documents)
    assert embeddings.texts == collection.documents
    assert len(collection.ids) == len(collection.documents) > 0
    for index, metadata in enumerate(collection.metadatas):
        assert metadata["data_type"] == "notebook"
        assert metadata["source"] == str(path)
        assert metadata["project"] == "example"
        assert metadata["chunk_index"] == index
        assert metadata["doc_id"]


def test_markers_survive_chunking(
    rag: tuple[RAG, FakeCollection, FakeEmbeddings], long_notebook: Path
) -> None:
    """Every marker stays attached to its cell's source across multiple chunks."""
    instance, collection, _ = rag
    instance.add(long_notebook)

    assert len(collection.documents) > 1
    stored = "\n".join(stored_lines(collection))
    assert "[cell 1: markdown]\n# Revenue report\n" in stored
    assert "[cell 2: code]\nimport pandas\n" in stored
    assert "[cell 3: code]\nimport matplotlib.pyplot as plt\n" in stored
    assert "matplotlib.pyplot.plot(series_19)" in stored


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What happened to revenue?", "[cell 1: markdown]\n# Revenue report"),
        (
            "Which cell plots with matplotlib?",
            "[cell 3: code]\nimport matplotlib.pyplot as plt",
        ),
    ],
)
def test_add_then_query_returns_matching_cell(
    rag: tuple[RAG, FakeCollection, FakeEmbeddings],
    long_notebook: Path,
    question: str,
    expected: str,
) -> None:
    """A local add/query flow retrieves the relevant cell with its marker."""
    instance, _, _ = rag
    instance.top_k = 1
    instance.add(long_notebook)

    result = instance.query(question)

    assert result.startswith(f"[Source: {long_notebook}, Relevance: ")
    assert expected in result


def test_empty_notebook_is_not_stored(
    rag: tuple[RAG, FakeCollection, FakeEmbeddings], tmp_path: Path
) -> None:
    """A notebook without cells does not produce embeddings or documents."""
    instance, collection, embeddings = rag
    instance.add(write_notebook(tmp_path / "empty.ipynb", []))
    assert collection.documents == []
    assert embeddings.texts == []
    assert instance.query("revenue") == "No relevant content found."


def test_empty_cell_keeps_marker(
    rag: tuple[RAG, FakeCollection, FakeEmbeddings], tmp_path: Path
) -> None:
    """A cell without source is still stored as its marker."""
    instance, collection, _ = rag
    path = write_notebook(
        tmp_path / "blank.ipynb", [{"cell_type": "code", "source": ""}]
    )
    instance.add(path)
    assert stored_lines(collection) == ["[cell 1: code]"]


def test_invalid_notebook_is_not_stored(
    rag: tuple[RAG, FakeCollection, FakeEmbeddings], tmp_path: Path
) -> None:
    """Unsupported notebooks fail before any embedding or storage."""
    instance, collection, embeddings = rag
    path = write_notebook(tmp_path / "old.ipynb", [], nbformat=3)
    with pytest.raises(ValueError, match="nbformat 4"):
        instance.add(path)
    assert collection.documents == []
    assert embeddings.texts == []
