"""Tests for notebook DataType routing and RAG ingestion."""

from collections.abc import Callable
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from unittest.mock import MagicMock, patch
from uuid import uuid4

from crewai_tools.rag.chunkers.text_chunker import TextChunker
from crewai_tools.rag.core import RAG
from crewai_tools.rag.data_types import DataType, DataTypes
from crewai_tools.rag.loaders.notebook_loader import NotebookLoader
from crewai_tools.tools.rag.rag_tool import RagTool
import pytest


CELLS = [
    ("markdown", "# Quarterly revenue analysis"),
    ("code", "import pandas as pd\ndf = pd.read_csv('revenue.csv')"),
    ("markdown", "Revenue grew 12% in Q3."),
    ("code", "df.plot(x='quarter', y='revenue')"),
]


def make_notebook(cells: list[tuple[str, str]]) -> dict[str, Any]:
    nb_cells: list[dict[str, Any]] = []
    for cell_type, source in cells:
        cell: dict[str, Any] = {
            "cell_type": cell_type,
            "metadata": {},
            "source": source.splitlines(keepends=True),
        }
        if cell_type == "code":
            cell["outputs"] = [
                {"output_type": "stream", "name": "stdout", "text": "OUTPUT-TEXT"},
                {
                    "output_type": "display_data",
                    "data": {"image/png": "iVBORw0KGgoAAAANSUhEUgAA"},
                    "metadata": {},
                },
            ]
            cell["execution_count"] = 1
        nb_cells.append(cell)
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "cells": nb_cells,
    }


@pytest.fixture
def write_notebook(tmp_path: Path) -> Callable[..., Path]:
    def _write(cells: list[tuple[str, str]], name: str = "analysis.ipynb") -> Path:
        path = tmp_path / name
        path.write_text(json.dumps(make_notebook(cells)), encoding="utf-8")
        return path

    return _write


class TestNotebookRouting:
    def test_local_ipynb_path_is_notebook(
        self, write_notebook: Callable[..., Path]
    ) -> None:
        path = write_notebook(CELLS)
        assert DataTypes.from_content(str(path)) == DataType.NOTEBOOK
        assert DataTypes.from_content(path) == DataType.NOTEBOOK

    def test_explicit_type_from_string(self) -> None:
        assert DataType("notebook") == DataType.NOTEBOOK

    def test_remote_ipynb_url_is_not_routed_to_notebook_loader(self) -> None:
        # NotebookLoader cannot fetch URLs, so remote notebooks keep web handling.
        assert (
            DataTypes.from_content("https://example.com/analysis.ipynb")
            == DataType.WEBSITE
        )

    def test_existing_extensions_unchanged(self, tmp_path: Path) -> None:
        json_file = tmp_path / "data.json"
        json_file.write_text("{}")
        assert DataTypes.from_content(str(json_file)) == DataType.JSON

    def test_loader_and_chunker(self) -> None:
        assert isinstance(DataType.NOTEBOOK.get_loader(), NotebookLoader)
        assert type(DataType.NOTEBOOK.get_chunker()) is TextChunker


@pytest.fixture(autouse=True)
def allow_tmp_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CREWAI_TOOLS_ALLOW_UNSAFE_PATHS", "true")


@pytest.fixture
def mock_rag_client() -> MagicMock:
    client = MagicMock()
    client.get_or_create_collection = MagicMock(return_value=None)
    client.add_documents = MagicMock(return_value=None)
    return client


@pytest.fixture
def rag_tool(mock_rag_client: MagicMock) -> RagTool:
    with (
        patch(
            "crewai_tools.adapters.crewai_rag_adapter.get_rag_client",
            return_value=mock_rag_client,
        ),
        patch(
            "crewai_tools.adapters.crewai_rag_adapter.create_client",
            return_value=mock_rag_client,
        ),
    ):
        return RagTool()


def stored_documents(client: MagicMock) -> list[dict[str, Any]]:
    assert client.add_documents.called
    return list(client.add_documents.call_args.kwargs["documents"])


class TestRagToolNotebookIngestion:
    @pytest.mark.parametrize("data_type", [None, "notebook", DataType.NOTEBOOK])
    def test_multi_cell_notebook_preserves_markers_and_source(
        self,
        rag_tool: RagTool,
        mock_rag_client: MagicMock,
        write_notebook: Callable[..., Path],
        data_type: str | DataType | None,
    ) -> None:
        path = write_notebook(CELLS)
        kwargs: dict[str, Any] = {} if data_type is None else {"data_type": data_type}

        rag_tool.add(str(path), **kwargs)

        documents = stored_documents(mock_rag_client)
        text = "\n".join(doc["content"] for doc in documents)
        for number, (cell_type, source) in enumerate(CELLS, start=1):
            assert f"[cell {number}: {cell_type}]\n{source}" in text
        assert "OUTPUT-TEXT" not in text
        assert "iVBORw0KGgo" not in text

        metadata = documents[0]["metadata"]
        assert metadata["data_type"] == str(DataType.NOTEBOOK)
        assert metadata["source"] == str(path)
        assert metadata["format"] == "ipynb"
        assert metadata["cell_count"] == 4
        assert metadata["code_cells"] == 2
        assert metadata["markdown_cells"] == 2
        assert metadata["kernel"] == "python3"

    def test_large_notebook_chunks_keep_every_cell_marker(
        self,
        rag_tool: RagTool,
        mock_rag_client: MagicMock,
        write_notebook: Callable[..., Path],
    ) -> None:
        cells = [
            ("markdown" if i % 2 else "code", f"cell {i} " + "word " * 120)
            for i in range(1, 21)
        ]
        path = write_notebook(cells)

        rag_tool.add(str(path))

        documents = stored_documents(mock_rag_client)
        assert len(documents) > 1
        markers = [
            m
            for doc in documents
            for m in re.findall(r"\[cell (\d+): (\w+)\]", doc["content"])
        ]
        expected = [(str(i), cell_type) for i, (cell_type, _) in enumerate(cells, 1)]
        assert sorted(set(markers), key=lambda m: int(m[0])) == expected
        assert all(
            doc["metadata"]["total_chunks"] == len(documents) for doc in documents
        )

    def test_empty_notebook_stores_nothing(
        self,
        rag_tool: RagTool,
        mock_rag_client: MagicMock,
        write_notebook: Callable[..., Path],
    ) -> None:
        path = write_notebook([])

        rag_tool.add(str(path))

        mock_rag_client.add_documents.assert_not_called()

    def test_empty_cells_keep_markers(
        self,
        rag_tool: RagTool,
        mock_rag_client: MagicMock,
        write_notebook: Callable[..., Path],
    ) -> None:
        path = write_notebook([("markdown", ""), ("code", "")])

        rag_tool.add(str(path))

        text = "\n".join(d["content"] for d in stored_documents(mock_rag_client))
        assert "[cell 1: markdown]" in text
        assert "[cell 2: code]" in text

    def test_missing_notebook_raises(self, rag_tool: RagTool, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="File does not exist"):
            rag_tool.add(str(tmp_path / "missing.ipynb"))

    def test_directory_ingestion_routes_notebooks(
        self,
        rag_tool: RagTool,
        mock_rag_client: MagicMock,
        write_notebook: Callable[..., Path],
        tmp_path: Path,
    ) -> None:
        write_notebook(CELLS)
        (tmp_path / "notes.txt").write_text("plain text notes")

        rag_tool.add(str(tmp_path), data_type="directory")

        documents = stored_documents(mock_rag_client)
        by_type = {doc["metadata"]["data_type"]: doc for doc in documents}
        assert "[cell 1: markdown]" in by_type[str(DataType.NOTEBOOK)]["content"]
        assert str(DataType.TEXT_FILE) in by_type


class FakeEmbeddingService:
    """Deterministic bag-of-words embeddings; no model calls."""

    DIM = 256

    def __init__(self, **kwargs: Any) -> None:
        pass

    def embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.DIM
        for word in re.findall(r"[a-z]+", text.lower()):
            digest = hashlib.sha256(word.encode()).digest()
            vector[int.from_bytes(digest[:2]) % self.DIM] += 1.0
        return vector if any(vector) else [1.0] + [0.0] * (self.DIM - 1)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(text) for text in texts]


class TestCoreRagNotebookAddQuery:
    def test_add_then_query_local_notebook(self, tmp_path: Path) -> None:
        # Cells use disjoint vocabularies so only cell 3 matches the question.
        cells = [
            ("markdown", "Setup: install dependencies first. " * 25),
            ("code", "import pandas as pd; frame = pd.read_csv('sales.csv'); " * 16),
            ("markdown", "Revenue grew twelve percent in the third quarter. " * 18),
        ]
        path = tmp_path / "report.ipynb"
        path.write_text(json.dumps(make_notebook(cells)), encoding="utf-8")

        with patch("crewai_tools.rag.core.EmbeddingService", FakeEmbeddingService):
            rag = RAG(collection_name=f"notebook-{uuid4().hex}", top_k=1)
        try:
            rag.add(str(path))

            assert rag.get_collection_info()["count"] > 1
            result = rag.query("revenue grew in the third quarter")
            assert f"[Source: {path}" in result
            assert "[cell 3: markdown]" in result
            assert "Revenue grew twelve percent" in result
        finally:
            rag.delete_collection()
