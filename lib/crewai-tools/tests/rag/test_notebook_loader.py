from collections.abc import Callable
import hashlib
import json
from pathlib import Path
from typing import Any

from crewai_tools.rag.base_loader import LoaderResult
from crewai_tools.rag.loaders.notebook_loader import NotebookLoader
from crewai_tools.rag.source_content import SourceContent
import pytest


@pytest.fixture(params=["file", "inline"])
def source_for(
    request: pytest.FixtureRequest, tmp_path: Path
) -> Callable[[str], SourceContent]:
    def make_source(text: str) -> SourceContent:
        if request.param == "file":
            path = tmp_path / "notebook.ipynb"
            path.write_text(text, encoding="utf-8")
            return SourceContent(path)
        return SourceContent(text)

    return make_source


@pytest.fixture
def load_notebook(
    source_for: Callable[[str], SourceContent],
) -> Callable[[Any], LoaderResult]:
    def load(notebook: Any) -> LoaderResult:
        return NotebookLoader().load(
            source_for(json.dumps(notebook, ensure_ascii=False))
        )

    return load


@pytest.fixture
def notebook() -> dict[str, Any]:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "language_info": {"name": "python"},
            "kernelspec": {"name": "python3"},
        },
        "cells": [
            {"cell_type": "markdown", "source": "# Café\n"},
            {"cell_type": "code", "source": ["print(", "'你好')\n"]},
            {"cell_type": "raw", "source": "  raw\t"},
        ],
    }


def test_mixed_cells(
    load_notebook: Callable[[Any], LoaderResult], notebook: dict[str, Any]
) -> None:
    result = load_notebook(notebook)
    assert result.content == (
        "[cell 1: markdown]\n# Café\n\n\n"
        "[cell 2: code]\nprint('你好')\n\n\n"
        "[cell 3: raw]\n  raw\t"
    )


def test_metadata(
    load_notebook: Callable[[Any], LoaderResult], notebook: dict[str, Any]
) -> None:
    assert load_notebook(notebook).metadata == {
        "format": "ipynb",
        "nbformat": 4,
        "nbformat_minor": 5,
        "cell_count": 3,
        "markdown_cells": 1,
        "code_cells": 1,
        "raw_cells": 1,
        "language": "python",
        "kernel": "python3",
    }


def test_empty_notebook(load_notebook: Callable[[Any], LoaderResult]) -> None:
    result = load_notebook({"nbformat": 4, "cells": []})
    assert result.content == ""
    assert result.metadata == {
        "format": "ipynb",
        "nbformat": 4,
        "cell_count": 0,
        "markdown_cells": 0,
        "code_cells": 0,
        "raw_cells": 0,
    }


@pytest.mark.parametrize("source", ["", []])
def test_empty_cell(
    load_notebook: Callable[[Any], LoaderResult], source: str | list[str]
) -> None:
    result = load_notebook(
        {"nbformat": 4, "cells": [{"cell_type": "code", "source": source}]}
    )
    assert result.content == "[cell 1: code]\n"
    assert result.metadata["cell_count"] == result.metadata["code_cells"] == 1
    assert result.metadata["markdown_cells"] == result.metadata["raw_cells"] == 0


@pytest.mark.parametrize(
    "source", ["  Café\t\r\n你好\u2028\n", ["  Café", "\t\r", "\n你好\u2028", "\n"]]
)
def test_source_fragments(
    load_notebook: Callable[[Any], LoaderResult], source: str | list[str]
) -> None:
    result = load_notebook(
        {"nbformat": 4, "cells": [{"cell_type": "raw", "source": source}]}
    )
    assert result.content == "[cell 1: raw]\n  Café\t\r\n你好\u2028\n"


def test_ignores_outputs_and_attachments(
    load_notebook: Callable[[Any], LoaderResult], tmp_path: Path
) -> None:
    sentinel = tmp_path / "executed"
    code = f"from pathlib import Path\nPath({str(sentinel)!r}).touch()"
    result = load_notebook(
        {
            "nbformat": 4,
            "cells": [
                {
                    "cell_type": "code",
                    "source": code,
                    "outputs": [{"output_type": "stream", "text": "OUTPUT_SENTINEL"}],
                },
                {
                    "cell_type": "markdown",
                    "source": "![image](attachment:picture.png)",
                    "attachments": {
                        "picture.png": {"image/png": "ATTACHMENT_SENTINEL"}
                    },
                },
            ],
        }
    )
    assert result.content == (
        f"[cell 1: code]\n{code}\n\n"
        "[cell 2: markdown]\n![image](attachment:picture.png)"
    )
    assert "OUTPUT_SENTINEL" not in json.dumps(result.metadata)
    assert "ATTACHMENT_SENTINEL" not in json.dumps(result.metadata)
    assert not sentinel.exists()


def test_source_and_doc_id(
    source_for: Callable[[str], SourceContent], notebook: dict[str, Any]
) -> None:
    source = source_for(json.dumps(notebook))
    loader = NotebookLoader()
    result = loader.load(source)
    assert result.source == source.source_ref
    expected_id = hashlib.sha256((result.source + result.content).encode()).hexdigest()
    assert result.doc_id == loader.load(source).doc_id == expected_id
    notebook["cells"][0]["source"] = "Changed"
    changed = loader.load(source_for(json.dumps(notebook)))
    assert changed.doc_id != result.doc_id
