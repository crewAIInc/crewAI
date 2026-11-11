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
    """Provide notebook input through both file and inline source paths."""

    def make_source(text: str) -> SourceContent:
        """Wrap text in the selected source representation."""
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
    """Provide a loader for serialized notebook fixtures."""

    def load(notebook: Any) -> LoaderResult:
        """Serialize a fixture and load it through the selected source path."""
        return NotebookLoader().load(
            source_for(json.dumps(notebook, ensure_ascii=False))
        )

    return load


@pytest.fixture
def notebook() -> dict[str, Any]:
    """Build a notebook with mixed cell types, Unicode, and scalar metadata."""
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
    """Preserve mixed cell order and source text with numbered type markers."""
    result = load_notebook(notebook)
    assert result.content == (
        "[cell 1: markdown]\n# Café\n\n\n"
        "[cell 2: code]\nprint('你好')\n\n\n"
        "[cell 3: raw]\n  raw\t"
    )


def test_metadata(
    load_notebook: Callable[[Any], LoaderResult], notebook: dict[str, Any]
) -> None:
    """Retain notebook versions, language, kernel, and counts for each cell type."""
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
    """Return empty content and zero counts when the notebook has no cells."""
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


def test_utf8_bom_file(tmp_path: Path) -> None:
    """Ignore a file's UTF-8 BOM while preserving BOM characters inside cells."""
    path = tmp_path / "notebook.ipynb"
    text = json.dumps(
        {"nbformat": 4, "cells": [{"cell_type": "raw", "source": "a\ufeffb"}]},
        ensure_ascii=False,
    )
    path.write_text(text, encoding="utf-8-sig")
    result = NotebookLoader().load(SourceContent(path))
    assert result.content == "[cell 1: raw]\na\ufeffb"
    assert result.source == str(path)

    path.write_text(text, encoding="utf-8")
    assert result == NotebookLoader().load(SourceContent(path))


@pytest.mark.parametrize("source", ["", []])
def test_empty_cell(
    load_notebook: Callable[[Any], LoaderResult], source: str | list[str]
) -> None:
    """Keep an empty cell's marker and include it in the cell counts."""
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
    """Join source fragments without changing whitespace or Unicode characters."""
    result = load_notebook(
        {"nbformat": 4, "cells": [{"cell_type": "raw", "source": source}]}
    )
    assert result.content == "[cell 1: raw]\n  Café\t\r\n你好\u2028\n"


def test_ignores_outputs_and_attachments(
    load_notebook: Callable[[Any], LoaderResult], tmp_path: Path
) -> None:
    """Extract only cell source, omitting payloads and never executing code."""
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
    """Keep source identity stable and update the document ID when content changes."""
    source = source_for(json.dumps(notebook))
    loader = NotebookLoader()
    result = loader.load(source)
    assert result.source == source.source_ref
    expected_id = hashlib.sha256((result.source + result.content).encode()).hexdigest()
    assert result.doc_id == loader.load(source).doc_id == expected_id
    notebook["cells"][0]["source"] = "Changed"
    changed = loader.load(source_for(json.dumps(notebook)))
    assert changed.doc_id != result.doc_id


@pytest.mark.parametrize("text", ["", '{"nbformat": 4,'])
def test_invalid_json(source_for: Callable[[str], SourceContent], text: str) -> None:
    """Reject empty or malformed JSON instead of returning a partial notebook."""
    with pytest.raises(ValueError):
        NotebookLoader().load(source_for(text))


@pytest.mark.parametrize(
    "notebook, error",
    [
        (None, "notebook"),
        ([], "notebook"),
        (42, "notebook"),
        ("text", "notebook"),
        ({"cells": []}, "nbformat"),
        ({"nbformat": 3, "cells": []}, "nbformat"),
        ({"nbformat": 5, "cells": []}, "nbformat"),
        ({"nbformat": 4.0, "cells": []}, "nbformat"),
        ({"nbformat": True, "cells": []}, "nbformat"),
        ({"nbformat": 4}, "cells"),
        ({"nbformat": 4, "cells": None}, "cells"),
        ({"nbformat": 4, "cells": {}}, "cells"),
        ({"nbformat": 4, "cells": "text"}, "cells"),
    ],
)
def test_invalid_notebook(
    load_notebook: Callable[[Any], LoaderResult], notebook: Any, error: str
) -> None:
    """Reject unsupported versions and malformed required notebook fields."""
    with pytest.raises(ValueError, match=error):
        load_notebook(notebook)


@pytest.mark.parametrize(
    "cell",
    [
        None,
        [],
        "text",
        {"source": "text"},
        {"cell_type": "unknown", "source": "text"},
        {"cell_type": [], "source": "text"},
        {"cell_type": None, "source": "text"},
        {"cell_type": "code"},
        {"cell_type": "code", "source": None},
        {"cell_type": "code", "source": 42},
        {"cell_type": "code", "source": {}},
        {"cell_type": "code", "source": ["valid", 42]},
    ],
)
def test_invalid_cell(load_notebook: Callable[[Any], LoaderResult], cell: Any) -> None:
    """Identify the cell with an invalid type, structure, or source value."""
    with pytest.raises(ValueError, match="cell 2"):
        load_notebook(
            {"nbformat": 4, "cells": [{"cell_type": "raw", "source": "ok"}, cell]}
        )


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        [],
        "text",
        {"language_info": None, "kernelspec": []},
        {"language_info": [], "kernelspec": "text"},
        {"language_info": "text", "kernelspec": None},
        {"language_info": {"name": 42}, "kernelspec": {"name": []}},
        {"cell_count": 999, "format": "other", "nested": {"a": "b"}},
    ],
)
def test_optional_metadata(
    load_notebook: Callable[[Any], LoaderResult], metadata: Any
) -> None:
    """Ignore malformed optional metadata and protect generated counts."""
    result = load_notebook(
        {
            "nbformat": 4,
            "metadata": metadata,
            "cells": [{"cell_type": "raw", "source": "text"}],
        }
    )
    assert result.content == "[cell 1: raw]\ntext"
    assert result.metadata == {
        "format": "ipynb",
        "nbformat": 4,
        "cell_count": 1,
        "markdown_cells": 0,
        "code_cells": 0,
        "raw_cells": 1,
    }


@pytest.mark.parametrize("minor", [0, 99, None, "5", -1, True, 5.0])
def test_minor_version(
    load_notebook: Callable[[Any], LoaderResult], minor: Any
) -> None:
    """Retain nonnegative integer minor versions and omit other values."""
    result = load_notebook({"nbformat": 4, "nbformat_minor": minor, "cells": []})
    if type(minor) is int and minor >= 0:
        assert result.metadata["nbformat_minor"] == minor
    else:
        assert "nbformat_minor" not in result.metadata
