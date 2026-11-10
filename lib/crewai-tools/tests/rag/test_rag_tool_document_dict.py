from __future__ import annotations

import os
from typing import Any
from unittest.mock import Mock, patch

from crewai_tools import RagTool
import pytest


@pytest.fixture(autouse=True)
def offline_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")


def add_documents(*items: Any, **kwargs: Any) -> list[dict[str, Any]]:
    """Add content through the public tool API and return the stored documents."""
    client = Mock()
    with patch(
        "crewai_tools.adapters.crewai_rag_adapter.create_client", return_value=client
    ):
        RagTool().add(*items, **kwargs)

    return client.add_documents.call_args.kwargs["documents"]


def test_document_dict_indexes_the_content_body() -> None:
    """The document body is indexed; the source only labels it."""
    documents = add_documents(
        {
            "source": "batch-A",
            "content": "Solve 2x + 3 = 11.",
            "metadata": {"subject": "math"},
        },
        data_type="text",
    )

    assert len(documents) == 1
    assert documents[0]["content"] == "Solve 2x + 3 = 11."
    assert documents[0]["metadata"]["source"] == "batch-A"
    assert documents[0]["metadata"]["subject"] == "math"


def test_document_dict_without_source_indexes_the_content() -> None:
    """A dict that only carries a body still indexes that body."""
    documents = add_documents({"content": "only body"}, data_type="text")

    assert len(documents) == 1
    assert documents[0]["content"] == "only body"


def test_identical_bodies_from_different_sources_stay_separate() -> None:
    """Equal bodies from distinct sources must not collapse into one record."""
    documents = add_documents(
        {"source": "batch-A", "content": "Solve 2x + 3 = 11."},
        {"source": "batch-B", "content": "Solve 2x + 3 = 11."},
        data_type="text",
    )

    assert len(documents) == 2
    assert [doc["content"] for doc in documents] == [
        "Solve 2x + 3 = 11.",
        "Solve 2x + 3 = 11.",
    ]
    assert [doc["metadata"]["source"] for doc in documents] == ["batch-A", "batch-B"]
    assert documents[0]["doc_id"] != documents[1]["doc_id"]


def test_the_same_source_keeps_its_identity() -> None:
    """Re-adding the same document keeps pointing at the same record."""
    documents = add_documents(
        {"source": "batch-A", "content": "Solve 2x + 3 = 11."},
        {"source": "batch-A", "content": "Solve 2x + 3 = 11."},
        data_type="text",
    )

    assert documents[0]["doc_id"] == documents[1]["doc_id"]


def test_document_dict_with_a_file_source_loads_the_file(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source that points at a real file keeps being loaded from disk."""
    monkeypatch.chdir(tmp_path)
    document = tmp_path / "notes.txt"
    document.write_text("file body", encoding="utf-8")

    documents = add_documents(
        {"source": "notes.txt", "metadata": {"subject": "notes"}},
        data_type="text_file",
    )

    assert len(documents) == 1
    assert documents[0]["content"] == "file body"
    assert documents[0]["metadata"]["source"] == "notes.txt"


def test_plain_string_content_is_unchanged() -> None:
    """Plain references keep indexing the reference itself."""
    documents = add_documents("plain body", data_type="text")

    assert len(documents) == 1
    assert documents[0]["content"] == "plain body"
    assert documents[0]["metadata"]["source"] == "plain body"
