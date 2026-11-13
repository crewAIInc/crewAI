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


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn every URL fetch into a failure."""

    def fail(*args: Any, **kwargs: Any) -> str:
        raise AssertionError("the document body must not be fetched")

    for module in ("text_loader", "csv_loader", "json_loader", "mdx_loader", "xml_loader"):
        monkeypatch.setattr(f"crewai_tools.rag.loaders.{module}.load_from_url", fail)


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


def test_document_dict_indexes_the_body_without_an_explicit_data_type() -> None:
    """The body is indexed even when the data type is detected."""
    documents = add_documents({"source": "batch-A", "content": "Solve 2x + 3 = 11."})

    assert len(documents) == 1
    assert documents[0]["content"] == "Solve 2x + 3 = 11."
    assert documents[0]["metadata"]["source"] == "batch-A"


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


def test_an_explicit_loader_type_loads_the_source_instead_of_the_body(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Asking for a loader makes the source the document to read."""
    monkeypatch.chdir(tmp_path)
    document = tmp_path / "notes.txt"
    document.write_text("file body", encoding="utf-8")

    documents = add_documents(
        {"source": "notes.txt", "content": "the body"},
        data_type="text_file",
    )

    assert len(documents) == 1
    assert documents[0]["content"] == "file body"


def test_plain_string_content_is_unchanged() -> None:
    """Plain references keep indexing the reference itself."""
    documents = add_documents("plain body", data_type="text")

    assert len(documents) == 1
    assert documents[0]["content"] == "plain body"
    assert documents[0]["metadata"]["source"] == "plain body"


def test_a_body_ending_with_a_known_extension_is_still_indexed_as_text() -> None:
    """A body that looks like a missing file is not looked up on disk."""
    documents = add_documents({"source": "batch-A", "content": "read notes.md please"})

    assert len(documents) == 1
    assert documents[0]["content"] == "read notes.md please"


def test_a_body_that_is_a_path_is_not_read_from_disk(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A body that names an existing file is indexed as the text it is."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "notes.txt").write_text("file body", encoding="utf-8")

    documents = add_documents({"source": "batch-A", "content": "notes.txt"})

    assert len(documents) == 1
    assert documents[0]["content"] == "notes.txt"


def test_a_body_that_is_a_url_is_not_fetched(no_network: None) -> None:
    """A body that names a URL is indexed as the text it is."""
    documents = add_documents(
        {"source": "batch-A", "content": "https://example.com/page"}
    )

    assert len(documents) == 1
    assert documents[0]["content"] == "https://example.com/page"


def test_a_source_label_matching_a_directory_keeps_the_body(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A label that matches a folder in the working directory is a label."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "docs").mkdir()

    documents = add_documents({"source": "docs", "content": "Solve 2x + 3 = 11."})

    assert len(documents) == 1
    assert documents[0]["content"] == "Solve 2x + 3 = 11."
    assert documents[0]["metadata"]["source"] == "docs"


def test_a_source_label_matching_a_file_keeps_the_body(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A label that matches a file in the working directory is a label."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "README.md").write_text("# the readme", encoding="utf-8")

    documents = add_documents(
        {"source": "README.md", "content": "Solve 2x + 3 = 11."},
        data_type="text",
    )

    assert len(documents) == 1
    assert documents[0]["content"] == "Solve 2x + 3 = 11."
    assert documents[0]["metadata"]["source"] == "README.md"


def test_a_path_like_source_label_is_kept_as_it_is(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A label is metadata, so it is not turned into an absolute path."""
    monkeypatch.chdir(tmp_path)

    documents = add_documents(
        {"source": os.path.join("docs", "report"), "content": "Solve 2x + 3 = 11."},
        data_type="text",
    )

    assert len(documents) == 1
    assert documents[0]["metadata"]["source"] == os.path.join("docs", "report")


def test_a_body_that_looks_like_a_path_is_not_read(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A body is text, never a path that gets read behind the caller's back."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "secret.txt").write_text("secret body", encoding="utf-8")

    documents = add_documents({"source": "batch-A", "content": "secret.txt"})

    assert len(documents) == 1
    assert documents[0]["content"] == "secret.txt"


def test_a_body_that_looks_like_a_path_outside_the_sandbox_is_indexed_as_text(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A body naming a path outside the sandbox is text, not a file to read."""
    monkeypatch.chdir(tmp_path)

    documents = add_documents({"source": "batch-A", "content": "../../etc/passwd"})

    assert len(documents) == 1
    assert documents[0]["content"] == "../../etc/passwd"


def test_a_body_that_looks_like_an_internal_url_is_not_fetched(
    no_network: None,
) -> None:
    """A body naming an internal URL is text, not a URL to fetch."""
    documents = add_documents(
        {"source": "batch-A", "content": "http://169.254.169.254/latest/meta-data/"}
    )

    assert len(documents) == 1
    assert documents[0]["content"] == "http://169.254.169.254/latest/meta-data/"


def test_a_body_without_a_source_is_still_checked_as_a_reference(
    tmp_path: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without a source the body is the reference, so it is validated."""
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="Blocked unsafe file path"):
        add_documents({"content": "../../etc/passwd"}, data_type="text_file")
