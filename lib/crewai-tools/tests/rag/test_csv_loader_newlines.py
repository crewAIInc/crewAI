from pathlib import Path
from unittest.mock import Mock

from crewai_tools.rag.loaders.csv_loader import CSVLoader
from crewai_tools.rag.source_content import SourceContent
import pytest


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"], ids=["lf", "crlf", "cr"])
@pytest.mark.parametrize("source_kind", ["inline", "file", "url"])
def test_csv_record_newlines(
    newline: str, source_kind: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = newline.join(["name,age", "Alice,30", "Bob,25", ""])
    source = text
    if source_kind == "file":
        path = tmp_path / "records.csv"
        path.write_bytes(text.encode("utf-8"))
        source = str(path)
    elif source_kind == "url":
        source = "https://example.com/records.csv"
        monkeypatch.setattr(
            "crewai_tools.security.safe_requests.safe_get",
            Mock(return_value=Mock(text=text)),
        )

    result = CSVLoader().load(SourceContent(source))

    assert result.metadata == {
        "format": "csv",
        "columns": ["name", "age"],
        "rows": 2,
    }
    assert "Row 1: name: Alice | age: 30" in result.content
    assert "Row 2: name: Bob | age: 25" in result.content
    assert result.doc_id == CSVLoader.generate_doc_id(result.source, result.content)


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"], ids=["lf", "crlf", "cr"])
@pytest.mark.parametrize("source_kind", ["inline", "file"])
def test_csv_preserves_quoted_newlines(
    newline: str, source_kind: str, tmp_path: Path
) -> None:
    text = f'name,note{newline}Alice,"first{newline}second"{newline}'
    source = text
    if source_kind == "file":
        path = tmp_path / "quoted.csv"
        path.write_bytes(text.encode("utf-8"))
        source = str(path)

    result = CSVLoader().load(SourceContent(source))

    assert result.metadata == {
        "format": "csv",
        "columns": ["name", "note"],
        "rows": 1,
    }
    assert f"Row 1: name: Alice | note: first{newline}second" in result.content
