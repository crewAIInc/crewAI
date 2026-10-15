from pathlib import Path

from crewai_tools.rag.loaders.jsonl_loader import JSONLLoader
from crewai_tools.rag.source_content import SourceContent
import pytest


@pytest.mark.parametrize("from_file", [False, True])
@pytest.mark.parametrize(
    "content, expected, count",
    [
        (
            '{"ok":true}\n[1,null]\n"café"\n42\nfalse\nnull',
            '{"ok":true}\n[1,null]\n"café"\n42\nfalse\nnull',
            6,
        ),
        ("1\n2", "1\n2", 2),
        ("1\n2\n", "1\n2", 2),
        ("1\r\n2", "1\n2", 2),
        ("1\r\n2\r\n", "1\n2", 2),
        ("", "", 0),
        (" \t1\r\n 2\t", "1\n2", 2),
        (
            "1e400\n0.123456789012345678901234567890",
            "1e400\n0.123456789012345678901234567890",
            2,
        ),
        ('"before\u2028after"', '"before\u2028after"', 1),
        ('"a\ufeffb"', '"a\ufeffb"', 1),
        (
            '{"values":[true,null,"NaN","Infinity"]}',
            '{"values":[true,null,"NaN","Infinity"]}',
            1,
        ),
    ],
)
def test_load_jsonl(
    tmp_path: Path, from_file: bool, content: str, expected: str, count: int
) -> None:
    """Preserve valid records and derive consistent metadata for file or inline input."""
    source = SourceContent(content)
    if from_file:
        path = tmp_path / "records.jsonl"
        path.write_bytes(content.encode("utf-8"))
        source = SourceContent(path)

    result = JSONLLoader().load(source)

    assert result.content == expected
    assert result.metadata == {"format": "jsonl", "record_count": count}
    assert result.source == (str(path) if from_file else source.source_ref)
    assert result.doc_id == JSONLLoader.generate_doc_id(result.source, expected)
    assert JSONLLoader().load(source).doc_id == result.doc_id


def test_doc_id_changes_with_file_content(tmp_path: Path) -> None:
    """Changing records at the same path must change the document ID."""
    path = tmp_path / "records.jsonl"
    path.write_text("1", encoding="utf-8")
    source = SourceContent(path)
    first = JSONLLoader().load(source)

    path.write_text("2", encoding="utf-8")

    assert JSONLLoader().load(source).doc_id != first.doc_id


@pytest.mark.parametrize("from_file", [False, True])
@pytest.mark.parametrize(
    "content, line_number",
    [
        ("1\n\n2", 2),
        ("1\n \n2", 2),
        ("1\n{bad}", 2),
        ("1\n{} {}", 2),
        ("\n", 1),
        (" \t", 1),
        ("\n1", 1),
        ("1\n\n", 2),
        ("1\r\n\r\n", 2),
        ("1\nNaN", 2),
        ("1\nInfinity", 2),
        ("1\n-Infinity", 2),
        ('1\n{"n":NaN}', 2),
        ("1\n[Infinity]", 2),
        ("\ufeff{}", 1),
    ],
)
def test_invalid_jsonl_record(
    tmp_path: Path, from_file: bool, content: str, line_number: int
) -> None:
    """Report the 1-based line containing a blank or invalid JSON record."""
    source = SourceContent(content)
    if from_file:
        path = tmp_path / "invalid.jsonl"
        path.write_bytes(content.encode("utf-8"))
        source = SourceContent(path)

    with pytest.raises(ValueError, match=rf"\bline {line_number}\b"):
        JSONLLoader().load(source)
