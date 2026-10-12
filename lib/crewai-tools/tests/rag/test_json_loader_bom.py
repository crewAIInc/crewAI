from pathlib import Path

from crewai_tools.rag.base_loader import LoaderResult
from crewai_tools.rag.loaders.json_loader import JSONLoader
from crewai_tools.rag.source_content import SourceContent


def _load(tmp_path: Path, content: bytes) -> LoaderResult:
    """Load exact local bytes through the public loader entry point."""
    path = tmp_path / "data.json"
    path.write_bytes(content)
    return JSONLoader().load(SourceContent(path))


def test_json_loader_parses_utf8_bom_object(tmp_path: Path) -> None:
    """A local UTF-8 BOM must not turn valid JSON into raw parse-error text."""
    result = _load(tmp_path, b'\xef\xbb\xbf{"message":"hello"}')
    assert result.metadata == {"format": "json", "type": "dict", "size": 1}
    assert result.content == 'message: "hello"'
    assert result.source == str(tmp_path / "data.json")
    assert result.doc_id == JSONLoader.generate_doc_id(result.source, result.content)


def test_json_loader_parses_utf8_bom_array(tmp_path: Path) -> None:
    """BOM-prefixed arrays retain the same structured output as plain JSON."""
    result = _load(tmp_path, b"\xef\xbb\xbf[1, 2]")
    assert result.metadata == {"format": "json", "type": "list", "size": 2}
    assert result.content == "1\n2"


def test_json_loader_preserves_internal_bom_and_malformed_fallback(
    tmp_path: Path,
) -> None:
    """Only a leading BOM is removed; content and malformed fallback stay intact."""
    internal = _load(tmp_path, b'{"value":"a\xef\xbb\xbf b"}')
    malformed = _load(tmp_path, b"\xef\xbb\xbf{invalid json}")
    assert "a\\ufeff b" in internal.content
    assert "parse_error" in malformed.metadata
    assert malformed.content == "{invalid json}"
