from crewai_tools.rag.loaders.json_loader import JSONLoader
from crewai_tools.rag.source_content import SourceContent


def test_json_loader_keeps_non_ascii_text_readable() -> None:
    raw = '{"city": "Málaga", "note": "日本語"}'

    result = JSONLoader().load(SourceContent(raw))

    assert "Málaga" in result.content
    assert "日本語" in result.content
    assert "\\u" not in result.content


def test_json_loader_list_keeps_non_ascii_text_readable() -> None:
    result = JSONLoader().load(SourceContent('["café", "naïve"]'))

    assert "café" in result.content
    assert "\\u" not in result.content
