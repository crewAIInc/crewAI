import pytest

from crewai_tools.rag.loaders.json_loader import JSONLoader
from crewai_tools.rag.source_content import SourceContent


def test_json_loader_keeps_non_ascii_text_readable() -> None:
    """Object values keep readable Unicode characters."""
    raw = '{"city": "Málaga", "note": "日本語"}'

    result = JSONLoader().load(SourceContent(raw))

    assert "Málaga" in result.content
    assert "日本語" in result.content
    assert "\\u" not in result.content


def test_json_loader_list_keeps_non_ascii_text_readable() -> None:
    """List items keep readable Unicode characters."""
    result = JSONLoader().load(SourceContent('["café", "naïve"]'))

    assert "café" in result.content
    assert "\\u" not in result.content


def test_json_loader_scalar_keeps_non_ascii_text_readable() -> None:
    """Scalar strings keep readable Unicode characters."""
    result = JSONLoader().load(SourceContent('"Málaga"'))

    assert result.content == '"Málaga"'


@pytest.mark.parametrize("surrogate", [r"\ud800", r"\udfff"])
@pytest.mark.parametrize("shape", ["object", "list", "scalar"])
def test_json_loader_lone_surrogate_stays_escaped(surrogate: str, shape: str) -> None:
    """Lone surrogate escapes remain ingestible without losing their value."""
    value = '"café ' + surrogate + '"'
    raw = {
        "object": '{"value": ' + value + '}',
        "list": '[' + value + ']',
        "scalar": value,
    }[shape]

    result = JSONLoader().load(SourceContent(raw))

    assert surrogate in result.content
    assert "café" in result.content
    assert result.content.encode("utf-8")
    assert result.doc_id


def test_json_loader_paired_surrogate_keeps_emoji_readable() -> None:
    """A valid surrogate pair is rendered as its Unicode character."""
    result = JSONLoader().load(SourceContent(r'"\ud83d\ude00"'))

    assert result.content == '"😀"'
