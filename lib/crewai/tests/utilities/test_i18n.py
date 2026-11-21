import json
from pathlib import Path

import pytest
from crewai.utilities.i18n import I18N


def test_load_prompts():
    i18n = I18N()
    i18n.load_prompts()
    assert i18n._prompts is not None


def test_slice():
    i18n = I18N()
    i18n.load_prompts()
    assert isinstance(i18n.slice("role_playing"), str)


def test_tools():
    i18n = I18N()
    i18n.load_prompts()
    assert isinstance(i18n.tools("ask_question"), str)


def test_retrieve():
    i18n = I18N()
    i18n.load_prompts()
    assert isinstance(i18n.retrieve("slices", "role_playing"), str)


def test_retrieve_not_found():
    i18n = I18N()
    i18n.load_prompts()
    with pytest.raises(Exception):
        i18n.retrieve("nonexistent_kind", "nonexistent_key")


def test_prompt_file():
    """Load prompt values from a custom JSON file."""
    import os

    path = os.path.join(os.path.dirname(__file__), "prompts.json")
    i18n = I18N(prompt_file=path)
    i18n.load_prompts()
    assert isinstance(i18n.retrieve("slices", "role_playing"), str)
    assert i18n.retrieve("slices", "role_playing") == "Lorem ipsum dolor sit amet"


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_custom_prompt_file_utf8(encoding: str, tmp_path: Path) -> None:
    """Accept an optional UTF-8 BOM while preserving prompt content."""
    prompt = "你是一位研究助手。\ufeff Preserve this content character."
    path = tmp_path / "prompts.json"
    path.write_text(
        json.dumps({"slices": {"role_playing": prompt}}, ensure_ascii=False),
        encoding=encoding,
    )

    i18n = I18N(prompt_file=str(path))

    assert i18n.slice("role_playing") == prompt


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig"])
def test_invalid_custom_prompt_json(encoding: str, tmp_path: Path) -> None:
    """Reject malformed custom JSON with or without a UTF-8 BOM."""
    path = tmp_path / "prompts.json"
    path.write_text('{"slices":', encoding=encoding)

    with pytest.raises(Exception, match="Error decoding JSON from the prompts file"):
        I18N(prompt_file=str(path))
