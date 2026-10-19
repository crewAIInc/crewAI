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
    import os

    path = os.path.join(os.path.dirname(__file__), "prompts.json")
    i18n = I18N(prompt_file=path)
    i18n.load_prompts()
    assert isinstance(i18n.retrieve("slices", "role_playing"), str)
    assert i18n.retrieve("slices", "role_playing") == "Lorem ipsum dolor sit amet"


def test_prompt_file_utf8_with_bom(tmp_path):
    import json

    file_path = tmp_path / "prompts_bom.json"
    content = json.dumps({"slices": {"role_playing": "Research assistant"}})
    file_path.write_text(content, encoding="utf-8-sig")

    i18n = I18N(prompt_file=str(file_path))
    assert i18n.slice("role_playing") == "Research assistant"

