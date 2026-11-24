import json

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


def test_prompt_file_merges_with_defaults(tmp_path):
    """A partial custom prompt file keeps every default it does not override."""
    prompt_file = tmp_path / "custom_prompts.json"
    prompt_file.write_text(
        json.dumps({"slices": {"format": "CUSTOM FORMAT"}}), encoding="utf-8"
    )

    i18n = I18N(prompt_file=str(prompt_file))

    assert i18n.slice("format") == "CUSTOM FORMAT"
    assert i18n.slice("role_playing") == I18N().slice("role_playing")
    assert i18n.tools("ask_question") == I18N().tools("ask_question")
