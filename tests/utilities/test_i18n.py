import pytest

from crewai.utilities.i18n import I18N


def test_load_prompts() -> None:
    i18n = I18N()
    i18n.load_prompts()
    assert i18n._prompts is not None


def test_slice() -> None:
    i18n = I18N()
    i18n.load_prompts()
    assert isinstance(i18n.slice("role_playing"), str)


def test_tools() -> None:
    i18n = I18N()
    i18n.load_prompts()
    assert isinstance(i18n.tools("ask_question"), str)


def test_retrieve() -> None:
    i18n = I18N()
    i18n.load_prompts()
    assert isinstance(i18n.retrieve("slices", "role_playing"), str)


def test_retrieve_not_found() -> None:
    i18n = I18N()
    i18n.load_prompts()
    with pytest.raises(Exception):
        i18n.retrieve("nonexistent_kind", "nonexistent_key")


def test_prompt_file() -> None:
    import os

    path = os.path.join(os.path.dirname(__file__), "prompts.json")
    i18n = I18N(prompt_file=path)
    i18n.load_prompts()
    assert isinstance(i18n.retrieve("slices", "role_playing"), str)
    assert i18n.retrieve("slices", "role_playing") == "Lorem ipsum dolor sit amet"
