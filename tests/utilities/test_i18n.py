import pytest

from crewai.utilities.i18n import I18N


def test_load_translation():
    i18n = I18N(language="en")
    i18n.load_translation()
    assert i18n._translations is not None


def test_slice():
    i18n = I18N(language="en")
    i18n.load_translation()
    assert isinstance(i18n.slice("role_playing"), str)


def test_errors():
    i18n = I18N(language="en")
    i18n.load_translation()
    assert isinstance(i18n.errors("unexpected_format"), str)


def test_tools():
    i18n = I18N(language="en")
    i18n.load_translation()
    assert isinstance(i18n.tools("ask_question"), str)


def test_retrieve():
    i18n = I18N(language="en")
    i18n.load_translation()
    assert isinstance(i18n.retrieve("slices", "role_playing"), str)


def test_retrieve_not_found():
    i18n = I18N(language="en")
    i18n.load_translation()
    with pytest.raises(Exception):
        i18n.retrieve("nonexistent_kind", "nonexistent_key")
