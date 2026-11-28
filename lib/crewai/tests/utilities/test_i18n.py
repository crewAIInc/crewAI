import json
import os

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


def test_prompt_file_utf8_bom(tmp_path):
    """A prompt file saved with a UTF-8 BOM must load without error (#8259)."""
    payload = {"slices": {"role_playing": "BOM test prompt"}}
    path = tmp_path / "prompts_bom.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps(payload).encode("utf-8"))

    i18n = I18N(prompt_file=str(path))
    i18n.load_prompts()
    assert i18n.retrieve("slices", "role_playing") == "BOM test prompt"


def test_prompt_file_malformed_json_still_fails(tmp_path):
    """A BOM-prefixed file with invalid JSON must still raise."""
    path = tmp_path / "bad.json"
    path.write_bytes(b"\xef\xbb\xbf" + b"{not valid json}")

    with pytest.raises(Exception, match="Error decoding JSON"):
        I18N(prompt_file=str(path)).load_prompts()
