"""CLI discovery for Ace Data Cloud; no network or secrets."""

from unittest.mock import patch

from crewai_cli.constants import ENV_VARS, MODELS, PROVIDERS
from crewai_cli.provider import select_provider


def test_ace_is_a_named_provider_choice():
    with patch("crewai_cli.provider.select_choice", return_value="Ace Data Cloud") as choose:
        assert select_provider({"openai": ["gpt-4.1-mini"]}) == "acedatacloud"
    assert "Ace Data Cloud" in choose.call_args.args[1]
    assert "acedatacloud" in PROVIDERS
    assert MODELS["acedatacloud"] == ["acedatacloud/gpt-4.1-mini"]
    assert ENV_VARS["acedatacloud"][0]["key_name"] == "ACEDATACLOUD_API_KEY"
    assert ENV_VARS["acedatacloud"][0]["hide_input"] is True


def test_json_setup_hides_api_key(tmp_path):
    from crewai_cli.create_json_crew import _setup_env

    with patch("crewai_cli.create_json_crew.load_env_vars", return_value={}), \
         patch("crewai_cli.create_json_crew.click.prompt", return_value="test-key") as prompt, \
         patch("crewai_cli.create_json_crew.write_env_file") as write:
        _setup_env(tmp_path, "acedatacloud/gpt-4.1-mini")
    assert prompt.call_args.kwargs["hide_input"] is True
    assert write.call_args.args[1]["ACEDATACLOUD_API_KEY"] == "test-key"
