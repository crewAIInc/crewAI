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


def test_default_json_picker_selects_ace_without_manual_input():
    from crewai_cli.create_json_crew import _PROVIDERS, _default_model_for_provider, _select_model

    provider_index = next(i for i, (key, label) in enumerate(_PROVIDERS) if key == "acedatacloud")
    assert _PROVIDERS[provider_index][1] == "Ace Data Cloud"
    assert _default_model_for_provider("acedatacloud") == "acedatacloud/gpt-4.1-mini"
    with patch("crewai_cli.create_json_crew.pick_one", side_effect=[provider_index, 0]), \
         patch("crewai_cli.create_json_crew.get_provider_models", return_value=[("gpt-4.1-mini", "GPT-4.1 Mini")]) as catalog, \
         patch("crewai_cli.create_json_crew.click.prompt") as manual:
        assert _select_model() == "acedatacloud/gpt-4.1-mini"
    assert catalog.call_args.args == ("acedatacloud", [("gpt-4.1-mini", "GPT-4.1 Mini")])
    manual.assert_not_called()


def test_classic_creation_hides_ace_key(tmp_path):
    from crewai_cli.create_crew import create_crew

    with patch("crewai_cli.create_crew.create_folder_structure", return_value=(tmp_path, "test_crew", "TestCrew")), \
         patch("crewai_cli.create_crew.load_env_vars", return_value={}), \
         patch("crewai_cli.create_crew.is_dmn_mode_enabled", return_value=False), \
         patch("crewai_cli.create_crew.get_provider_data", return_value={"openai": ["gpt-4.1-mini"]}), \
         patch("crewai_cli.create_crew.select_provider", return_value="acedatacloud"), \
         patch("crewai_cli.create_crew.select_model", return_value="acedatacloud/gpt-4.1-mini"), \
         patch("crewai_cli.create_crew.click.prompt", return_value="test-key") as prompt, \
         patch("crewai_cli.create_crew.write_env_file") as write, \
         patch("crewai_cli.create_crew.copy_template"):
        create_crew("test", parent_folder=str(tmp_path))
    assert prompt.call_args.kwargs["hide_input"] is True
    assert write.call_args.args[1]["MODEL"] == "acedatacloud/gpt-4.1-mini"
    assert write.call_args.args[1]["ACEDATACLOUD_API_KEY"] == "test-key"
