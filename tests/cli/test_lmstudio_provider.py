"""Tests for LMStudio provider integration in the CLI."""

import shutil
import tempfile
from pathlib import Path
from unittest import mock

import pytest

from crewai.cli.constants import ENV_VARS, PROVIDERS
from crewai.cli.create_crew import create_crew


class TestLMStudioConstants:
    """Tests verifying LMStudio is properly defined in CLI constants."""

    def test_lmstudio_in_providers_list(self):
        assert "lmstudio" in PROVIDERS

    def test_lmstudio_env_vars_defined(self):
        assert "lmstudio" in ENV_VARS

    def test_lmstudio_env_vars_has_model_entry(self):
        lmstudio_vars = ENV_VARS["lmstudio"]
        model_entry = next(
            (d for d in lmstudio_vars if d.get("key_name") == "MODEL"), None
        )
        assert model_entry is not None
        assert "prompt" in model_entry

    def test_lmstudio_env_vars_has_api_base_entry(self):
        lmstudio_vars = ENV_VARS["lmstudio"]
        api_base_entry = next(
            (d for d in lmstudio_vars if d.get("key_name") == "OPENAI_API_BASE"), None
        )
        assert api_base_entry is not None
        assert api_base_entry["default_value"] == "http://localhost:1234/v1"

    def test_lmstudio_env_vars_has_api_key_entry(self):
        lmstudio_vars = ENV_VARS["lmstudio"]
        api_key_entry = next(
            (d for d in lmstudio_vars if d.get("key_name") == "OPENAI_API_KEY"), None
        )
        assert api_key_entry is not None
        assert "prompt" in api_key_entry


class TestLMStudioCreateCrew:
    """Tests verifying the create crew flow works with LMStudio provider."""

    @pytest.fixture
    def temp_dir(self):
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d)

    @mock.patch("crewai.cli.create_crew.get_provider_data")
    @mock.patch("crewai.cli.create_crew.select_provider")
    @mock.patch("click.prompt")
    @mock.patch("click.confirm", return_value=False)
    def test_create_crew_with_lmstudio_all_values(
        self,
        mock_confirm,
        mock_prompt,
        mock_select_provider,
        mock_get_provider_data,
        temp_dir,
    ):
        """Test creating a crew with LMStudio provider setting all values."""
        mock_get_provider_data.return_value = {"lmstudio": []}
        mock_select_provider.return_value = "lmstudio"

        # Simulate user inputs for: model name, api base url, api key
        mock_prompt.side_effect = [
            "my-local-model",
            "http://localhost:1234/v1",
            "lm-studio-key",
        ]

        create_crew("test_crew", provider="lmstudio", parent_folder=str(temp_dir))

        # Check the .env file was created with correct values
        env_file = temp_dir / "test_crew" / ".env"
        assert env_file.exists()
        env_content = env_file.read_text()
        assert "MODEL=my-local-model" in env_content
        assert "OPENAI_API_BASE=http://localhost:1234/v1" in env_content
        assert "OPENAI_API_KEY=lm-studio-key" in env_content

    @mock.patch("crewai.cli.create_crew.get_provider_data")
    @mock.patch("crewai.cli.create_crew.select_provider")
    @mock.patch("click.prompt")
    @mock.patch("click.confirm", return_value=False)
    def test_create_crew_with_lmstudio_default_base_url(
        self,
        mock_confirm,
        mock_prompt,
        mock_select_provider,
        mock_get_provider_data,
        temp_dir,
    ):
        """Test creating a crew with LMStudio using default base URL."""
        mock_get_provider_data.return_value = {"lmstudio": []}
        mock_select_provider.return_value = "lmstudio"

        # User provides model, accepts default URL, skips API key
        mock_prompt.side_effect = ["my-local-model", "http://localhost:1234/v1", ""]

        create_crew("test_crew", provider="lmstudio", parent_folder=str(temp_dir))

        env_file = temp_dir / "test_crew" / ".env"
        assert env_file.exists()
        env_content = env_file.read_text()
        assert "MODEL=my-local-model" in env_content
        assert "OPENAI_API_BASE=http://localhost:1234/v1" in env_content
        # API key should not be present when skipped
        assert "OPENAI_API_KEY" not in env_content

    @mock.patch("crewai.cli.create_crew.get_provider_data")
    @mock.patch("crewai.cli.create_crew.select_provider")
    @mock.patch("click.prompt")
    @mock.patch("click.confirm", return_value=False)
    def test_create_crew_with_lmstudio_custom_base_url(
        self,
        mock_confirm,
        mock_prompt,
        mock_select_provider,
        mock_get_provider_data,
        temp_dir,
    ):
        """Test creating a crew with LMStudio using a custom base URL."""
        mock_get_provider_data.return_value = {"lmstudio": []}
        mock_select_provider.return_value = "lmstudio"

        # User provides model, custom URL, and API key
        mock_prompt.side_effect = [
            "my-local-model",
            "http://192.168.1.100:1234/v1",
            "my-key",
        ]

        create_crew("test_crew", provider="lmstudio", parent_folder=str(temp_dir))

        env_file = temp_dir / "test_crew" / ".env"
        assert env_file.exists()
        env_content = env_file.read_text()
        assert "MODEL=my-local-model" in env_content
        assert "OPENAI_API_BASE=http://192.168.1.100:1234/v1" in env_content
        assert "OPENAI_API_KEY=my-key" in env_content

    @mock.patch("crewai.cli.create_crew.get_provider_data")
    @mock.patch("crewai.cli.create_crew.select_provider")
    @mock.patch("click.prompt")
    @mock.patch("click.confirm", return_value=False)
    def test_create_crew_with_lmstudio_no_model(
        self,
        mock_confirm,
        mock_prompt,
        mock_select_provider,
        mock_get_provider_data,
        temp_dir,
    ):
        """Test creating a crew with LMStudio when user skips model name."""
        mock_get_provider_data.return_value = {"lmstudio": []}
        mock_select_provider.return_value = "lmstudio"

        # User skips model, accepts default URL, skips API key
        mock_prompt.side_effect = ["", "http://localhost:1234/v1", ""]

        create_crew("test_crew", provider="lmstudio", parent_folder=str(temp_dir))

        env_file = temp_dir / "test_crew" / ".env"
        assert env_file.exists()
        env_content = env_file.read_text()
        # MODEL should not be present when skipped
        assert "MODEL=" not in env_content
        assert "OPENAI_API_BASE=http://localhost:1234/v1" in env_content


class TestDefaultValueInPrompt:
    """Tests verifying the default_value mechanism in env var prompts."""

    @pytest.fixture
    def temp_dir(self):
        d = tempfile.mkdtemp()
        yield Path(d)
        shutil.rmtree(d)

    @mock.patch("crewai.cli.create_crew.get_provider_data")
    @mock.patch("crewai.cli.create_crew.select_provider")
    @mock.patch("click.prompt")
    @mock.patch("click.confirm", return_value=False)
    def test_default_value_is_used_when_user_accepts(
        self,
        mock_confirm,
        mock_prompt,
        mock_select_provider,
        mock_get_provider_data,
        temp_dir,
    ):
        """Test that default_value is written to .env when user accepts it."""
        mock_get_provider_data.return_value = {"lmstudio": []}
        mock_select_provider.return_value = "lmstudio"

        # User enters model, accepts default base URL, enters API key
        mock_prompt.side_effect = [
            "test-model",
            "http://localhost:1234/v1",
            "test-key",
        ]

        create_crew("test_crew", provider="lmstudio", parent_folder=str(temp_dir))

        env_file = temp_dir / "test_crew" / ".env"
        env_content = env_file.read_text()
        assert "OPENAI_API_BASE=http://localhost:1234/v1" in env_content
