import io
import os
import zipfile
from unittest.mock import MagicMock, patch

import httpx
import pytest
from click.testing import CliRunner

from crewai.cli.cli import template_add, template_list
from crewai.cli.remote_template.main import TemplateCommand


@pytest.fixture
def runner():
    return CliRunner()


SAMPLE_REPOS = [
    {"name": "template_deep_research", "description": "Deep research template", "private": False},
    {"name": "template_pull_request_review", "description": "PR review template", "private": False},
    {"name": "template_conversational_example", "description": "Conversational demo", "private": False},
    {"name": "crewai", "description": "Main repo", "private": False},
    {"name": "marketplace-crew-template", "description": "Marketplace", "private": False},
]


def _make_zipball(files: dict[str, str], top_dir: str = "crewAIInc-template_test-abc123") -> bytes:
    """Create an in-memory zipball mimicking GitHub's format."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(f"{top_dir}/", "")
        for path, content in files.items():
            zf.writestr(f"{top_dir}/{path}", content)
    return buf.getvalue()


# --- CLI command tests ---


@patch("crewai.cli.cli.TemplateCommand")
def test_template_list_command(mock_cls, runner):
    mock_instance = MagicMock()
    mock_cls.return_value = mock_instance

    result = runner.invoke(template_list)

    assert result.exit_code == 0
    mock_cls.assert_called_once()
    mock_instance.list_templates.assert_called_once()


@patch("crewai.cli.cli.TemplateCommand")
def test_template_add_command(mock_cls, runner):
    mock_instance = MagicMock()
    mock_cls.return_value = mock_instance

    result = runner.invoke(template_add, ["deep_research"])

    assert result.exit_code == 0
    mock_cls.assert_called_once()
    mock_instance.add_template.assert_called_once_with("deep_research", None)


@patch("crewai.cli.cli.TemplateCommand")
def test_template_add_with_output_dir(mock_cls, runner):
    mock_instance = MagicMock()
    mock_cls.return_value = mock_instance

    result = runner.invoke(template_add, ["deep_research", "-o", "my_project"])

    assert result.exit_code == 0
    mock_instance.add_template.assert_called_once_with("deep_research", "my_project")


# --- TemplateCommand unit tests ---


class TestTemplateCommand:
    @pytest.fixture
    def cmd(self):
        with patch.object(TemplateCommand, "__init__", return_value=None):
            instance = TemplateCommand()
            instance._telemetry = MagicMock()
            return instance

    @patch("crewai.cli.remote_template.main.httpx.get")
    def test_fetch_templates_filters_by_prefix(self, mock_get, cmd):
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_REPOS
        mock_response.raise_for_status = MagicMock()
        # Return empty on page 2 to stop pagination
        mock_empty = MagicMock()
        mock_empty.json.return_value = []
        mock_empty.raise_for_status = MagicMock()
        mock_get.side_effect = [mock_response, mock_empty]

        templates = cmd._fetch_templates()

        assert len(templates) == 3
        assert all(t["name"].startswith("template_") for t in templates)

    @patch("crewai.cli.remote_template.main.httpx.get")
    def test_fetch_templates_excludes_private(self, mock_get, cmd):
        repos = [
            {"name": "template_private_one", "description": "", "private": True},
            {"name": "template_public_one", "description": "", "private": False},
        ]
        mock_response = MagicMock()
        mock_response.json.return_value = repos
        mock_response.raise_for_status = MagicMock()
        mock_empty = MagicMock()
        mock_empty.json.return_value = []
        mock_empty.raise_for_status = MagicMock()
        mock_get.side_effect = [mock_response, mock_empty]

        templates = cmd._fetch_templates()

        assert len(templates) == 1
        assert templates[0]["name"] == "template_public_one"

    @patch("crewai.cli.remote_template.main.httpx.get")
    def test_fetch_templates_api_error(self, mock_get, cmd):
        mock_get.side_effect = httpx.HTTPError("connection error")

        with pytest.raises(SystemExit):
            cmd._fetch_templates()

    @patch("crewai.cli.remote_template.main.click.prompt", return_value="q")
    @patch("crewai.cli.remote_template.main.httpx.get")
    def test_list_templates_prints_output(self, mock_get, mock_prompt, cmd):
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_REPOS
        mock_response.raise_for_status = MagicMock()
        mock_empty = MagicMock()
        mock_empty.json.return_value = []
        mock_empty.raise_for_status = MagicMock()
        mock_get.side_effect = [mock_response, mock_empty]

        with patch("crewai.cli.remote_template.main.console") as mock_console:
            cmd.list_templates()
            assert mock_console.print.call_count > 0

    @patch("crewai.cli.remote_template.main.httpx.get")
    def test_resolve_repo_name_with_prefix(self, mock_get, cmd):
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_REPOS
        mock_response.raise_for_status = MagicMock()
        mock_empty = MagicMock()
        mock_empty.json.return_value = []
        mock_empty.raise_for_status = MagicMock()
        mock_get.side_effect = [mock_response, mock_empty]

        result = cmd._resolve_repo_name("template_deep_research")
        assert result == "template_deep_research"

    @patch("crewai.cli.remote_template.main.httpx.get")
    def test_resolve_repo_name_without_prefix(self, mock_get, cmd):
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_REPOS
        mock_response.raise_for_status = MagicMock()
        mock_empty = MagicMock()
        mock_empty.json.return_value = []
        mock_empty.raise_for_status = MagicMock()
        mock_get.side_effect = [mock_response, mock_empty]

        result = cmd._resolve_repo_name("deep_research")
        assert result == "template_deep_research"

    @patch("crewai.cli.remote_template.main.httpx.get")
    def test_resolve_repo_name_not_found(self, mock_get, cmd):
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_REPOS
        mock_response.raise_for_status = MagicMock()
        mock_empty = MagicMock()
        mock_empty.json.return_value = []
        mock_empty.raise_for_status = MagicMock()
        mock_get.side_effect = [mock_response, mock_empty]

        result = cmd._resolve_repo_name("nonexistent")
        assert result is None

    def test_extract_zip(self, cmd, tmp_path):
        files = {
            "README.md": "# Test Template",
            "src/main.py": "print('hello')",
            "config/settings.yaml": "key: value",
        }
        zip_bytes = _make_zipball(files)
        dest = str(tmp_path / "output")

        cmd._extract_zip(zip_bytes, dest)

        assert os.path.isfile(os.path.join(dest, "README.md"))
        assert os.path.isfile(os.path.join(dest, "src", "main.py"))
        assert os.path.isfile(os.path.join(dest, "config", "settings.yaml"))

        with open(os.path.join(dest, "src", "main.py")) as f:
            assert f.read() == "print('hello')"

    @patch.object(TemplateCommand, "_extract_zip")
    @patch.object(TemplateCommand, "_download_zip")
    @patch.object(TemplateCommand, "_resolve_repo_name")
    def test_add_template_success(self, mock_resolve, mock_download, mock_extract, cmd, tmp_path):
        mock_resolve.return_value = "template_deep_research"
        mock_download.return_value = b"fake-zip-bytes"

        os.chdir(tmp_path)
        cmd.add_template("deep_research")

        mock_resolve.assert_called_once_with("deep_research")
        mock_download.assert_called_once_with("template_deep_research")
        expected_dest = os.path.join(str(tmp_path), "deep_research")
        mock_extract.assert_called_once_with(b"fake-zip-bytes", expected_dest)

    @patch.object(TemplateCommand, "_resolve_repo_name")
    def test_add_template_not_found(self, mock_resolve, cmd):
        mock_resolve.return_value = None

        with pytest.raises(SystemExit):
            cmd.add_template("nonexistent")

    @patch.object(TemplateCommand, "_extract_zip")
    @patch.object(TemplateCommand, "_download_zip")
    @patch("crewai.cli.remote_template.main.click.prompt", return_value="my_project")
    @patch.object(TemplateCommand, "_resolve_repo_name")
    def test_add_template_dir_exists_prompts_rename(self, mock_resolve, mock_prompt, mock_download, mock_extract, cmd, tmp_path):
        mock_resolve.return_value = "template_deep_research"
        mock_download.return_value = b"fake-zip-bytes"
        existing = tmp_path / "deep_research"
        existing.mkdir()

        os.chdir(tmp_path)
        cmd.add_template("deep_research")

        expected_dest = os.path.join(str(tmp_path), "my_project")
        mock_extract.assert_called_once_with(b"fake-zip-bytes", expected_dest)

    @patch.object(TemplateCommand, "_resolve_repo_name")
    @patch("crewai.cli.remote_template.main.click.prompt", return_value="q")
    def test_add_template_dir_exists_quit(self, mock_prompt, mock_resolve, cmd, tmp_path):
        mock_resolve.return_value = "template_deep_research"
        existing = tmp_path / "deep_research"
        existing.mkdir()

        os.chdir(tmp_path)
        cmd.add_template("deep_research")
        # Should return without downloading

    @patch.object(TemplateCommand, "_install_repo")
    @patch("crewai.cli.remote_template.main.click.prompt", return_value="2")
    @patch("crewai.cli.remote_template.main.httpx.get")
    def test_list_templates_selects_and_installs(self, mock_get, mock_prompt, mock_install, cmd):
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_REPOS
        mock_response.raise_for_status = MagicMock()
        mock_empty = MagicMock()
        mock_empty.json.return_value = []
        mock_empty.raise_for_status = MagicMock()
        mock_get.side_effect = [mock_response, mock_empty]

        with patch("crewai.cli.remote_template.main.console"):
            cmd.list_templates()

        # Templates are sorted by name; index 1 (choice "2") = template_deep_research
        mock_install.assert_called_once_with("template_deep_research")

    @patch.object(TemplateCommand, "_install_repo")
    @patch("crewai.cli.remote_template.main.click.prompt", return_value="q")
    @patch("crewai.cli.remote_template.main.httpx.get")
    def test_list_templates_quit(self, mock_get, mock_prompt, mock_install, cmd):
        mock_response = MagicMock()
        mock_response.json.return_value = SAMPLE_REPOS
        mock_response.raise_for_status = MagicMock()
        mock_empty = MagicMock()
        mock_empty.json.return_value = []
        mock_empty.raise_for_status = MagicMock()
        mock_get.side_effect = [mock_response, mock_empty]

        with patch("crewai.cli.remote_template.main.console"):
            cmd.list_templates()

        mock_install.assert_not_called()
