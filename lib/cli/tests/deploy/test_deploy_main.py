import sys
import unittest
from io import StringIO
from pathlib import Path
import subprocess
from unittest.mock import MagicMock, Mock, patch

import pytest
import json

import crewai_cli.deploy.main as deploy_main
import httpx
from crewai_cli.deploy.validate import Severity, ValidationResult
from crewai_cli.utils import parse_toml


def test_ensure_lockfile_for_deploy_runs_install_when_lock_missing(
    monkeypatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    calls = []

    def fake_install_crew(proxy_options, *, raise_on_error=False):
        calls.append((proxy_options, raise_on_error))

    monkeypatch.setattr("crewai_cli.install_crew.install_crew", fake_install_crew)

    deploy_main._ensure_lockfile_for_deploy()

    assert calls == [([], True)]


def test_ensure_lockfile_for_deploy_skips_when_lock_exists(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    (tmp_path / "uv.lock").write_text("# lock\n")
    calls = []

    def fake_install_crew(proxy_options, *, raise_on_error=False):
        calls.append((proxy_options, raise_on_error))

    monkeypatch.setattr("crewai_cli.install_crew.install_crew", fake_install_crew)

    deploy_main._ensure_lockfile_for_deploy()

    assert calls == []


def test_ensure_lockfile_for_deploy_skips_without_pyproject(
    monkeypatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    calls = []

    def fake_install_crew(proxy_options, *, raise_on_error=False):
        calls.append((proxy_options, raise_on_error))

    monkeypatch.setattr("crewai_cli.install_crew.install_crew", fake_install_crew)

    deploy_main._ensure_lockfile_for_deploy()

    assert calls == []


def test_ensure_lockfile_for_deploy_failure_exits_nonzero(
    monkeypatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")

    def fake_install_crew(proxy_options, *, raise_on_error=False):
        raise subprocess.CalledProcessError(42, ["uv", "sync"])

    monkeypatch.setattr("crewai_cli.install_crew.install_crew", fake_install_crew)

    with pytest.raises(SystemExit) as exc_info:
        deploy_main._ensure_lockfile_for_deploy()

    assert exc_info.value.code == 42


class _FakeDeployValidator:
    def __init__(self, results: list[ValidationResult]):
        self.results = results

    @property
    def errors(self) -> list[ValidationResult]:
        return [
            result
            for result in self.results
            if result.severity is Severity.ERROR
        ]

    def run(self) -> list[ValidationResult]:
        return self.results


def test_prepare_project_for_deploy_blocks_install_when_validation_fails(
    monkeypatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    install_calls = []
    rendered_results = []
    missing_lockfile = ValidationResult(
        Severity.ERROR,
        "missing_lockfile",
        "Expected to find a lockfile",
    )
    invalid_config = ValidationResult(
        Severity.ERROR,
        "invalid_crew_json",
        "crew.jsonc has invalid configuration",
    )

    monkeypatch.setattr(
        deploy_main,
        "DeployValidator",
        lambda: _FakeDeployValidator([missing_lockfile, invalid_config]),
    )
    monkeypatch.setattr(deploy_main, "render_report", rendered_results.append)

    def fake_install_crew(proxy_options, *, raise_on_error=False):
        install_calls.append((proxy_options, raise_on_error))

    monkeypatch.setattr("crewai_cli.install_crew.install_crew", fake_install_crew)

    assert deploy_main._prepare_project_for_deploy(skip_validate=False) is False

    assert install_calls == []
    assert [[result.code for result in results] for results in rendered_results] == [
        ["invalid_crew_json"]
    ]


def test_prepare_project_for_deploy_creates_missing_lock_after_validation(
    monkeypatch, tmp_path: Path
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    install_calls = []
    missing_lockfile = ValidationResult(
        Severity.ERROR,
        "missing_lockfile",
        "Expected to find a lockfile",
    )
    validators = [
        _FakeDeployValidator([missing_lockfile]),
        _FakeDeployValidator([]),
    ]

    def fake_validator():
        return validators.pop(0)

    def fake_install_crew(proxy_options, *, raise_on_error=False):
        install_calls.append((proxy_options, raise_on_error))
        (tmp_path / "uv.lock").write_text("# lock\n")

    monkeypatch.setattr(deploy_main, "DeployValidator", fake_validator)
    monkeypatch.setattr(deploy_main, "render_report", lambda results: None)
    monkeypatch.setattr("crewai_cli.install_crew.install_crew", fake_install_crew)

    assert deploy_main._prepare_project_for_deploy(skip_validate=False) is True

    assert install_calls == [([], True)]
    assert validators == []


def test_deployment_page_url_prefers_deployment_id():
    assert (
        deploy_main._deployment_page_url(
            "https://app.crewai.com",
            {"uuid": "crew-uuid", "deployment_id": 128687},
        )
        == "https://app.crewai.com/crewai_plus/deployments/128687"
    )


def test_deployment_page_url_prefers_nested_deployment_id_over_crew_uuid():
    assert (
        deploy_main._deployment_page_url(
            "https://app.crewai.com",
            {"uuid": "crew-uuid", "deployment": {"deployment_id": 128687}},
        )
        == "https://app.crewai.com/crewai_plus/deployments/128687"
    )


def test_deployment_page_url_does_not_use_uuid_as_deployment_id():
    assert (
        deploy_main._deployment_page_url(
            "https://app.crewai.com/",
            {"uuid": "crew-uuid", "deployment": {"uuid": "deployment-uuid"}},
        )
        is None
    )


class TestDeployCommand(unittest.TestCase):
    @patch("crewai_cli.command.get_auth_token")
    @patch("crewai_cli.deploy.main.get_project_name")
    @patch("crewai_cli.command.PlusAPI")
    def setUp(
        self,
        mock_plus_api,
        mock_get_project_name,
        mock_get_auth_token,
    ):
        self.mock_get_auth_token = mock_get_auth_token
        self.mock_get_project_name = mock_get_project_name
        self.mock_plus_api = mock_plus_api

        self.mock_get_auth_token.return_value = "test_token"
        self.mock_get_project_name.return_value = "test_project"

        self.deploy_command = deploy_main.DeployCommand()
        self.mock_client = self.deploy_command.plus_api_client
        self.mock_client.base_url = "https://app.crewai.com"
        self.mock_browser_open_patcher = patch(
            "crewai_cli.deploy.main.webbrowser.open"
        )
        self.mock_browser_open = self.mock_browser_open_patcher.start()
        self.addCleanup(self.mock_browser_open_patcher.stop)

    def test_init_success(self):
        self.assertEqual(self.deploy_command.project_name, "test_project")
        self.mock_plus_api.assert_called_once_with(api_key="test_token")

    @patch("builtins.input")
    def test_confirm_zip_input_only_confirms_env_vars(self, mock_input):
        self.deploy_command._confirm_zip_input({"MODEL": "openai/gpt-5"}, False)

        mock_input.assert_called_once_with("Press Enter to continue with 1 env vars: MODEL")

    @patch("crewai_cli.command.get_auth_token")
    def test_init_failure(self, mock_get_auth_token):
        mock_get_auth_token.side_effect = Exception("Auth failed")

        with self.assertRaises(SystemExit):
            deploy_main.DeployCommand()

    def test_validate_response_successful_response(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {"message": "Success"}
        mock_response.status_code = 200
        mock_response.is_success = True

        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command._validate_response(mock_response)
            assert fake_out.getvalue() == ""

    def test_validate_response_json_decode_error(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.side_effect = json.JSONDecodeError("Decode error", "", 0)
        mock_response.status_code = 500
        mock_response.content = b"Invalid JSON"

        with patch("sys.stdout", new=StringIO()) as fake_out:
            with pytest.raises(SystemExit):
                self.deploy_command._validate_response(mock_response)
            output = fake_out.getvalue()
            assert (
                "Failed to parse response from Enterprise API failed. Details:"
                in output
            )
            assert "Status Code: 500" in output
            assert "Response:\nInvalid JSON" in output

    def test_validate_response_422_error(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {
            "field1": ["Error message 1"],
            "field2": ["Error message 2"],
        }
        mock_response.status_code = 422
        mock_response.is_success = False

        with patch("sys.stdout", new=StringIO()) as fake_out:
            with pytest.raises(SystemExit):
                self.deploy_command._validate_response(mock_response)
            output = fake_out.getvalue()
            assert (
                "Failed to complete operation. Please fix the following errors:"
                in output
            )
            assert "Field1 Error message 1" in output
            assert "Field2 Error message 2" in output

    def test_validate_response_other_error(self):
        mock_response = Mock(spec=httpx.Response)
        mock_response.json.return_value = {"error": "Something went wrong"}
        mock_response.status_code = 500
        mock_response.is_success = False

        with patch("sys.stdout", new=StringIO()) as fake_out:
            with pytest.raises(SystemExit):
                self.deploy_command._validate_response(mock_response)
            output = fake_out.getvalue()
            assert "Request to Enterprise API failed. Details:" in output
            assert "Details:\nSomething went wrong" in output

    def test_standard_no_param_error_message(self):
        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command._standard_no_param_error_message()
            self.assertIn("No UUID provided", fake_out.getvalue())

    def test_display_deployment_info(self):
        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command._display_deployment_info(
                {"uuid": "test-uuid", "id": 128687, "status": "deployed"}
            )
            self.assertIn("Deploying the crew...", fake_out.getvalue())
            self.assertIn("test-uuid", fake_out.getvalue())
            self.assertIn("deployed", fake_out.getvalue())
            self.assertIn(
                "https://app.crewai.com/crewai_plus/deployments/128687",
                fake_out.getvalue(),
            )
        self.mock_browser_open.assert_called_once_with(
            "https://app.crewai.com/crewai_plus/deployments/128687"
        )

    def test_display_deployment_info_warns_when_browser_open_returns_false(self):
        self.mock_browser_open.return_value = False

        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command._display_deployment_info(
                {"uuid": "test-uuid", "id": 128687, "status": "deployed"}
            )
            self.assertIn(
                "Could not open the deployment page automatically.",
                fake_out.getvalue(),
            )

        self.mock_browser_open.assert_called_once_with(
            "https://app.crewai.com/crewai_plus/deployments/128687"
        )

    def test_display_deployment_info_warns_when_browser_open_raises(self):
        self.mock_browser_open.side_effect = RuntimeError("no browser")

        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command._display_deployment_info(
                {"uuid": "test-uuid", "id": 128687, "status": "deployed"}
            )
            self.assertIn(
                "Could not open the deployment page automatically.",
                fake_out.getvalue(),
            )

        self.mock_browser_open.assert_called_once_with(
            "https://app.crewai.com/crewai_plus/deployments/128687"
        )

    def test_display_creation_success_resolves_deployment_page_id_from_status(self):
        status_response = MagicMock()
        status_response.is_success = True
        status_response.json.return_value = {
            "uuid": "new-uuid",
            "id": 128774,
            "status": "created",
        }
        self.mock_client.crew_status_by_uuid.return_value = status_response

        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command._display_creation_success(
                {"uuid": "new-uuid", "status": "created"}
            )
            output = fake_out.getvalue()

        self.assertIn("crewai deploy push --uuid new-uuid", output)
        self.assertIn(
            "https://app.crewai.com/crewai_plus/deployments/128774",
            output,
        )
        self.assertNotIn(
            "https://app.crewai.com/crewai_plus/deployments/new-uuid",
            output,
        )
        self.mock_client.crew_status_by_uuid.assert_called_once_with("new-uuid")
        self.mock_browser_open.assert_called_once_with(
            "https://app.crewai.com/crewai_plus/deployments/128774"
        )

    def test_open_deployment_page_does_not_open_uuid_url_when_status_lookup_fails(
        self,
    ):
        status_response = MagicMock()
        status_response.is_success = False
        self.mock_client.crew_status_by_uuid.return_value = status_response

        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command._open_deployment_page({"uuid": "new-uuid"})
            output = fake_out.getvalue()

        self.mock_client.crew_status_by_uuid.assert_called_once_with("new-uuid")
        self.mock_browser_open.assert_not_called()
        self.assertNotIn("deployments/new-uuid", output)

    def test_display_logs(self):
        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command._display_logs(
                [{"timestamp": "2023-01-01", "level": "INFO", "message": "Test log"}]
            )
            self.assertIn("2023-01-01 - INFO: Test log", fake_out.getvalue())

    @patch("crewai_cli.deploy.main.git.Repository")
    @patch("crewai_cli.deploy.main.DeployCommand._display_deployment_info")
    def test_deploy_with_uuid(self, mock_display, mock_repository):
        mock_repository.return_value.origin_url.return_value = (
            "https://github.com/test/repo.git"
        )
        mock_repository.return_value.create_initial_commit_if_needed.return_value = (
            False
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"uuid": "test-uuid"}
        self.mock_client.deploy_by_uuid.return_value = mock_response

        self.deploy_command.deploy(uuid="test-uuid", skip_validate=True)

        self.mock_client.deploy_by_uuid.assert_called_once_with("test-uuid")
        mock_display.assert_called_once_with({"uuid": "test-uuid"})

    @patch("crewai_cli.deploy.main.git.Repository")
    @patch("crewai_cli.deploy.main.DeployCommand._display_deployment_info")
    def test_deploy_with_project_name(self, mock_display, mock_repository):
        mock_repository.return_value.origin_url.return_value = (
            "https://github.com/test/repo.git"
        )
        mock_repository.return_value.create_initial_commit_if_needed.return_value = (
            False
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"uuid": "test-uuid"}
        self.mock_client.deploy_by_name.return_value = mock_response

        self.deploy_command.deploy(skip_validate=True)

        self.mock_client.deploy_by_name.assert_called_once_with("test_project")
        mock_display.assert_called_once_with({"uuid": "test-uuid"})

    @patch("crewai_cli.deploy.main.create_project_zip")
    @patch("crewai_cli.deploy.main.git.Repository")
    @patch("crewai_cli.deploy.main.DeployCommand._display_deployment_info")
    def test_deploy_with_remote_keeps_remote_path_when_fetch_fails(
        self, mock_display, mock_repository, mock_create_project_zip
    ):
        repository = mock_repository.return_value
        repository.origin_url.return_value = "https://github.com/test/repo.git"
        repository.fetch.side_effect = ValueError("fetch failed")
        repository.create_initial_commit_if_needed.return_value = False
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.json.return_value = {"uuid": "test-uuid"}
        self.mock_client.deploy_by_name.return_value = mock_response

        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command.deploy(skip_validate=True)
            output = fake_out.getvalue()

        mock_repository.assert_called_once_with(fetch=False)
        repository.fetch.assert_called_once_with()
        self.assertIn("Continuing with remote deployment", output)
        self.mock_client.deploy_by_name.assert_called_once_with("test_project")
        self.mock_client.update_crew_from_zip.assert_not_called()
        mock_create_project_zip.assert_not_called()
        mock_display.assert_called_once_with({"uuid": "test-uuid"})

    @patch("crewai_cli.deploy.main.create_project_zip")
    @patch("crewai_cli.deploy.main.git.Repository")
    @patch("crewai_cli.deploy.main.DeployCommand._display_deployment_info")
    def test_deploy_with_remote_keeps_remote_path_when_initial_commit_fails(
        self, mock_display, mock_repository, mock_create_project_zip
    ):
        repository = mock_repository.return_value
        repository.origin_url.return_value = "https://github.com/test/repo.git"
        repository.create_initial_commit_if_needed.side_effect = RuntimeError(
            "commit failed"
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.is_success = True
        mock_response.json.return_value = {"uuid": "test-uuid"}
        self.mock_client.deploy_by_name.return_value = mock_response

        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command.deploy(skip_validate=True)
            output = fake_out.getvalue()

        mock_repository.assert_called_once_with(fetch=False)
        repository.fetch.assert_called_once_with()
        self.assertIn("Continuing with remote deployment", output)
        self.mock_client.deploy_by_name.assert_called_once_with("test_project")
        self.mock_client.update_crew_from_zip.assert_not_called()
        mock_create_project_zip.assert_not_called()
        mock_display.assert_called_once_with({"uuid": "test-uuid"})

    @patch("crewai_cli.deploy.main.create_project_zip")
    @patch("crewai_cli.deploy.main.fetch_and_json_env_file")
    @patch("crewai_cli.deploy.main.git.Repository")
    @patch("crewai_cli.deploy.main.DeployCommand._display_deployment_info")
    def test_deploy_with_uuid_without_remote_updates_from_zip(
        self, mock_display, mock_repository, mock_fetch_env, mock_create_project_zip
    ):
        mock_fetch_env.return_value = {"ENV_VAR": "value"}
        mock_repository.return_value.origin_url.return_value = None
        mock_repository.return_value.create_initial_commit_if_needed.return_value = (
            False
        )
        mock_create_project_zip.return_value = Path("/tmp/test_project.zip")
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"uuid": "test-uuid"}
        self.mock_client.update_crew_from_zip.return_value = mock_response

        self.deploy_command.deploy(uuid="test-uuid", skip_validate=True)

        self.mock_client.update_crew_from_zip.assert_called_once_with(
            "test-uuid",
            Path("/tmp/test_project.zip"),
            env={"ENV_VAR": "value"},
        )
        self.mock_client.deploy_by_uuid.assert_not_called()
        mock_display.assert_called_once_with({"uuid": "test-uuid"})

    @patch("crewai_cli.deploy.main.create_project_zip")
    @patch("crewai_cli.deploy.main.fetch_and_json_env_file")
    @patch("crewai_cli.deploy.main.git.Repository")
    @patch("crewai_cli.deploy.main.DeployCommand._display_deployment_info")
    def test_deploy_with_project_name_without_remote_updates_from_zip(
        self, mock_display, mock_repository, mock_fetch_env, mock_create_project_zip
    ):
        mock_fetch_env.return_value = {"ENV_VAR": "value"}
        mock_repository.return_value.origin_url.return_value = None
        mock_repository.return_value.create_initial_commit_if_needed.return_value = (
            False
        )
        mock_create_project_zip.return_value = Path("/tmp/test_project.zip")
        status_response = MagicMock()
        status_response.status_code = 200
        status_response.is_success = True
        status_response.json.return_value = {"uuid": "test-uuid"}
        update_response = MagicMock()
        update_response.status_code = 200
        update_response.json.return_value = {"uuid": "test-uuid"}
        self.mock_client.crew_status_by_name.return_value = status_response
        self.mock_client.update_crew_from_zip.return_value = update_response

        self.deploy_command.deploy(skip_validate=True)

        self.mock_client.crew_status_by_name.assert_called_once_with("test_project")
        self.mock_client.update_crew_from_zip.assert_called_once_with(
            "test-uuid",
            Path("/tmp/test_project.zip"),
            env={"ENV_VAR": "value"},
        )
        self.mock_client.deploy_by_name.assert_not_called()
        mock_display.assert_called_once_with({"uuid": "test-uuid"})

    @patch("crewai_cli.deploy.main.fetch_and_json_env_file")
    @patch("crewai_cli.deploy.main.git.Repository")
    @patch("builtins.input")
    @pytest.mark.timeout(180)
    def test_create_crew(self, mock_input, mock_repository, mock_fetch_env):
        mock_fetch_env.return_value = {"ENV_VAR": "value"}
        mock_repository.return_value.origin_url.return_value = (
            "https://github.com/test/repo.git"
        )
        mock_repository.return_value.create_initial_commit_if_needed.return_value = (
            False
        )
        mock_input.return_value = ""

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.is_success = True
        mock_response.json.return_value = {"uuid": "new-uuid", "status": "created"}
        self.mock_client.create_crew.return_value = mock_response

        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command.create_crew(skip_validate=True)
            self.assertIn("Deployment created successfully!", fake_out.getvalue())
            self.assertIn("new-uuid", fake_out.getvalue())

    @patch("crewai_cli.deploy.main.create_project_zip")
    @patch("crewai_cli.deploy.main.fetch_and_json_env_file")
    @patch("crewai_cli.deploy.main.git.Repository")
    def test_create_crew_without_git_repo_initializes_and_uses_zip(
        self, mock_repository, mock_fetch_env, mock_create_project_zip
    ):
        mock_fetch_env.return_value = {"ENV_VAR": "value"}
        mock_repository.side_effect = ValueError("not a Git repository")
        initialized_repository = MagicMock()
        initialized_repository.origin_url.return_value = None
        mock_repository.initialize.return_value = initialized_repository
        mock_create_project_zip.return_value = Path("/tmp/test_project.zip")
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.is_success = True
        mock_response.json.return_value = {"uuid": "zip-uuid", "status": "created"}
        self.mock_client.create_crew_from_zip.return_value = mock_response

        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command.create_crew(confirm=True, skip_validate=True)
            output = fake_out.getvalue()

        self.assertIn("Initialized a local Git repository", output)
        self.assertIn("Deploying from a ZIP upload", output)
        mock_repository.initialize.assert_called_once_with()
        mock_create_project_zip.assert_called_once_with(
            "test_project", repository=initialized_repository
        )
        self.mock_client.create_crew_from_zip.assert_called_once_with(
            Path("/tmp/test_project.zip"),
            name="test_project",
            env={"ENV_VAR": "value"},
        )
        self.mock_client.create_crew.assert_not_called()

    @patch("crewai_cli.deploy.main.git.Repository")
    def test_prepare_git_repository_returns_repo_when_init_commit_fails(
        self, mock_repository
    ):
        recovered_repository = MagicMock()
        mock_repository.side_effect = [
            ValueError("not a Git repository"),
            recovered_repository,
        ]
        mock_repository.initialize.side_effect = RuntimeError("commit failed")

        with patch("sys.stdout", new=StringIO()) as fake_out:
            repository = self.deploy_command._prepare_git_repository()

        self.assertIs(repository, recovered_repository)
        self.assertIn("Git auto-setup did not complete", fake_out.getvalue())
        mock_repository.initialize.assert_called_once_with()
        self.assertEqual(mock_repository.call_count, 2)

    @patch("crewai_cli.deploy.main.create_project_zip")
    @patch("crewai_cli.deploy.main.fetch_and_json_env_file")
    @patch("crewai_cli.deploy.main.git.Repository")
    def test_create_crew_without_remote_uses_zip(
        self, mock_repository, mock_fetch_env, mock_create_project_zip
    ):
        mock_fetch_env.return_value = {"ENV_VAR": "value"}
        mock_repository.return_value.origin_url.return_value = None
        mock_repository.return_value.create_initial_commit_if_needed.return_value = (
            False
        )
        mock_create_project_zip.return_value = Path("/tmp/test_project.zip")
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.is_success = True
        mock_response.json.return_value = {"uuid": "zip-uuid", "status": "created"}
        self.mock_client.create_crew_from_zip.return_value = mock_response

        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command.create_crew(confirm=True, skip_validate=True)
            output = fake_out.getvalue()

        self.assertIn("No origin remote found.", output)
        self.assertIn("Deploying from a ZIP upload", output)
        mock_create_project_zip.assert_called_once_with(
            "test_project", repository=mock_repository.return_value
        )
        self.mock_client.create_crew_from_zip.assert_called_once_with(
            Path("/tmp/test_project.zip"),
            name="test_project",
            env={"ENV_VAR": "value"},
        )
        self.mock_client.create_crew.assert_not_called()

    @patch("crewai_cli.deploy.main.create_project_zip")
    @patch("crewai_cli.deploy.main.fetch_and_json_env_file")
    @patch("crewai_cli.deploy.main.git.Repository")
    def test_create_crew_without_remote_uses_git_file_list_when_commit_fails(
        self, mock_repository, mock_fetch_env, mock_create_project_zip
    ):
        mock_fetch_env.return_value = {"ENV_VAR": "value"}
        repository = mock_repository.return_value
        repository.origin_url.return_value = None
        repository.create_initial_commit_if_needed.side_effect = RuntimeError(
            "commit failed"
        )
        mock_create_project_zip.return_value = Path("/tmp/test_project.zip")
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.is_success = True
        mock_response.json.return_value = {"uuid": "zip-uuid", "status": "created"}
        self.mock_client.create_crew_from_zip.return_value = mock_response

        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command.create_crew(confirm=True, skip_validate=True)
            output = fake_out.getvalue()

        self.assertIn("Continuing with ZIP deployment using Git", output)
        self.assertIn("file listing", output)
        mock_create_project_zip.assert_called_once_with(
            "test_project", repository=repository
        )
        self.mock_client.create_crew_from_zip.assert_called_once_with(
            Path("/tmp/test_project.zip"),
            name="test_project",
            env={"ENV_VAR": "value"},
        )
        self.mock_client.create_crew.assert_not_called()

    def test_list_crews(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"name": "Crew1", "uuid": "uuid1", "status": "active"},
            {"name": "Crew2", "uuid": "uuid2", "status": "inactive"},
        ]
        self.mock_client.list_crews.return_value = mock_response

        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command.list_crews()
            self.assertIn("Crew1 (uuid1) active", fake_out.getvalue())
            self.assertIn("Crew2 (uuid2) inactive", fake_out.getvalue())

    def test_get_crew_status(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"name": "InternalCrew", "status": "active"}
        self.mock_client.crew_status_by_name.return_value = mock_response

        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command.get_crew_status()
            self.assertIn("InternalCrew", fake_out.getvalue())
            self.assertIn("active", fake_out.getvalue())

    def test_get_crew_logs(self):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            {"timestamp": "2023-01-01", "level": "INFO", "message": "Log1"},
            {"timestamp": "2023-01-02", "level": "ERROR", "message": "Log2"},
        ]
        self.mock_client.crew_by_name.return_value = mock_response

        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command.get_crew_logs(None)
            self.assertIn("2023-01-01 - INFO: Log1", fake_out.getvalue())
            self.assertIn("2023-01-02 - ERROR: Log2", fake_out.getvalue())

    def test_remove_crew(self):
        mock_response = MagicMock()
        mock_response.status_code = 204
        self.mock_client.delete_crew_by_name.return_value = mock_response

        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command.remove_crew(None)
            self.assertIn(
                "Crew 'test_project' removed successfully", fake_out.getvalue()
            )

    @unittest.skipIf(sys.version_info < (3, 11), "Requires Python 3.11+")
    def test_parse_toml_python_311_plus(self):
        toml_content = """
        [tool.poetry]
        name = "test_project"
        version = "0.1.0"

        [tool.poetry.dependencies]
        python = "^3.11"
        crewai = { extras = ["tools"], version = ">=0.51.0,<1.0.0" }
        """
        parsed = parse_toml(toml_content)
        self.assertEqual(parsed["tool"]["poetry"]["name"], "test_project")

    @patch(
        "builtins.open",
        new_callable=unittest.mock.mock_open,
        read_data="""
        [project]
        name = "test_project"
        version = "0.1.0"
        requires-python = ">=3.10,<3.14"
        dependencies = ["crewai"]
        """,
    )
    def test_get_project_name_python_310(self, mock_open):
        from crewai_cli.utils import get_project_name

        project_name = get_project_name()
        print("project_name", project_name)
        self.assertEqual(project_name, "test_project")

    @unittest.skipIf(sys.version_info < (3, 11), "Requires Python 3.11+")
    @patch(
        "builtins.open",
        new_callable=unittest.mock.mock_open,
        read_data="""
    [project]
    name = "test_project"
    version = "0.1.0"
    requires-python = ">=3.10,<3.14"
    dependencies = ["crewai"]
    """,
    )
    def test_get_project_name_python_311_plus(self, mock_open):
        from crewai_cli.utils import get_project_name

        project_name = get_project_name()
        self.assertEqual(project_name, "test_project")

    def test_get_crewai_version(self):
        from crewai_cli.version import get_crewai_version

        assert isinstance(get_crewai_version(), str)
