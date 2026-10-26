import sys
import unittest
from io import StringIO
from pathlib import Path
import subprocess
from unittest.mock import MagicMock, Mock, patch

import pytest
import json

import crewai_cli.deploy.main as deploy_main
from crewai_cli.deploy.archive import ArchiveError
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

    def _status_response(
        self,
        *,
        uuid: str = "test-uuid",
        zip_deployment: bool | None = None,
        source_type: str | None = None,
        is_success: bool = True,
    ) -> MagicMock:
        response = MagicMock()
        response.is_success = is_success
        payload: dict = {"uuid": uuid}
        if zip_deployment is not None:
            payload["zip_deployment"] = zip_deployment
        if source_type is not None:
            payload["source_type"] = source_type
        response.json.return_value = payload
        return response

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
        self.mock_client.crew_status_by_uuid.return_value = self._status_response(
            zip_deployment=False
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"uuid": "test-uuid"}
        self.mock_client.deploy_by_uuid.return_value = mock_response

        self.deploy_command.deploy(uuid="test-uuid", skip_validate=True)

        self.mock_client.crew_status_by_uuid.assert_called_once_with("test-uuid")
        self.mock_client.deploy_by_uuid.assert_called_once_with("test-uuid")
        mock_display.assert_called_once_with({"uuid": "test-uuid"})

    @patch("crewai_cli.deploy.main.git.Repository")
    @patch("crewai_cli.deploy.main.DeployCommand._display_deployment_info")
    def test_a_push_amp_holds_back_for_a_running_comparison_says_why_and_force_goes_ahead(
        self, mock_display, mock_repository
    ):
        mock_repository.return_value.origin_url.return_value = "https://github.com/test/repo.git"
        mock_repository.return_value.create_initial_commit_if_needed.return_value = False
        self.mock_client.crew_status_by_uuid.return_value = self._status_response(zip_deployment=False)
        held = MagicMock()
        held.status_code = 409
        sentence = (
            "A models comparison is running on this deployment (started 05:03 UTC): redeploying now "
            "stops it, and the runs under way are lost. Wait for it to finish, or redeploy anyway from the "
            "deployment's page, or with `crewai deploy push --force` (a crewAI CLI newer than 1.15.28)."
        )
        # AMP's 409: the sentence in `error` too, for CLIs that print only that
        held.json.return_value = {
            "error": sentence,
            "code": "comparison_running",
            "message": sentence,
            "evaluation": {"id": "a78a325f-9e04-4962-94b3-1280003333e3", "started_at": "2026-10-14T05:03:00Z"},
        }
        self.mock_client.deploy_by_uuid.return_value = held

        with patch("sys.stdout", new=StringIO()) as out, self.assertRaises(SystemExit) as stopped:
            self.deploy_command.deploy(uuid="test-uuid", skip_validate=True)

        self.assertEqual(stopped.exception.code, 1)
        printed = " ".join(out.getvalue().split())
        self.assertIn("A models comparison is running on this deployment (started 05:03 UTC)", printed)
        self.assertIn("`crewai deploy push --force`", printed)
        self.assertIn("Nothing was deployed.", printed)
        self.assertNotIn("comparison_running", printed)
        self.assertNotIn("Request to Enterprise API failed", printed)
        mock_display.assert_not_called()

        ok = MagicMock()
        ok.status_code = 200
        ok.json.return_value = {"uuid": "test-uuid"}
        self.mock_client.deploy_by_uuid.reset_mock()
        self.mock_client.deploy_by_uuid.return_value = ok
        self.deploy_command.deploy(uuid="test-uuid", skip_validate=True, force=True)
        self.mock_client.deploy_by_uuid.assert_called_once_with("test-uuid", force=True)
        mock_display.assert_called_once_with({"uuid": "test-uuid"})

    @patch("crewai_cli.deploy.main.create_project_zip")
    @patch("crewai_cli.deploy.main.fetch_and_json_env_file")
    @patch("crewai_cli.deploy.main.git.Repository")
    @patch("crewai_cli.deploy.main.DeployCommand._display_deployment_info")
    def test_amps_refusal_is_recognised_on_every_route_in_either_shape(
        self, mock_display, mock_repository, mock_fetch_env, mock_create_project_zip
    ):
        """AMP names the refusal in `code`, its `error` carrying the sentence; an
        AMP from before named it in `error`. Either way, deploying by id, by name
        or through a ZIP update, the CLI says why and deploys nothing."""
        sentence = (
            "A models comparison is running on this deployment (started 05:03 UTC): "
            "redeploying now stops it, and the runs under way are lost."
        )
        shapes = {
            "named in code": {
                "error": sentence,
                "code": "comparison_running",
                "message": sentence,
            },
            "named in error": {"error": "comparison_running", "message": sentence},
        }
        # route: (deploy kwargs, status call, zip deployment, AMP call, origin)
        routes = {
            "by id": (
                {"uuid": "test-uuid"},
                "crew_status_by_uuid",
                False,
                "deploy_by_uuid",
                "https://github.com/test/repo.git",
            ),
            "by name": (
                {},
                "crew_status_by_name",
                False,
                "deploy_by_name",
                "https://github.com/test/repo.git",
            ),
            "zip update": (
                {"uuid": "test-uuid"},
                "crew_status_by_uuid",
                True,
                "update_crew_from_zip",
                None,
            ),
        }
        mock_fetch_env.return_value = {}
        mock_create_project_zip.return_value = Path("/tmp/test_project.zip")
        mock_repository.return_value.create_initial_commit_if_needed.return_value = (
            False
        )
        for route, (kwargs, status_call, zip_deployment, call, origin) in routes.items():
            for shape, body in shapes.items():
                with self.subTest(route=route, shape=shape):
                    self.mock_client.reset_mock()
                    mock_display.reset_mock()
                    mock_repository.return_value.origin_url.return_value = origin
                    getattr(
                        self.mock_client, status_call
                    ).return_value = self._status_response(
                        zip_deployment=zip_deployment
                    )
                    held = MagicMock()
                    held.status_code = 409
                    held.json.return_value = body
                    getattr(self.mock_client, call).return_value = held

                    with (
                        patch("sys.stdout", new=StringIO()) as out,
                        self.assertRaises(SystemExit) as stopped,
                    ):
                        self.deploy_command.deploy(skip_validate=True, **kwargs)

                    self.assertEqual(stopped.exception.code, 1)
                    printed = " ".join(out.getvalue().split())
                    self.assertIn(sentence, printed)
                    self.assertIn("Nothing was deployed.", printed)
                    self.assertNotIn("comparison_running", printed)
                    self.assertNotIn("Request to Enterprise API failed", printed)
                    getattr(self.mock_client, call).assert_called_once()
                    mock_display.assert_not_called()

    @patch("crewai_cli.deploy.main.git.Repository")
    @patch("crewai_cli.deploy.main.DeployCommand._display_deployment_info")
    def test_amps_hold_back_sentence_is_printed_literally_whatever_brackets_it_has(
        self, mock_display, mock_repository
    ):
        mock_repository.return_value.origin_url.return_value = "https://github.com/test/repo.git"
        mock_repository.return_value.create_initial_commit_if_needed.return_value = False
        self.mock_client.crew_status_by_uuid.return_value = self._status_response(zip_deployment=False)
        held = MagicMock()
        held.status_code = 409
        held.json.return_value = {
            "error": "A comparison [bold]running[/] on [crew] [/unmatched] — redeploy with --force.",
            "code": "comparison_running",
            "message": "A comparison [bold]running[/] on [crew] [/unmatched] — redeploy with --force.",
        }
        self.mock_client.deploy_by_uuid.return_value = held

        with patch("sys.stdout", new=StringIO()) as out, self.assertRaises(SystemExit):
            self.deploy_command.deploy(uuid="test-uuid", skip_validate=True)

        printed = " ".join(out.getvalue().split())
        self.assertIn("A comparison [bold]running[/] on [crew] [/unmatched] — redeploy with --force.", printed)
        self.assertIn("Nothing was deployed.", printed)

    @patch("crewai_cli.deploy.main.git.Repository")
    @patch("crewai_cli.deploy.main.DeployCommand._display_deployment_info")
    def test_deploy_with_project_name(self, mock_display, mock_repository):
        mock_repository.return_value.origin_url.return_value = (
            "https://github.com/test/repo.git"
        )
        mock_repository.return_value.create_initial_commit_if_needed.return_value = (
            False
        )
        self.mock_client.crew_status_by_name.return_value = self._status_response(
            zip_deployment=False
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"uuid": "test-uuid"}
        self.mock_client.deploy_by_name.return_value = mock_response

        self.deploy_command.deploy(skip_validate=True)

        self.mock_client.crew_status_by_name.assert_called_once_with("test_project")
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
        self.mock_client.crew_status_by_name.return_value = self._status_response(
            zip_deployment=False
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
        self.mock_client.crew_status_by_name.return_value = self._status_response(
            zip_deployment=False
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
        self.mock_client.crew_status_by_uuid.return_value = self._status_response(
            zip_deployment=True
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"uuid": "test-uuid"}
        self.mock_client.update_crew_from_zip.return_value = mock_response

        self.deploy_command.deploy(uuid="test-uuid", skip_validate=True)

        self.mock_client.crew_status_by_uuid.assert_called_once_with("test-uuid")
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
        update_response = MagicMock()
        update_response.status_code = 200
        update_response.json.return_value = {"uuid": "test-uuid"}
        self.mock_client.crew_status_by_name.return_value = self._status_response(
            zip_deployment=True
        )
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

    @patch("crewai_cli.deploy.main.create_project_zip")
    @patch("crewai_cli.deploy.main.fetch_and_json_env_file")
    @patch("crewai_cli.deploy.main.git.Repository")
    @patch("crewai_cli.deploy.main.DeployCommand._display_deployment_info")
    def test_deploy_zip_amp_source_uploads_even_when_origin_exists(
        self, mock_display, mock_repository, mock_fetch_env, mock_create_project_zip
    ):
        mock_fetch_env.return_value = {"ENV_VAR": "value"}
        mock_repository.return_value.origin_url.return_value = (
            "https://github.com/test/repo.git"
        )
        mock_repository.return_value.create_initial_commit_if_needed.return_value = (
            False
        )
        mock_create_project_zip.return_value = Path("/tmp/test_project.zip")
        self.mock_client.crew_status_by_uuid.return_value = self._status_response(
            zip_deployment=True
        )
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
    def test_a_zip_source_type_uploads_the_code_even_when_origin_exists(
        self, mock_display, mock_repository, mock_fetch_env, mock_create_project_zip
    ):
        # What AMP's status actually sends: `source_type`, no `zip_deployment`.
        # A redeploy by uuid here rebuilds the ZIP AMP already holds, not the code.
        mock_fetch_env.return_value = {"ENV_VAR": "value"}
        mock_repository.return_value.origin_url.return_value = (
            "https://github.com/test/repo.git"
        )
        mock_repository.return_value.create_initial_commit_if_needed.return_value = (
            False
        )
        mock_create_project_zip.return_value = Path("/tmp/test_project.zip")
        for source_type in ("zip", "studio"):
            with self.subTest(source_type=source_type):
                self.mock_client.reset_mock()
                self.mock_client.crew_status_by_uuid.return_value = (
                    self._status_response(source_type=source_type)
                )
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

    @patch("crewai_cli.deploy.main.create_project_zip")
    @patch("crewai_cli.deploy.main.git.Repository")
    @patch("crewai_cli.deploy.main.DeployCommand._display_deployment_info")
    def test_a_git_source_type_redeploys_from_git_even_without_origin(
        self, mock_display, mock_repository, mock_create_project_zip
    ):
        mock_repository.return_value.origin_url.return_value = None
        mock_repository.return_value.create_initial_commit_if_needed.return_value = (
            False
        )
        for source_type in ("github", "any_git"):
            with self.subTest(source_type=source_type):
                self.mock_client.reset_mock()
                self.mock_client.crew_status_by_uuid.return_value = (
                    self._status_response(source_type=source_type)
                )
                mock_response = MagicMock()
                mock_response.status_code = 200
                mock_response.json.return_value = {"uuid": "test-uuid"}
                self.mock_client.deploy_by_uuid.return_value = mock_response

                self.deploy_command.deploy(uuid="test-uuid", skip_validate=True)

                self.mock_client.deploy_by_uuid.assert_called_once_with("test-uuid")
                self.mock_client.update_crew_from_zip.assert_not_called()
        mock_create_project_zip.assert_not_called()

    def test_zip_deployment_is_read_before_source_type(self):
        from crewai_cli.deploy.main import _zip_deployment_flag

        self.assertFalse(_zip_deployment_flag({"zip_deployment": False, "source_type": "zip"}))
        self.assertTrue(_zip_deployment_flag({"source_type": "zip"}))
        self.assertIsNone(_zip_deployment_flag({"source_type": "something-new"}))
        self.assertIsNone(_zip_deployment_flag({"uuid": "x"}))

    @patch("crewai_cli.deploy.main.git.Repository")
    @patch("crewai_cli.deploy.main.DeployCommand._display_deployment_info")
    def test_deploy_falls_back_to_origin_when_zip_deployment_is_missing(
        self, mock_display, mock_repository
    ):
        mock_repository.return_value.origin_url.return_value = (
            "https://github.com/test/repo.git"
        )
        mock_repository.return_value.create_initial_commit_if_needed.return_value = (
            False
        )
        self.mock_client.crew_status_by_uuid.return_value = self._status_response()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"uuid": "test-uuid"}
        self.mock_client.deploy_by_uuid.return_value = mock_response

        self.deploy_command.deploy(uuid="test-uuid", skip_validate=True)

        self.mock_client.deploy_by_uuid.assert_called_once_with("test-uuid")
        self.mock_client.update_crew_from_zip.assert_not_called()
        mock_display.assert_called_once_with({"uuid": "test-uuid"})

    @patch("crewai_cli.deploy.main.git.Repository")
    @patch("crewai_cli.deploy.main.DeployCommand._display_deployment_info")
    def test_deploy_falls_back_to_origin_when_status_request_fails(
        self, mock_display, mock_repository
    ):
        mock_repository.return_value.origin_url.return_value = (
            "https://github.com/test/repo.git"
        )
        mock_repository.return_value.create_initial_commit_if_needed.return_value = (
            False
        )
        self.mock_client.crew_status_by_uuid.return_value = self._status_response(
            zip_deployment=True, is_success=False
        )
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"uuid": "test-uuid"}
        self.mock_client.deploy_by_uuid.return_value = mock_response

        self.deploy_command.deploy(uuid="test-uuid", skip_validate=True)

        self.mock_client.deploy_by_uuid.assert_called_once_with("test-uuid")
        self.mock_client.update_crew_from_zip.assert_not_called()
        mock_display.assert_called_once_with({"uuid": "test-uuid"})

    @patch("crewai_cli.deploy.main.create_project_zip")
    @patch("crewai_cli.deploy.main.fetch_and_json_env_file")
    @patch("crewai_cli.deploy.main.git.Repository")
    @patch("crewai_cli.deploy.main.DeployCommand._display_deployment_info")
    def test_deploy_falls_back_to_zip_when_status_fails_and_origin_is_missing(
        self, mock_display, mock_repository, mock_fetch_env, mock_create_project_zip
    ):
        mock_fetch_env.return_value = {"ENV_VAR": "value"}
        mock_repository.return_value.origin_url.return_value = None
        mock_repository.return_value.create_initial_commit_if_needed.return_value = (
            False
        )
        mock_create_project_zip.return_value = Path("/tmp/test_project.zip")
        self.mock_client.crew_status_by_uuid.side_effect = httpx.ConnectError(
            "offline"
        )
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

    @patch("crewai_cli.deploy.main.fetch_and_json_env_file")
    @patch("crewai_cli.deploy.main.git.Repository")
    @patch("builtins.input")
    @pytest.mark.timeout(180)
    def test_create_crew_reports_the_created_uuid_from_the_git_path(
        self, mock_input, mock_repository, mock_fetch_env
    ):
        """The attempt span cannot carry the uuid; the post-success span must."""
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

        with patch.object(self.deploy_command, "_telemetry") as telemetry:
            with patch("sys.stdout", new=StringIO()):
                self.deploy_command.create_crew(skip_validate=True)

        telemetry.crew_deployment_created_span.assert_called_once_with(
            uuid="new-uuid", source="cli"
        )
        telemetry.create_crew_deployment_span.assert_called_once_with(source="cli")

    @patch("crewai_cli.deploy.main.create_project_zip")
    @patch("crewai_cli.deploy.main.fetch_and_json_env_file")
    @patch("crewai_cli.deploy.main.git.Repository")
    def test_create_crew_reports_the_created_uuid_from_the_zip_path(
        self, mock_repository, mock_fetch_env, mock_create_project_zip
    ):
        """The two creation paths converge, so the uuid must arrive from both."""
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

        with patch.object(self.deploy_command, "_telemetry") as telemetry:
            with patch("sys.stdout", new=StringIO()):
                self.deploy_command.create_crew(skip_validate=True, confirm=True)

        telemetry.crew_deployment_created_span.assert_called_once_with(
            uuid="zip-uuid", source="cli"
        )

    @patch("crewai_cli.deploy.main.fetch_and_json_env_file")
    @patch("crewai_cli.deploy.main.git.Repository")
    @patch("builtins.input")
    def test_a_failed_create_counts_the_attempt_but_reports_no_creation(
        self, mock_input, mock_repository, mock_fetch_env
    ):
        """The whole point of two spans: a failed create is an attempt, not a success.

        Collapsing them into one post-success span would silently convert the
        creation-attempt metric into a creation-success metric, which the churn
        figures are built on.
        """
        mock_fetch_env.return_value = {"ENV_VAR": "value"}
        mock_repository.return_value.origin_url.return_value = (
            "https://github.com/test/repo.git"
        )
        mock_repository.return_value.create_initial_commit_if_needed.return_value = (
            False
        )
        mock_input.return_value = ""

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.is_success = False
        mock_response.json.return_value = {"error": "boom"}
        self.mock_client.create_crew.return_value = mock_response

        with patch.object(self.deploy_command, "_telemetry") as telemetry:
            with patch("sys.stdout", new=StringIO()):
                with self.assertRaises(SystemExit):
                    self.deploy_command.create_crew(skip_validate=True)

        telemetry.create_crew_deployment_span.assert_called_once_with(source="cli")
        telemetry.crew_deployment_created_span.assert_not_called()

    # --- why a create failed (the Crew Deployment Failed span) ---------------

    def _git_path(self, mock_input, mock_repository, mock_fetch_env):
        mock_fetch_env.return_value = {"ENV_VAR": "value"}
        mock_repository.return_value.origin_url.return_value = (
            "https://github.com/test/repo.git"
        )
        mock_repository.return_value.create_initial_commit_if_needed.return_value = (
            False
        )
        mock_input.return_value = ""

    @staticmethod
    def _api_response(status_code: int, body: object) -> MagicMock:
        response = MagicMock()
        response.status_code = status_code
        response.is_success = 200 <= status_code < 300
        if isinstance(body, Exception):
            response.json.side_effect = body
        else:
            response.json.return_value = body
        return response

    @patch("crewai_cli.deploy.main.fetch_and_json_env_file")
    @patch("crewai_cli.deploy.main.git.Repository")
    @patch("builtins.input")
    def test_a_rejected_create_reports_the_api_class_and_status(
        self, mock_input, mock_repository, mock_fetch_env
    ):
        """A 4xx names the class and carries the exact code; the message never leaves."""
        self._git_path(mock_input, mock_repository, mock_fetch_env)
        self.mock_client.create_crew.return_value = self._api_response(
            422, {"name": ["has already been taken"]}
        )

        with patch.object(self.deploy_command, "_telemetry") as telemetry:
            with patch("sys.stdout", new=StringIO()):
                with self.assertRaises(SystemExit):
                    self.deploy_command.create_crew(skip_validate=True)

        telemetry.crew_deployment_failed_span.assert_called_once_with(
            "api_4xx", source="cli", status_code=422
        )
        telemetry.crew_deployment_created_span.assert_not_called()

    @patch("crewai_cli.deploy.main.fetch_and_json_env_file")
    @patch("crewai_cli.deploy.main.git.Repository")
    @patch("builtins.input")
    def test_a_server_error_reports_api_5xx(
        self, mock_input, mock_repository, mock_fetch_env
    ):
        self._git_path(mock_input, mock_repository, mock_fetch_env)
        self.mock_client.create_crew.return_value = self._api_response(
            503, {"error": "upstream unavailable"}
        )

        with patch.object(self.deploy_command, "_telemetry") as telemetry:
            with patch("sys.stdout", new=StringIO()):
                with self.assertRaises(SystemExit):
                    self.deploy_command.create_crew(skip_validate=True)

        telemetry.crew_deployment_failed_span.assert_called_once_with(
            "api_5xx", source="cli", status_code=503
        )

    @patch("crewai_cli.deploy.main.fetch_and_json_env_file")
    @patch("crewai_cli.deploy.main.git.Repository")
    @patch("builtins.input")
    def test_a_non_json_success_body_reports_invalid_json(
        self, mock_input, mock_repository, mock_fetch_env
    ):
        """A proxy's 200 HTML page: _validate_response rejects it, so it is a
        failure with a 2xx attached, and it is not the same failure as a JSON
        body that is missing the creation fields."""
        self._git_path(mock_input, mock_repository, mock_fetch_env)
        self.mock_client.create_crew.return_value = self._api_response(
            200, ValueError("not json")
        )

        with patch.object(self.deploy_command, "_telemetry") as telemetry:
            with patch("sys.stdout", new=StringIO()):
                with self.assertRaises(SystemExit):
                    self.deploy_command.create_crew(skip_validate=True)

        telemetry.crew_deployment_failed_span.assert_called_once_with(
            "invalid_json", source="cli", status_code=200
        )

    @patch("crewai_cli.deploy.main.fetch_and_json_env_file")
    @patch("crewai_cli.deploy.main.git.Repository")
    @patch("builtins.input")
    def test_a_transport_failure_reports_network_error_and_propagates(
        self, mock_input, mock_repository, mock_fetch_env
    ):
        """No response, so no status; the exception reaches the caller unchanged."""
        self._git_path(mock_input, mock_repository, mock_fetch_env)
        self.mock_client.create_crew.side_effect = httpx.ConnectError("refused")

        with patch.object(self.deploy_command, "_telemetry") as telemetry:
            with patch("sys.stdout", new=StringIO()):
                with pytest.raises(httpx.ConnectError, match="refused"):
                    self.deploy_command.create_crew(skip_validate=True)

        telemetry.crew_deployment_failed_span.assert_called_once_with(
            "network_error", source="cli"
        )
        telemetry.crew_deployment_created_span.assert_not_called()

    @patch("crewai_cli.deploy.main.create_project_zip")
    @patch("crewai_cli.deploy.main.fetch_and_json_env_file")
    @patch("crewai_cli.deploy.main.git.Repository")
    def test_a_failed_archive_reports_zip_error(
        self, mock_repository, mock_fetch_env, mock_create_project_zip
    ):
        mock_fetch_env.return_value = {"ENV_VAR": "value"}
        mock_repository.side_effect = ValueError("not a Git repository")
        initialized_repository = MagicMock()
        initialized_repository.origin_url.return_value = None
        mock_repository.initialize.return_value = initialized_repository
        mock_create_project_zip.side_effect = ArchiveError(
            "No deployable project files were found."
        )

        with patch.object(self.deploy_command, "_telemetry") as telemetry:
            with patch("sys.stdout", new=StringIO()):
                with pytest.raises(ArchiveError, match="No deployable project files"):
                    self.deploy_command.create_crew(skip_validate=True, confirm=True)

        telemetry.crew_deployment_failed_span.assert_called_once_with(
            "zip_error", source="cli"
        )
        self.mock_client.create_crew_from_zip.assert_not_called()

    @patch("crewai_cli.deploy.main.fetch_and_json_env_file")
    @patch("crewai_cli.deploy.main.git.Repository")
    @patch("builtins.input")
    def test_a_git_helper_error_is_not_a_zip_error(
        self, mock_input, mock_repository, mock_fetch_env
    ):
        """The git helpers raise plain ValueError; only ArchiveError is a ZIP problem."""
        self._git_path(mock_input, mock_repository, mock_fetch_env)
        mock_repository.return_value.origin_url.side_effect = ValueError(
            "Git remote lookup failed"
        )

        with patch.object(self.deploy_command, "_telemetry") as telemetry:
            with patch("sys.stdout", new=StringIO()):
                with pytest.raises(ValueError, match="Git remote lookup failed"):
                    self.deploy_command.create_crew(skip_validate=True)

        telemetry.crew_deployment_failed_span.assert_called_once_with(
            "unexpected", source="cli"
        )

    @patch("crewai_cli.deploy.main.fetch_and_json_env_file")
    @patch("crewai_cli.deploy.main.git.Repository")
    @patch("builtins.input")
    def test_a_non_json_error_page_keeps_its_api_class(
        self, mock_input, mock_repository, mock_fetch_env
    ):
        """A gateway's HTML 502 is an api_5xx, not an invalid response."""
        self._git_path(mock_input, mock_repository, mock_fetch_env)
        self.mock_client.create_crew.return_value = self._api_response(
            502, ValueError("not json")
        )

        with patch.object(self.deploy_command, "_telemetry") as telemetry:
            with patch("sys.stdout", new=StringIO()):
                with self.assertRaises(SystemExit):
                    self.deploy_command.create_crew(skip_validate=True)

        telemetry.crew_deployment_failed_span.assert_called_once_with(
            "api_5xx", source="cli", status_code=502
        )

    @patch("crewai_cli.deploy.main.fetch_and_json_env_file")
    @patch("crewai_cli.deploy.main.git.Repository")
    @patch("builtins.input")
    def test_a_success_body_that_is_not_a_creation_reports_invalid_creation_response(
        self, mock_input, mock_repository, mock_fetch_env
    ):
        """A 2xx JSON body without a deployment uuid is a broken creation
        contract, not a success, and must not crash."""
        self._git_path(mock_input, mock_repository, mock_fetch_env)
        for body in ([{"uuid": "in-a-list"}], {"status": "created"}, {}):
            with self.subTest(body=body):
                self.mock_client.create_crew.return_value = self._api_response(
                    200, body
                )
                with patch.object(self.deploy_command, "_telemetry") as telemetry:
                    with patch("sys.stdout", new=StringIO()) as fake_out:
                        with self.assertRaises(SystemExit) as exit_info:
                            self.deploy_command.create_crew(skip_validate=True)

                assert exit_info.exception.code == 1
                assert "no deployment uuid was returned" in fake_out.getvalue()
                telemetry.crew_deployment_failed_span.assert_called_once_with(
                    "invalid_creation_response", source="cli", status_code=200
                )
                telemetry.crew_deployment_created_span.assert_not_called()

    @patch("crewai_cli.deploy.main.fetch_and_json_env_file")
    @patch("crewai_cli.deploy.main.git.Repository")
    @patch("builtins.input")
    def test_an_abort_at_the_prompt_reports_user_declined(
        self, mock_input, mock_repository, mock_fetch_env
    ):
        self._git_path(mock_input, mock_repository, mock_fetch_env)
        mock_input.side_effect = KeyboardInterrupt

        with patch.object(self.deploy_command, "_telemetry") as telemetry:
            with patch("sys.stdout", new=StringIO()):
                with self.assertRaises(KeyboardInterrupt):
                    self.deploy_command.create_crew(skip_validate=True)

        telemetry.crew_deployment_failed_span.assert_called_once_with(
            "user_declined", source="cli"
        )
        self.mock_client.create_crew.assert_not_called()

    @patch("crewai_cli.deploy.main.fetch_and_json_env_file")
    @patch("crewai_cli.deploy.main.git.Repository")
    @patch("builtins.input")
    def test_a_failure_from_the_run_tui_keeps_its_source(
        self, mock_input, mock_repository, mock_fetch_env
    ):
        """The TUI succeeds far less often than the CLI; the split must survive."""
        self._git_path(mock_input, mock_repository, mock_fetch_env)
        self.mock_client.create_crew.return_value = self._api_response(
            403, {"error": "forbidden"}
        )

        with patch.object(self.deploy_command, "_telemetry") as telemetry:
            with patch("sys.stdout", new=StringIO()):
                with self.assertRaises(SystemExit):
                    self.deploy_command.create_crew(
                        skip_validate=True, confirm=True, source="tui"
                    )

        telemetry.crew_deployment_failed_span.assert_called_once_with(
            "api_4xx", source="tui", status_code=403
        )

    @patch("crewai_cli.deploy.main.fetch_and_json_env_file")
    @patch("crewai_cli.deploy.main.git.Repository")
    @patch("builtins.input")
    def test_a_successful_create_reports_no_failure(
        self, mock_input, mock_repository, mock_fetch_env
    ):
        self._git_path(mock_input, mock_repository, mock_fetch_env)
        self.mock_client.create_crew.return_value = self._api_response(
            201, {"uuid": "new-uuid", "status": "created"}
        )

        with patch.object(self.deploy_command, "_telemetry") as telemetry:
            with patch("sys.stdout", new=StringIO()):
                self.deploy_command.create_crew(skip_validate=True)

        telemetry.crew_deployment_failed_span.assert_not_called()
        telemetry.crew_deployment_created_span.assert_called_once_with(
            uuid="new-uuid", source="cli"
        )

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

    def test_remove_by_uuid_names_the_removed_deployment_not_the_local_project(self):
        self.mock_client.delete_crew_by_uuid.return_value = httpx.Response(204)

        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command.remove_crew("other-uuid")

        output = fake_out.getvalue()
        self.assertIn("Crew 'other-uuid' removed successfully", output)
        self.assertNotIn("test_project", output)

    def test_failed_remove_by_uuid_shows_the_api_error(self):
        self.mock_client.delete_crew_by_uuid.return_value = httpx.Response(
            403, json={"error": "You do not have permission to delete this crew"}
        )

        with patch("sys.stdout", new=StringIO()) as fake_out:
            with self.assertRaises(SystemExit):
                self.deploy_command.remove_crew("other-uuid")

        output = fake_out.getvalue()
        self.assertIn("Failed to remove crew 'other-uuid'", output)
        self.assertIn("You do not have permission to delete this crew", output)
        self.assertNotIn("test_project", output)

    def test_failed_remove_by_name_shows_the_api_error(self):
        self.mock_client.delete_crew_by_name.return_value = httpx.Response(
            404, json={"error": "Crew not found"}
        )

        with patch("sys.stdout", new=StringIO()) as fake_out:
            with self.assertRaises(SystemExit):
                self.deploy_command.remove_crew(None)

        output = fake_out.getvalue()
        self.assertIn("Failed to remove crew 'test_project'", output)
        self.assertIn("Crew not found", output)

    def test_status_of_a_failed_deployment_shows_amp_error_and_fix(self):
        # The shape AMP's status endpoint returns for a failed git clone.
        self.mock_client.crew_status_by_uuid.return_value = httpx.Response(
            200,
            json={
                "name": "lead_qualifier",
                "status": "Provisioning Failed, try again.",
                "state": "Failed",
                "error_message": (
                    "Host key verification failed. [git@github.com:org/repo.git]"
                ),
                "error_category": "git_clone_failure",
                "error_suggested_fix": "Use an HTTPS [origin] URL.",
            },
        )

        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command.get_crew_status("test-uuid")

        output = fake_out.getvalue()
        self.assertIn("Provisioning Failed, try again.", output)
        # Printed verbatim: square brackets are not swallowed as Rich markup.
        self.assertRegex(
            output,
            r"Error:\s+Host key verification failed\. \[git@github\.com:org/repo\.git\]",
        )
        self.assertRegex(output, r"Fix:\s+Use an HTTPS \[origin\] URL\.")

    def test_status_of_a_healthy_deployment_prints_no_error_lines(self):
        # AMP sends the error fields as null outside a failure.
        self.mock_client.crew_status_by_uuid.return_value = httpx.Response(
            200,
            json={
                "name": "lead_qualifier",
                "status": "Online",
                "state": "Successful",
                "error_message": None,
                "error_category": None,
                "error_suggested_fix": None,
            },
        )

        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command.get_crew_status("test-uuid")

        output = fake_out.getvalue()
        self.assertRegex(output, r"Status:\s+Online")
        self.assertNotIn("Error:", output)
        self.assertNotIn("Fix:", output)

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


PROJECT_ID = "1f0c6a3e-8d0b-4c4e-9a52-6a9f6d1d2b7e"


class TestDeploySendsTheProjectId(unittest.TestCase):
    """Every create and push carries `[tool.crewai].project_id` when the
    project has one, so AMP knows which project a deployment runs."""

    @patch("crewai_cli.command.get_auth_token", return_value="test_token")
    @patch("crewai_cli.deploy.main.get_project_id", return_value=PROJECT_ID)
    @patch("crewai_cli.deploy.main.get_project_name", return_value="test_project")
    @patch("crewai_cli.command.PlusAPI")
    def setUp(self, _plus_api, _name, _project_id, _token):
        self.deploy_command = deploy_main.DeployCommand()
        self.mock_client = self.deploy_command.plus_api_client
        self.mock_client.base_url = "https://app.crewai.com"
        browser = patch("crewai_cli.deploy.main.webbrowser.open")
        browser.start()
        self.addCleanup(browser.stop)

    def _created(self, uuid: str = "new-uuid") -> MagicMock:
        response = MagicMock()
        response.status_code = 201
        response.is_success = True
        response.json.return_value = {"uuid": uuid, "status": "created"}
        return response

    def _status(self, zip_deployment: bool) -> MagicMock:
        response = MagicMock()
        response.is_success = True
        response.json.return_value = {"uuid": "test-uuid", "zip_deployment": zip_deployment}
        return response

    def test_the_command_reads_the_project_id(self):
        self.assertEqual(self.deploy_command.project_id, PROJECT_ID)

    @patch("crewai_cli.deploy.main.fetch_and_json_env_file", return_value={})
    @patch("crewai_cli.deploy.main.git.Repository")
    def test_create_from_git_sends_it(self, mock_repository, _env):
        mock_repository.return_value.origin_url.return_value = "https://github.com/t/r.git"
        mock_repository.return_value.create_initial_commit_if_needed.return_value = False
        self.mock_client.create_crew.return_value = self._created()

        with patch("sys.stdout", new=StringIO()):
            self.deploy_command.create_crew(confirm=True, skip_validate=True)

        payload = self.mock_client.create_crew.call_args.args[0]
        # Once, at the top level, where AMP reads it on every deploy route.
        self.assertEqual(payload["project_id"], PROJECT_ID)
        self.assertNotIn("project_id", payload["deploy"])

    @patch("crewai_cli.deploy.main.create_project_zip", return_value=Path("/tmp/p.zip"))
    @patch("crewai_cli.deploy.main.fetch_and_json_env_file", return_value={})
    @patch("crewai_cli.deploy.main.git.Repository")
    def test_create_from_a_zip_sends_it(self, mock_repository, _env, _zip):
        mock_repository.return_value.origin_url.return_value = None
        mock_repository.return_value.create_initial_commit_if_needed.return_value = False
        self.mock_client.create_crew_from_zip.return_value = self._created()

        with patch("sys.stdout", new=StringIO()):
            self.deploy_command.create_crew(confirm=True, skip_validate=True)

        self.mock_client.create_crew_from_zip.assert_called_once_with(
            Path("/tmp/p.zip"), name="test_project", env={}, project_id=PROJECT_ID
        )

    @patch("crewai_cli.deploy.main.git.Repository")
    @patch("crewai_cli.deploy.main.DeployCommand._display_deployment_info")
    def test_a_push_from_git_sends_it(self, _display, mock_repository):
        mock_repository.return_value.origin_url.return_value = "https://github.com/t/r.git"
        mock_repository.return_value.create_initial_commit_if_needed.return_value = False
        self.mock_client.crew_status_by_uuid.return_value = self._status(False)
        self.mock_client.crew_status_by_name.return_value = self._status(False)

        self.deploy_command.deploy(uuid="test-uuid", skip_validate=True)
        self.deploy_command.deploy(skip_validate=True)

        self.mock_client.deploy_by_uuid.assert_called_once_with(
            "test-uuid", project_id=PROJECT_ID
        )
        self.mock_client.deploy_by_name.assert_called_once_with(
            "test_project", project_id=PROJECT_ID
        )

    @patch("crewai_cli.deploy.main.create_project_zip", return_value=Path("/tmp/p.zip"))
    @patch("crewai_cli.deploy.main.fetch_and_json_env_file", return_value={})
    @patch("crewai_cli.deploy.main.git.Repository")
    @patch("crewai_cli.deploy.main.DeployCommand._display_deployment_info")
    def test_a_push_from_a_zip_sends_it(self, _display, mock_repository, _env, _zip):
        mock_repository.return_value.origin_url.return_value = None
        mock_repository.return_value.create_initial_commit_if_needed.return_value = False
        self.mock_client.crew_status_by_uuid.return_value = self._status(True)

        self.deploy_command.deploy(uuid="test-uuid", skip_validate=True)

        self.mock_client.update_crew_from_zip.assert_called_once_with(
            "test-uuid", Path("/tmp/p.zip"), env={}, project_id=PROJECT_ID
        )


def test_a_project_without_an_id_sends_none(monkeypatch, tmp_path: Path):
    """Read, never minted: a deploy does not rewrite pyproject.toml."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n")
    monkeypatch.setattr("crewai_cli.command.get_auth_token", lambda: "t")
    monkeypatch.setattr("crewai_cli.command.PlusAPI", MagicMock())
    monkeypatch.setattr(deploy_main, "get_project_name", lambda require=False: "demo")

    command = deploy_main.DeployCommand()

    assert command.project_id is None and command._project() == {}
    assert "project_id" not in (tmp_path / "pyproject.toml").read_text()


def test_the_project_id_is_read_from_pyproject(monkeypatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        f"[project]\nname = 'demo'\n\n[tool.crewai]\nproject_id = \"{PROJECT_ID}\"\n"
    )
    monkeypatch.setattr("crewai_cli.command.get_auth_token", lambda: "t")
    monkeypatch.setattr("crewai_cli.command.PlusAPI", MagicMock())
    monkeypatch.setattr(deploy_main, "get_project_name", lambda require=False: "demo")

    assert deploy_main.DeployCommand()._project() == {"project_id": PROJECT_ID}


# `crewai deploy test`: one run of the deployment through AMP, answered locally.

KICKOFF_ID = "5f0b1c2d-3e4f-4a5b-8c6d-7e8f9a0b1c2d"
_RealHttpxClient = httpx.Client
AmpAnswer = httpx.Response | httpx.HTTPError


class _FakeAmp:
    """AMP's kickoff and status routes. The kickoff answers once; the status
    answers come in order, and the last one repeats."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.kickoff: AmpAnswer = httpx.Response(200, json={"kickoff_id": KICKOFF_ID})
        self.statuses: list[AmpAnswer] = [_status("SUCCESS", result="done")]

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "POST":
            answer = self.kickoff
        elif len(self.statuses) > 1:
            answer = self.statuses.pop(0)
        else:
            answer = self.statuses[0]
        if isinstance(answer, httpx.HTTPError):
            raise answer
        return answer

    @property
    def status_requests(self) -> list[httpx.Request]:
        return [request for request in self.requests if request.method == "GET"]


class _FakeClock:
    """Time that moves only when the command sleeps."""

    def __init__(self) -> None:
        self.now = 0.0

    def sleep(self, seconds: float) -> None:
        self.now += seconds

    def monotonic(self) -> float:
        return self.now


def _status(state: str, **fields: object) -> httpx.Response:
    return httpx.Response(200, json={"state": state, **fields})


def _printed(capsys: pytest.CaptureFixture[str]) -> str:
    """What the command printed, with Rich's line wrapping undone."""
    return " ".join(capsys.readouterr().out.split())


@pytest.fixture
def amp(monkeypatch, tmp_path: Path):
    """A logged-in project named `my crew` whose PlusAPI talks to a _FakeAmp."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pyproject.toml").write_text(
        "[project]\nname = 'my crew'\ndependencies = ['crewai']\n"
    )
    monkeypatch.setattr("crewai_cli.command.get_auth_token", lambda: "t")
    monkeypatch.setattr(deploy_main, "time", _FakeClock())
    fake = _FakeAmp()
    transport = httpx.MockTransport(fake)
    # A machine or OS proxy would mount a real transport ahead of the mock one.
    monkeypatch.setattr("crewai_core.plus_api._os_proxies", dict)
    with (
        patch.dict("crewai_core.plus_api._MACHINE_TRANSPORT_ENV", clear=True),
        patch(
            "crewai_core.plus_api.httpx.Client",
            side_effect=lambda **kwargs: _RealHttpxClient(transport=transport, **kwargs),
        ),
    ):
        yield fake


def test_deploy_test_runs_the_deployment_by_uuid_and_prints_its_result_as_written(
    amp: _FakeAmp, capsys
):
    amp.statuses = [
        _status("PENDING", status="Task is pending"),
        _status("RUNNING", status="Task is Running"),
        _status("SUCCESS", result="[bold]The report[/] is done"),
    ]

    deploy_main.DeployCommand().test_deployment("dep-uuid", {"topic": "AI"}, 600)

    kickoff, *polls = amp.requests
    assert kickoff.method == "POST"
    assert kickoff.url.path == "/crewai_plus/api/v1/crew/dep-uuid/kickoff"
    assert json.loads(kickoff.content) == {"inputs": {"topic": "AI"}}
    # AMP gives a sleeping deployment 60 s to wake before it answers.
    assert kickoff.extensions["timeout"]["read"] == 90.0
    # A hung status read must not carry the command far past --timeout.
    assert {poll.extensions["timeout"]["read"] for poll in polls} == {30.0}
    assert [poll.url.path for poll in polls] == [
        f"/crewai_plus/api/v1/crew/dep-uuid/tasks/{KICKOFF_ID}/status"
    ] * 3
    printed = _printed(capsys)
    assert f"Kickoff ID: {KICKOFF_ID}" in printed
    assert f"Run {KICKOFF_ID} succeeded." in printed
    assert "[bold]The report[/] is done" in printed


def test_deploy_test_without_a_uuid_runs_the_deployment_named_after_the_project(
    amp: _FakeAmp,
):
    deploy_main.DeployCommand().test_deployment(None, {}, 600)

    assert [request.url.raw_path for request in amp.requests] == [
        b"/crewai_plus/api/v1/crew/by-name/my%20crew/kickoff",
        f"/crewai_plus/api/v1/crew/by-name/my%20crew/tasks/{KICKOFF_ID}/status".encode(),
    ]
    assert json.loads(amp.requests[0].content) == {"inputs": {}}


@pytest.mark.parametrize(
    ("result", "line"),
    [
        (
            '{"summary": ":warning: ' + "x" * 120 + '", "items": [1, 2, 3]}',
            '{"summary": ":warning: ' + "x" * 120 + '", "items": [1, 2, 3]}',
        ),
        (
            {"summary": "y" * 100, "count": 2},
            json.dumps({"summary": "y" * 100, "count": 2}),
        ),
    ],
    ids=["long-json-string", "json-object"],
)
def test_deploy_test_prints_the_result_unwrapped_on_its_own_line(
    amp: _FakeAmp, capsys, result: object, line: str
):
    amp.statuses = [_status("SUCCESS", result=result)]

    deploy_main.DeployCommand().test_deployment("dep-uuid", {}, 600)

    assert line in capsys.readouterr().out.splitlines()


def test_deploy_test_says_when_the_run_returned_nothing(amp: _FakeAmp, capsys):
    amp.statuses = [_status("SUCCESS", result=None)]

    deploy_main.DeployCommand().test_deployment("dep-uuid", {}, 600)

    assert "It returned no result." in _printed(capsys)


@pytest.mark.parametrize(
    ("status", "message"),
    [
        (
            _status("FAILED", status="Boom: [red]the LLM said no[/red]"),
            f"Run {KICKOFF_ID} ended in state FAILED: Boom: [red]the LLM said no[/red]",
        ),
        (
            _status("PAUSED", status="Awaiting human feedback"),
            f"Run {KICKOFF_ID} is paused, waiting for human feedback. Answer it in AMP",
        ),
        (
            _status("REVOKED", status="Execution was stopped"),
            f"Run {KICKOFF_ID} ended in state REVOKED: Execution was stopped",
        ),
        (
            _status("NOT FOUND", status="Task not found"),
            f"Run {KICKOFF_ID} ended in state NOT FOUND: Task not found",
        ),
        (
            _status("SOMETHING NEW"),
            f"Run {KICKOFF_ID} ended in state SOMETHING NEW: no details",
        ),
        (
            httpx.Response(200, text="<html>ok</html>"),
            "AMP answered the run's status with something unexpected: "
            "<html>ok</html>",
        ),
    ],
    ids=["failed", "paused", "revoked", "not-found", "unknown-state", "not-json"],
)
def test_deploy_test_exits_1_on_every_outcome_but_success(
    amp: _FakeAmp, capsys, status: httpx.Response, message: str
):
    amp.statuses = [_status("RUNNING"), status]

    with pytest.raises(SystemExit) as exc_info:
        deploy_main.DeployCommand().test_deployment("dep-uuid", {}, 600)

    assert exc_info.value.code == 1
    assert message in _printed(capsys)
    assert len(amp.status_requests) == 2


@pytest.mark.parametrize(
    ("answer", "message"),
    [
        (
            httpx.Response(
                422,
                json={
                    "error": "deployment_is_not_ready",
                    "message": "The deployment is not ready. (Status='deploying')",
                },
            ),
            "AMP did not start the run (422): The deployment is not ready. "
            "(Status='deploying')",
        ),
        (
            httpx.Response(
                403,
                json={
                    "error": "forbidden",
                    "message": "You don't have permission to run this automation.",
                },
            ),
            "AMP did not start the run (403): You don't have permission to run "
            "this automation.",
        ),
        (
            httpx.Response(
                429,
                json={
                    "error": "execution_limit_reached",
                    "message": "Your organization reached its execution limit.",
                },
            ),
            "AMP did not start the run (429): Your organization reached its "
            "execution limit.",
        ),
        (
            httpx.Response(404, json={"error": "deployment_not_found"}),
            "AMP did not start the run (404): deployment_not_found",
        ),
        (
            httpx.Response(502, text="<html>Bad Gateway</html>"),
            "AMP did not start the run (502): <html>Bad Gateway</html>",
        ),
        (
            httpx.Response(200, json={"task_id": "x"}),
            'AMP answered without a kickoff_id: {"task_id":"x"}',
        ),
        (
            httpx.ReadTimeout("timed out"),
            "The request to start the run failed (ReadTimeout('timed out')). It "
            "may have started anyway",
        ),
    ],
    ids=[
        "not-ready",
        "no-permission",
        "execution-limit",
        "not-found",
        "bad-gateway",
        "no-kickoff-id",
        "network-timeout",
    ],
)
def test_deploy_test_exits_1_with_amps_reason_when_the_run_does_not_start(
    amp: _FakeAmp, capsys, answer: AmpAnswer, message: str
):
    amp.kickoff = answer

    with pytest.raises(SystemExit) as exc_info:
        deploy_main.DeployCommand().test_deployment("dep-uuid", {}, 600)

    assert exc_info.value.code == 1
    assert message in _printed(capsys)
    assert amp.status_requests == []


def test_deploy_test_rides_out_a_few_failed_status_reads(amp: _FakeAmp, capsys):
    amp.statuses = [
        httpx.Response(503, text="Service Unavailable"),
        httpx.ConnectError("connection refused"),
        _status("RUNNING"),
        httpx.Response(500, json={"error": "internal_server_error"}),
        httpx.ReadTimeout("timed out"),
        _status("SUCCESS", result="done"),
    ]

    deploy_main.DeployCommand().test_deployment("dep-uuid", {}, 600)

    assert f"Run {KICKOFF_ID} succeeded." in _printed(capsys)


def test_deploy_test_exits_1_after_three_failed_status_reads_in_a_row(
    amp: _FakeAmp, capsys
):
    amp.statuses = [
        httpx.Response(502, text="Bad Gateway"),
        httpx.ConnectError("connection refused"),
        httpx.Response(500, json={"message": "Something went wrong"}),
    ]

    with pytest.raises(SystemExit) as exc_info:
        deploy_main.DeployCommand().test_deployment("dep-uuid", {}, 600)

    assert exc_info.value.code == 1
    assert (
        f"Could not read run {KICKOFF_ID}'s status 3 times in a row "
        "(last: 500: Something went wrong). It may still be running in AMP."
    ) in _printed(capsys)


def test_deploy_test_exits_1_at_once_when_amp_refuses_the_status(
    amp: _FakeAmp, capsys
):
    amp.statuses = [httpx.Response(404, json={"message": "Deployment not found"})]

    with pytest.raises(SystemExit) as exc_info:
        deploy_main.DeployCommand().test_deployment("dep-uuid", {}, 600)

    assert exc_info.value.code == 1
    assert "AMP refused the run's status (404: Deployment not found)" in _printed(
        capsys
    )
    assert len(amp.status_requests) == 1


def test_deploy_test_gives_up_after_its_timeout_and_names_the_run(
    amp: _FakeAmp, capsys
):
    amp.statuses = [_status("RUNNING")]

    with pytest.raises(SystemExit) as exc_info:
        deploy_main.DeployCommand().test_deployment("dep-uuid", {}, 10)

    assert exc_info.value.code == 1
    assert (
        f"Run {KICKOFF_ID} did not finish within 10 seconds. It keeps running in AMP."
    ) in _printed(capsys)
    # One read every 3 s on the fake clock: at 3, 6, 9 and 12 s.
    assert len(amp.status_requests) == 4


def test_a_run_request_must_name_its_deployment():
    from crewai_cli.plus_api import PlusAPI

    with pytest.raises(ValueError, match="Pass the deployment's uuid or name"):
        PlusAPI("t").kickoff_crew({})
