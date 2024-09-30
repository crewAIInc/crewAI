import unittest
from io import StringIO
from unittest.mock import MagicMock, patch
import sys

from crewai.cli.deploy.main import DeployCommand
from crewai.cli.utils import parse_toml


class TestDeployCommand(unittest.TestCase):
    @patch("crewai.cli.command.get_auth_token")
    @patch("crewai.cli.deploy.main.get_project_name")
    @patch("crewai.cli.command.PlusAPI")
    def setUp(self, mock_plus_api, mock_get_project_name, mock_get_auth_token):
        self.mock_get_auth_token = mock_get_auth_token
        self.mock_get_project_name = mock_get_project_name
        self.mock_plus_api = mock_plus_api

        self.mock_get_auth_token.return_value = "test_token"
        self.mock_get_project_name.return_value = "test_project"

        self.deploy_command = DeployCommand()
        self.mock_client = self.deploy_command.plus_api_client

    def test_init_success(self):
        self.assertEqual(self.deploy_command.project_name, "test_project")
        self.mock_plus_api.assert_called_once_with(api_key="test_token")

    @patch("crewai.cli.command.get_auth_token")
    def test_init_failure(self, mock_get_auth_token):
        mock_get_auth_token.side_effect = Exception("Auth failed")

        with self.assertRaises(SystemExit):
            DeployCommand()

    def test_handle_plus_api_error(self):
        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command._handle_plus_api_error(
                {"error": "Test error", "message": "Test message"}
            )
            self.assertIn("Error: Test error", fake_out.getvalue())
            self.assertIn("Message: Test message", fake_out.getvalue())

    def test_standard_no_param_error_message(self):
        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command._standard_no_param_error_message()
            self.assertIn("No UUID provided", fake_out.getvalue())

    def test_display_deployment_info(self):
        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command._display_deployment_info(
                {"uuid": "test-uuid", "status": "deployed"}
            )
            self.assertIn("Deploying the crew...", fake_out.getvalue())
            self.assertIn("test-uuid", fake_out.getvalue())
            self.assertIn("deployed", fake_out.getvalue())

    def test_display_logs(self):
        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command._display_logs(
                [{"timestamp": "2023-01-01", "level": "INFO", "message": "Test log"}]
            )
            self.assertIn("2023-01-01 - INFO: Test log", fake_out.getvalue())

    @patch("crewai.cli.deploy.main.DeployCommand._display_deployment_info")
    def test_deploy_with_uuid(self, mock_display):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"uuid": "test-uuid"}
        self.mock_client.deploy_by_uuid.return_value = mock_response

        self.deploy_command.deploy(uuid="test-uuid")

        self.mock_client.deploy_by_uuid.assert_called_once_with("test-uuid")
        mock_display.assert_called_once_with({"uuid": "test-uuid"})

    @patch("crewai.cli.deploy.main.DeployCommand._display_deployment_info")
    def test_deploy_with_project_name(self, mock_display):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"uuid": "test-uuid"}
        self.mock_client.deploy_by_name.return_value = mock_response

        self.deploy_command.deploy()

        self.mock_client.deploy_by_name.assert_called_once_with("test_project")
        mock_display.assert_called_once_with({"uuid": "test-uuid"})

    @patch("crewai.cli.deploy.main.fetch_and_json_env_file")
    @patch("crewai.cli.deploy.main.get_git_remote_url")
    @patch("builtins.input")
    def test_create_crew(self, mock_input, mock_get_git_remote_url, mock_fetch_env):
        mock_fetch_env.return_value = {"ENV_VAR": "value"}
        mock_get_git_remote_url.return_value = "https://github.com/test/repo.git"
        mock_input.return_value = ""

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"uuid": "new-uuid", "status": "created"}
        self.mock_client.create_crew.return_value = mock_response

        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command.create_crew()
            self.assertIn("Deployment created successfully!", fake_out.getvalue())
            self.assertIn("new-uuid", fake_out.getvalue())

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
        mock_response.json.return_value = {"name": "TestCrew", "status": "active"}
        self.mock_client.crew_status_by_name.return_value = mock_response

        with patch("sys.stdout", new=StringIO()) as fake_out:
            self.deploy_command.get_crew_status()
            self.assertIn("TestCrew", fake_out.getvalue())
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
    [tool.poetry]
    name = "test_project"
    version = "0.1.0"

    [tool.poetry.dependencies]
    python = "^3.10"
    crewai = { extras = ["tools"], version = ">=0.51.0,<1.0.0" }
    """,
    )
    def test_get_project_name_python_310(self, mock_open):
        from crewai.cli.utils import get_project_name

        project_name = get_project_name()
        self.assertEqual(project_name, "test_project")

    @unittest.skipIf(sys.version_info < (3, 11), "Requires Python 3.11+")
    @patch(
        "builtins.open",
        new_callable=unittest.mock.mock_open,
        read_data="""
    [tool.poetry]
    name = "test_project"
    version = "0.1.0"

    [tool.poetry.dependencies]
    python = "^3.11"
    crewai = { extras = ["tools"], version = ">=0.51.0,<1.0.0" }
    """,
    )
    def test_get_project_name_python_311_plus(self, mock_open):
        from crewai.cli.utils import get_project_name

        project_name = get_project_name()
        self.assertEqual(project_name, "test_project")

    def test_get_crewai_version(self):
        from crewai.cli.utils import get_crewai_version
        assert type(get_crewai_version()) is str
