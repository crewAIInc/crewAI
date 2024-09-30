import unittest
import unittest.mock
from crewai.cli.tools.main import ToolCommand
from io import StringIO
from unittest.mock import patch, MagicMock


class TestToolCommand(unittest.TestCase):
    @patch("crewai.cli.tools.main.subprocess.run")
    @patch("crewai.cli.plus_api.PlusAPI.get_tool")
    def test_install_success(self, mock_get, mock_subprocess_run):
        mock_get_response = MagicMock()
        mock_get_response.status_code = 200
        mock_get_response.json.return_value = {
            "handle": "sample-tool",
            "repository": {"handle": "sample-repo", "url": "https://example.com/repo"},
        }
        mock_get.return_value = mock_get_response
        mock_subprocess_run.return_value = MagicMock(stderr=None)

        tool_command = ToolCommand()

        with patch("sys.stdout", new=StringIO()) as fake_out:
            tool_command.install("sample-tool")
            output = fake_out.getvalue()

        mock_get.assert_called_once_with("sample-tool")
        mock_subprocess_run.assert_any_call(
            ["poetry", "add", "--source", "crewai-sample-repo", "sample-tool"],
            capture_output=False,
            text=True,
            check=True,
        )

        self.assertIn("Succesfully installed sample-tool", output)

    @patch("crewai.cli.plus_api.PlusAPI.get_tool")
    def test_install_tool_not_found(self, mock_get):
        mock_get_response = MagicMock()
        mock_get_response.status_code = 404
        mock_get.return_value = mock_get_response

        tool_command = ToolCommand()

        with patch("sys.stdout", new=StringIO()) as fake_out:
            with self.assertRaises(SystemExit):
                tool_command.install("non-existent-tool")
            output = fake_out.getvalue()

        mock_get.assert_called_once_with("non-existent-tool")
        self.assertIn("No tool found with this name", output)

    @patch("crewai.cli.plus_api.PlusAPI.get_tool")
    def test_install_api_error(self, mock_get):
        mock_get_response = MagicMock()
        mock_get_response.status_code = 500
        mock_get.return_value = mock_get_response

        tool_command = ToolCommand()

        with patch("sys.stdout", new=StringIO()) as fake_out:
            with self.assertRaises(SystemExit):
                tool_command.install("error-tool")
            output = fake_out.getvalue()

        mock_get.assert_called_once_with("error-tool")
        self.assertIn("Failed to get tool details", output)

    @patch("crewai.cli.tools.main.get_project_name", return_value="sample-tool")
    @patch("crewai.cli.tools.main.get_project_version", return_value="1.0.0")
    @patch(
        "crewai.cli.tools.main.get_project_description", return_value="A sample tool"
    )
    @patch("crewai.cli.tools.main.subprocess.run")
    @patch(
        "crewai.cli.tools.main.os.listdir", return_value=["sample-tool-1.0.0.tar.gz"]
    )
    @patch(
        "crewai.cli.tools.main.open",
        new_callable=unittest.mock.mock_open,
        read_data=b"sample tarball content",
    )
    @patch("crewai.cli.plus_api.PlusAPI.publish_tool")
    def test_publish_success(
        self,
        mock_publish,
        mock_open,
        mock_listdir,
        mock_subprocess_run,
        mock_get_project_description,
        mock_get_project_version,
        mock_get_project_name,
    ):
        mock_publish_response = MagicMock()
        mock_publish_response.status_code = 200
        mock_publish_response.json.return_value = {"handle": "sample-tool"}
        mock_publish.return_value = mock_publish_response

        tool_command = ToolCommand()
        tool_command.publish(is_public=True)

        mock_get_project_name.assert_called_once_with(require=True)
        mock_get_project_version.assert_called_once_with(require=True)
        mock_get_project_description.assert_called_once_with(require=False)
        mock_subprocess_run.assert_called_once_with(
            ["poetry", "build", "-f", "sdist", "--output", unittest.mock.ANY],
            check=True,
            capture_output=False,
        )
        mock_open.assert_called_once_with(unittest.mock.ANY, "rb")
        mock_publish.assert_called_once_with(
            handle="sample-tool",
            is_public=True,
            version="1.0.0",
            description="A sample tool",
            encoded_file=unittest.mock.ANY,
        )

    @patch("crewai.cli.tools.main.get_project_name", return_value="sample-tool")
    @patch("crewai.cli.tools.main.get_project_version", return_value="1.0.0")
    @patch(
        "crewai.cli.tools.main.get_project_description", return_value="A sample tool"
    )
    @patch("crewai.cli.tools.main.subprocess.run")
    @patch(
        "crewai.cli.tools.main.os.listdir", return_value=["sample-tool-1.0.0.tar.gz"]
    )
    @patch(
        "crewai.cli.tools.main.open",
        new_callable=unittest.mock.mock_open,
        read_data=b"sample tarball content",
    )
    @patch("crewai.cli.plus_api.PlusAPI.publish_tool")
    def test_publish_failure(
        self,
        mock_publish,
        mock_open,
        mock_listdir,
        mock_subprocess_run,
        mock_get_project_description,
        mock_get_project_version,
        mock_get_project_name,
    ):
        mock_publish_response = MagicMock()
        mock_publish_response.status_code = 422
        mock_publish_response.json.return_value = {"name": ["is already taken"]}
        mock_publish.return_value = mock_publish_response

        tool_command = ToolCommand()

        with patch("sys.stdout", new=StringIO()) as fake_out:
            with self.assertRaises(SystemExit):
                tool_command.publish(is_public=True)
            output = fake_out.getvalue()

        mock_publish.assert_called_once()
        self.assertIn("Failed to publish tool", output)
        self.assertIn("Name is already taken", output)

    @patch("crewai.cli.tools.main.get_project_name", return_value="sample-tool")
    @patch("crewai.cli.tools.main.get_project_version", return_value="1.0.0")
    @patch(
        "crewai.cli.tools.main.get_project_description", return_value="A sample tool"
    )
    @patch("crewai.cli.tools.main.subprocess.run")
    @patch(
        "crewai.cli.tools.main.os.listdir", return_value=["sample-tool-1.0.0.tar.gz"]
    )
    @patch(
        "crewai.cli.tools.main.open",
        new_callable=unittest.mock.mock_open,
        read_data=b"sample tarball content",
    )
    @patch("crewai.cli.plus_api.PlusAPI.publish_tool")
    def test_publish_api_error(
        self,
        mock_publish,
        mock_open,
        mock_listdir,
        mock_subprocess_run,
        mock_get_project_description,
        mock_get_project_version,
        mock_get_project_name,
    ):
        mock_publish_response = MagicMock()
        mock_publish_response.status_code = 500
        mock_publish.return_value = mock_publish_response

        tool_command = ToolCommand()

        with patch("sys.stdout", new=StringIO()) as fake_out:
            with self.assertRaises(SystemExit):
                tool_command.publish(is_public=True)
            output = fake_out.getvalue()

        mock_publish.assert_called_once()
        self.assertIn("Failed to publish tool", output)

    @patch("crewai.cli.plus_api.PlusAPI.login_to_tool_repository")
    @patch("crewai.cli.tools.main.subprocess.run")
    def test_login_success(self, mock_subprocess_run, mock_login):
        mock_login_response = MagicMock()
        mock_login_response.status_code = 200
        mock_login_response.json.return_value = {
            "repositories": [
                {
                    "handle": "tools",
                    "url": "https://example.com/repo",
                }
            ],
            "credential": {
                "username": "user",
                "password": "pass"
            },
        }
        mock_login.return_value = mock_login_response

        mock_subprocess_run.return_value = MagicMock(stderr=None)

        tool_command = ToolCommand()

        with patch("sys.stdout", new=StringIO()) as fake_out:
            tool_command.login()
            output = fake_out.getvalue()

        mock_login.assert_called_once()
        mock_subprocess_run.assert_any_call(
            [
                "poetry",
                "source",
                "add",
                "--priority=explicit",
                "crewai-tools",
                "https://example.com/repo",
            ],
            text=True,
            check=True
        )
        mock_subprocess_run.assert_any_call(
            [
                "poetry",
                "config",
                "http-basic.crewai-tools",
                "user",
                "pass",
            ],
            capture_output=False,
            text=True,
            check=True,
        )
        self.assertIn("Succesfully authenticated to the tool repository", output)
