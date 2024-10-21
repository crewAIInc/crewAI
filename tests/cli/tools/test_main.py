import os
import tempfile
import unittest
import unittest.mock
from contextlib import contextmanager
from io import StringIO
from unittest import mock
from unittest.mock import MagicMock, patch

from pytest import raises

from crewai.cli.tools.main import ToolCommand


@contextmanager
def in_temp_dir():
    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        try:
            yield temp_dir
        finally:
            os.chdir(original_dir)


@patch("crewai.cli.tools.main.subprocess.run")
def test_create_success(mock_subprocess):
    with in_temp_dir():
        tool_command = ToolCommand()

        with patch.object(tool_command, "login") as mock_login, patch(
            "sys.stdout", new=StringIO()
        ) as fake_out:
            tool_command.create("test-tool")
            output = fake_out.getvalue()

        assert os.path.isdir("test_tool")
        assert os.path.isfile(os.path.join("test_tool", "README.md"))
        assert os.path.isfile(os.path.join("test_tool", "pyproject.toml"))
        assert os.path.isfile(
            os.path.join("test_tool", "src", "test_tool", "__init__.py")
        )
        assert os.path.isfile(os.path.join("test_tool", "src", "test_tool", "tool.py"))

        with open(os.path.join("test_tool", "src", "test_tool", "tool.py"), "r") as f:
            content = f.read()
            assert "class TestTool" in content

        mock_login.assert_called_once()
        mock_subprocess.assert_called_once_with(["git", "init"], check=True)

        assert "Creating custom tool test_tool..." in output


@patch("crewai.cli.tools.main.subprocess.run")
@patch("crewai.cli.plus_api.PlusAPI.get_tool")
def test_install_success(mock_get, mock_subprocess_run):
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

    mock_get.assert_has_calls([mock.call("sample-tool"), mock.call().json()])
    mock_subprocess_run.assert_any_call(
        [
            "uv",
            "add",
            "--index",
            "sample-repo=https://example.com/repo",
            "sample-tool",
        ],
        capture_output=False,
        text=True,
        check=True,
    )

    assert "Succesfully installed sample-tool" in output


@patch("crewai.cli.plus_api.PlusAPI.get_tool")
def test_install_tool_not_found(mock_get):
    mock_get_response = MagicMock()
    mock_get_response.status_code = 404
    mock_get.return_value = mock_get_response

    tool_command = ToolCommand()

    with patch("sys.stdout", new=StringIO()) as fake_out:
        try:
            tool_command.install("non-existent-tool")
        except SystemExit:
            pass
        output = fake_out.getvalue()

    mock_get.assert_called_once_with("non-existent-tool")
    assert "No tool found with this name" in output


@patch("crewai.cli.plus_api.PlusAPI.get_tool")
def test_install_api_error(mock_get):
    mock_get_response = MagicMock()
    mock_get_response.status_code = 500
    mock_get.return_value = mock_get_response

    tool_command = ToolCommand()

    with patch("sys.stdout", new=StringIO()) as fake_out:
        try:
            tool_command.install("error-tool")
        except SystemExit:
            pass
        output = fake_out.getvalue()

    mock_get.assert_called_once_with("error-tool")
    assert "Failed to get tool details" in output


@patch("crewai.cli.tools.main.git.Repository.is_synced", return_value=False)
def test_publish_when_not_in_sync(mock_is_synced):
    with patch("sys.stdout", new=StringIO()) as fake_out, raises(SystemExit):
        tool_command = ToolCommand()
        tool_command.publish(is_public=True)

    assert "Local changes need to be resolved before publishing" in fake_out.getvalue()


@patch("crewai.cli.tools.main.get_project_name", return_value="sample-tool")
@patch("crewai.cli.tools.main.get_project_version", return_value="1.0.0")
@patch("crewai.cli.tools.main.get_project_description", return_value="A sample tool")
@patch("crewai.cli.tools.main.subprocess.run")
@patch("crewai.cli.tools.main.os.listdir", return_value=["sample-tool-1.0.0.tar.gz"])
@patch(
    "crewai.cli.tools.main.open",
    new_callable=unittest.mock.mock_open,
    read_data=b"sample tarball content",
)
@patch("crewai.cli.plus_api.PlusAPI.publish_tool")
@patch("crewai.cli.tools.main.git.Repository.is_synced", return_value=False)
def test_publish_when_not_in_sync_and_force(
    mock_is_synced,
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
    tool_command.publish(is_public=True, force=True)

    mock_get_project_name.assert_called_with(require=True)
    mock_get_project_version.assert_called_with(require=True)
    mock_get_project_description.assert_called_with(require=False)
    mock_subprocess_run.assert_called_with(
        ["uv", "build", "--sdist", "--out-dir", unittest.mock.ANY],
        check=True,
        capture_output=False,
    )
    mock_open.assert_called_with(unittest.mock.ANY, "rb")
    mock_publish.assert_called_with(
        handle="sample-tool",
        is_public=True,
        version="1.0.0",
        description="A sample tool",
        encoded_file=unittest.mock.ANY,
    )


@patch("crewai.cli.tools.main.get_project_name", return_value="sample-tool")
@patch("crewai.cli.tools.main.get_project_version", return_value="1.0.0")
@patch("crewai.cli.tools.main.get_project_description", return_value="A sample tool")
@patch("crewai.cli.tools.main.subprocess.run")
@patch("crewai.cli.tools.main.os.listdir", return_value=["sample-tool-1.0.0.tar.gz"])
@patch(
    "crewai.cli.tools.main.open",
    new_callable=unittest.mock.mock_open,
    read_data=b"sample tarball content",
)
@patch("crewai.cli.plus_api.PlusAPI.publish_tool")
@patch("crewai.cli.tools.main.git.Repository.is_synced", return_value=True)
def test_publish_success(
    mock_is_synced,
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

    mock_get_project_name.assert_called_with(require=True)
    mock_get_project_version.assert_called_with(require=True)
    mock_get_project_description.assert_called_with(require=False)
    mock_subprocess_run.assert_called_with(
        ["uv", "build", "--sdist", "--out-dir", unittest.mock.ANY],
        check=True,
        capture_output=False,
    )
    mock_open.assert_called_with(unittest.mock.ANY, "rb")
    mock_publish.assert_called_with(
        handle="sample-tool",
        is_public=True,
        version="1.0.0",
        description="A sample tool",
        encoded_file=unittest.mock.ANY,
    )


@patch("crewai.cli.tools.main.get_project_name", return_value="sample-tool")
@patch("crewai.cli.tools.main.get_project_version", return_value="1.0.0")
@patch("crewai.cli.tools.main.get_project_description", return_value="A sample tool")
@patch("crewai.cli.tools.main.subprocess.run")
@patch("crewai.cli.tools.main.os.listdir", return_value=["sample-tool-1.0.0.tar.gz"])
@patch(
    "crewai.cli.tools.main.open",
    new_callable=unittest.mock.mock_open,
    read_data=b"sample tarball content",
)
@patch("crewai.cli.plus_api.PlusAPI.publish_tool")
def test_publish_failure(
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
        try:
            tool_command.publish(is_public=True)
        except SystemExit:
            pass
        output = fake_out.getvalue()

    mock_publish.assert_called_once()
    assert "Failed to complete operation" in output
    assert "Name is already taken" in output


@patch("crewai.cli.tools.main.get_project_name", return_value="sample-tool")
@patch("crewai.cli.tools.main.get_project_version", return_value="1.0.0")
@patch("crewai.cli.tools.main.get_project_description", return_value="A sample tool")
@patch("crewai.cli.tools.main.subprocess.run")
@patch("crewai.cli.tools.main.os.listdir", return_value=["sample-tool-1.0.0.tar.gz"])
@patch(
    "crewai.cli.tools.main.open",
    new_callable=unittest.mock.mock_open,
    read_data=b"sample tarball content",
)
@patch("crewai.cli.plus_api.PlusAPI.publish_tool")
def test_publish_api_error(
    mock_publish,
    mock_open,
    mock_listdir,
    mock_subprocess_run,
    mock_get_project_description,
    mock_get_project_version,
    mock_get_project_name,
):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.json.return_value = {"error": "Internal Server Error"}
    mock_response.ok = False
    mock_publish.return_value = mock_response

    tool_command = ToolCommand()

    with patch("sys.stdout", new=StringIO()) as fake_out:
        try:
            tool_command.publish(is_public=True)
        except SystemExit:
            pass
        output = fake_out.getvalue()

    mock_publish.assert_called_once()
    assert "Request to Enterprise API failed" in output
