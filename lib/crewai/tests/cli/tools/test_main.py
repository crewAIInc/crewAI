import os
import tempfile
import unittest
import unittest.mock
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch

import pytest
from crewai.cli.shared.token_manager import TokenManager
from crewai.cli.tools.main import ToolCommand
from pytest import raises


@contextmanager
def in_temp_dir():
    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        try:
            yield temp_dir
        finally:
            os.chdir(original_dir)


@pytest.fixture
def tool_command():
    # Create a temporary directory for each test to avoid token storage conflicts
    with tempfile.TemporaryDirectory() as temp_dir:
        # Mock the secure storage path to use the temp directory
        with patch.object(
            TokenManager, "_get_secure_storage_path", return_value=Path(temp_dir)
        ):
            TokenManager().save_tokens(
                "test-token", (datetime.now() + timedelta(seconds=36000)).timestamp()
            )
            tool_command = ToolCommand()
            with patch.object(tool_command, "login"):
                yield tool_command


@patch("crewai.cli.tools.main.subprocess.run")
def test_create_success(mock_subprocess, capsys, tool_command):
    with in_temp_dir():
        tool_command.create("test-tool")
        output = capsys.readouterr().out
        assert "Creating custom tool test_tool..." in output

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

        mock_subprocess.assert_called_once_with(["git", "init"], check=True)


@patch("crewai.cli.tools.main.subprocess.run")
@patch("crewai.cli.plus_api.PlusAPI.get_tool")
@patch("crewai.cli.tools.main.ToolCommand._print_current_organization")
def test_install_success(
    mock_print_org, mock_get, mock_subprocess_run, capsys, tool_command
):
    mock_get_response = MagicMock()
    mock_get_response.status_code = 200
    mock_get_response.json.return_value = {
        "handle": "sample-tool",
        "repository": {"handle": "sample-repo", "url": "https://example.com/repo"},
    }
    mock_get.return_value = mock_get_response
    mock_subprocess_run.return_value = MagicMock(stderr=None)

    tool_command.install("sample-tool")
    output = capsys.readouterr().out
    assert "Successfully installed sample-tool" in output

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
        env=unittest.mock.ANY,
    )

    # Verify _print_current_organization was called
    mock_print_org.assert_called_once()


@patch("crewai.cli.tools.main.subprocess.run")
@patch("crewai.cli.plus_api.PlusAPI.get_tool")
def test_install_success_from_pypi(mock_get, mock_subprocess_run, capsys, tool_command):
    mock_get_response = MagicMock()
    mock_get_response.status_code = 200
    mock_get_response.json.return_value = {
        "handle": "sample-tool",
        "repository": {"handle": "sample-repo", "url": "https://example.com/repo"},
        "source": "pypi",
    }
    mock_get.return_value = mock_get_response
    mock_subprocess_run.return_value = MagicMock(stderr=None)

    tool_command.install("sample-tool")
    output = capsys.readouterr().out
    assert "Successfully installed sample-tool" in output

    mock_get.assert_has_calls([mock.call("sample-tool"), mock.call().json()])
    mock_subprocess_run.assert_any_call(
        [
            "uv",
            "add",
            "sample-tool",
        ],
        capture_output=False,
        text=True,
        check=True,
        env=unittest.mock.ANY,
    )


@patch("crewai.cli.plus_api.PlusAPI.get_tool")
def test_install_tool_not_found(mock_get, capsys, tool_command):
    mock_get_response = MagicMock()
    mock_get_response.status_code = 404
    mock_get.return_value = mock_get_response

    with raises(SystemExit):
        tool_command.install("non-existent-tool")
    output = capsys.readouterr().out
    assert "No tool found with this name" in output

    mock_get.assert_called_once_with("non-existent-tool")


@patch("crewai.cli.plus_api.PlusAPI.get_tool")
def test_install_api_error(mock_get, capsys, tool_command):
    mock_get_response = MagicMock()
    mock_get_response.status_code = 500
    mock_get.return_value = mock_get_response

    with raises(SystemExit):
        tool_command.install("error-tool")
    output = capsys.readouterr().out
    assert "Failed to get tool details" in output

    mock_get.assert_called_once_with("error-tool")


@patch("crewai.cli.tools.main.git.Repository.is_synced", return_value=False)
def test_publish_when_not_in_sync(mock_is_synced, capsys, tool_command):
    with raises(SystemExit):
        tool_command.publish(is_public=True)

    output = capsys.readouterr().out
    assert "Local changes need to be resolved before publishing" in output


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
@patch(
    "crewai.cli.tools.main.extract_available_exports",
    return_value=[{"name": "SampleTool"}],
)
@patch("crewai.cli.tools.main.ToolCommand._print_current_organization")
def test_publish_when_not_in_sync_and_force(
    mock_print_org,
    mock_available_exports,
    mock_is_synced,
    mock_publish,
    mock_open,
    mock_listdir,
    mock_subprocess_run,
    mock_get_project_description,
    mock_get_project_version,
    mock_get_project_name,
    tool_command,
):
    mock_publish_response = MagicMock()
    mock_publish_response.status_code = 200
    mock_publish_response.json.return_value = {"handle": "sample-tool"}
    mock_publish.return_value = mock_publish_response

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
        available_exports=[{"name": "SampleTool"}],
    )
    mock_print_org.assert_called_once()


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
@patch(
    "crewai.cli.tools.main.extract_available_exports",
    return_value=[{"name": "SampleTool"}],
)
def test_publish_success(
    mock_available_exports,
    mock_is_synced,
    mock_publish,
    mock_open,
    mock_listdir,
    mock_subprocess_run,
    mock_get_project_description,
    mock_get_project_version,
    mock_get_project_name,
    tool_command,
):
    mock_publish_response = MagicMock()
    mock_publish_response.status_code = 200
    mock_publish_response.json.return_value = {"handle": "sample-tool"}
    mock_publish.return_value = mock_publish_response

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
        available_exports=[{"name": "SampleTool"}],
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
@patch(
    "crewai.cli.tools.main.extract_available_exports",
    return_value=[{"name": "SampleTool"}],
)
def test_publish_failure(
    mock_available_exports,
    mock_publish,
    mock_open,
    mock_listdir,
    mock_subprocess_run,
    mock_get_project_description,
    mock_get_project_version,
    mock_get_project_name,
    capsys,
    tool_command,
):
    mock_publish_response = MagicMock()
    mock_publish_response.status_code = 422
    mock_publish_response.json.return_value = {"name": ["is already taken"]}
    mock_publish.return_value = mock_publish_response

    with raises(SystemExit):
        tool_command.publish(is_public=True)
    output = capsys.readouterr().out
    assert "Failed to complete operation" in output
    assert "Name is already taken" in output

    mock_publish.assert_called_once()


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
@patch(
    "crewai.cli.tools.main.extract_available_exports",
    return_value=[{"name": "SampleTool"}],
)
def test_publish_api_error(
    mock_available_exports,
    mock_publish,
    mock_open,
    mock_listdir,
    mock_subprocess_run,
    mock_get_project_description,
    mock_get_project_version,
    mock_get_project_name,
    capsys,
    tool_command,
):
    mock_response = MagicMock()
    mock_response.status_code = 500
    mock_response.json.return_value = {"error": "Internal Server Error"}
    mock_response.ok = False
    mock_publish.return_value = mock_response

    with raises(SystemExit):
        tool_command.publish(is_public=True)
    output = capsys.readouterr().out
    assert "Request to Enterprise API failed" in output

    mock_publish.assert_called_once()


@patch("crewai.cli.tools.main.Settings")
def test_print_current_organization_with_org(mock_settings, capsys, tool_command):
    mock_settings_instance = MagicMock()
    mock_settings_instance.org_uuid = "test-org-uuid"
    mock_settings_instance.org_name = "Test Organization"
    mock_settings.return_value = mock_settings_instance
    tool_command._print_current_organization()
    output = capsys.readouterr().out
    assert "Current organization: Test Organization (test-org-uuid)" in output


@patch("crewai.cli.tools.main.Settings")
def test_print_current_organization_without_org(mock_settings, capsys, tool_command):
    mock_settings_instance = MagicMock()
    mock_settings_instance.org_uuid = None
    mock_settings_instance.org_name = None
    mock_settings.return_value = mock_settings_instance
    tool_command._print_current_organization()
    output = capsys.readouterr().out
    assert "No organization currently set" in output
    assert "org switch <org_id>" in output
