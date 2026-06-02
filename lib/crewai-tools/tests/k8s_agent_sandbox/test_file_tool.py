import base64
import shlex
import pytest
from unittest.mock import MagicMock, patch

from crewai_tools.tools.k8s_agent_sandbox.file_tool import K8sFileTool


@pytest.fixture
def k8s_file_tool():
    """Fixture to provide a fresh instance of the tool for each test."""
    return K8sFileTool(template="template")


@pytest.fixture
def mock_sandbox():
    """Fixture to provide a mocked sandbox object."""
    return MagicMock()


def test_run_read_success(k8s_file_tool, mock_sandbox):
    """Test the 'read' action cleanly extracts stdout and escapes paths."""
    mock_response = MagicMock(exit_code=0, stdout="file contents\n", stderr="")
    mock_sandbox.commands.run.return_value = mock_response

    file_path = "my secret file.txt" # Testing space escaping
    expected_inner = f"cat {shlex.quote(file_path)}"
    expected_cmd = f"sh -c {shlex.quote(expected_inner)}"

    with patch.object(k8s_file_tool, '_get_sandbox', return_value=(mock_sandbox, True)), \
         patch.object(k8s_file_tool, '_release_sandbox') as mock_release:

        result = k8s_file_tool._run(action="read", file_path=file_path)

        assert result == "file contents\n"
        mock_sandbox.commands.run.assert_called_once_with(expected_cmd)
        mock_release.assert_called_once_with(mock_sandbox, True)


def test_run_write_success(k8s_file_tool, mock_sandbox):
    """Test the 'write' action properly base64 encodes the payload."""
    mock_response = MagicMock(exit_code=0, stdout="", stderr="")
    mock_sandbox.commands.run.return_value = mock_response

    file_path = "/tmp/test.txt"
    content = "Hello World!"
    encoded = base64.b64encode(content.encode('utf-8')).decode('utf-8')

    expected_inner = f"echo '{encoded}' | base64 -d > {shlex.quote(file_path)}"
    expected_cmd = f"sh -c {shlex.quote(expected_inner)}"

    with patch.object(k8s_file_tool, '_get_sandbox', return_value=(mock_sandbox, True)), \
         patch.object(k8s_file_tool, '_release_sandbox') as mock_release:

        result = k8s_file_tool._run(action="write", file_path=file_path, content=content)

        assert result == f"Successfully executed 'write' on {file_path}."
        mock_sandbox.commands.run.assert_called_once_with(expected_cmd)
        mock_release.assert_called_once_with(mock_sandbox, True)


def test_run_append_success(k8s_file_tool, mock_sandbox):
    """Test the 'append' action uses the correct redirect operator."""
    mock_response = MagicMock(exit_code=0, stdout="", stderr="")
    mock_sandbox.commands.run.return_value = mock_response

    content = "New Line"
    encoded = base64.b64encode(content.encode('utf-8')).decode('utf-8')

    expected_inner = f"echo '{encoded}' | base64 -d >> {shlex.quote('log.txt')}"
    expected_cmd = f"sh -c {shlex.quote(expected_inner)}"

    with patch.object(k8s_file_tool, '_get_sandbox', return_value=(mock_sandbox, True)):
        result = k8s_file_tool._run(action="append", file_path="log.txt", content=content)

        assert "Successfully executed 'append'" in result
        mock_sandbox.commands.run.assert_called_once_with(expected_cmd)


def test_run_delete_and_list(k8s_file_tool, mock_sandbox):
    """Test 'delete' and 'list' actions trigger the correct inner shell commands."""
    mock_response = MagicMock(exit_code=0, stdout="total 0\n", stderr="")
    mock_sandbox.commands.run.return_value = mock_response

    with patch.object(k8s_file_tool, '_get_sandbox', return_value=(mock_sandbox, True)):
        # Test Delete
        k8s_file_tool._run(action="delete", file_path="file.txt")
        expected_delete = f"sh -c {shlex.quote('rm -rf ' + shlex.quote('file.txt'))}"
        mock_sandbox.commands.run.assert_called_with(expected_delete)

        # Test List
        k8s_file_tool._run(action="list", file_path="dir/")
        expected_list = f"sh -c {shlex.quote('ls -la ' + shlex.quote('dir/'))}"
        mock_sandbox.commands.run.assert_called_with(expected_list)


def test_missing_content_error(k8s_file_tool, mock_sandbox):
    """Test that writing or appending without content fails immediately."""
    with patch.object(k8s_file_tool, '_get_sandbox', return_value=(mock_sandbox, True)), \
         patch.object(k8s_file_tool, '_release_sandbox') as mock_release:

        result = k8s_file_tool._run(action="write", file_path="test.txt", content=None)

        assert "parameter is required" in result
        mock_sandbox.commands.run.assert_not_called()
        mock_release.assert_called_once_with(mock_sandbox, True)


def test_invalid_action_error(k8s_file_tool, mock_sandbox):
    """Test that an unknown action returns an error and doesn't run the shell."""
    with patch.object(k8s_file_tool, '_get_sandbox', return_value=(mock_sandbox, True)):
        result = k8s_file_tool._run(action="fly", file_path="test.txt")

        assert "Unknown action 'fly'" in result
        mock_sandbox.commands.run.assert_not_called()


def test_run_failure_bubbles_stderr(k8s_file_tool, mock_sandbox):
    """Test that a non-zero exit code bubbles the stderr to the agent."""
    mock_response = MagicMock(exit_code=1, stdout="", stderr="cat: test.txt: No such file or directory")
    mock_sandbox.commands.run.return_value = mock_response

    with patch.object(k8s_file_tool, '_get_sandbox', return_value=(mock_sandbox, True)):
        result = k8s_file_tool._run(action="read", file_path="test.txt")

        assert "Error executing 'read' on test.txt:" in result
        assert "No such file or directory" in result


def test_run_finally_executes_on_exception(k8s_file_tool, mock_sandbox):
    """Test that the sandbox is safely released even if the SDK crashes."""
    mock_sandbox.commands.run.side_effect = Exception("Network Disconnected")

    with patch.object(k8s_file_tool, '_get_sandbox', return_value=(mock_sandbox, True)), \
         patch.object(k8s_file_tool, '_release_sandbox') as mock_release:

        with pytest.raises(Exception, match="Network Disconnected"):
            k8s_file_tool._run(action="read", file_path="test.txt")

        mock_release.assert_called_once_with(mock_sandbox, True)
