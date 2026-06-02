import pytest
from unittest.mock import MagicMock, patch

from crewai_tools.tools.k8s_agent_sandbox.exec_tool import K8sExecTool


@pytest.fixture
def k8s_exec_tool():
    """Fixture to provide a fresh instance of the tool for each test."""
    return K8sExecTool(template="template")


@pytest.fixture
def mock_sandbox():
    """Fixture to provide a mocked sandbox object."""
    return MagicMock()


def test_run_success(k8s_exec_tool, mock_sandbox):
    """Test that a successful command returns the stdout correctly."""

    # 1. Setup the mock response to simulate a successful command (exit_code 0)
    mock_response = MagicMock()
    mock_response.exit_code = 0
    mock_response.stdout = "Hello from the sandbox!\n"
    mock_response.stderr = ""
    mock_sandbox.commands.run.return_value = mock_response

    # 2. Mock the parent class lifecycle methods
    with patch.object(k8s_exec_tool, '_get_sandbox', return_value=(mock_sandbox, True)) as mock_get_sandbox, \
         patch.object(k8s_exec_tool, '_release_sandbox') as mock_release_sandbox:

        # 3. Execute the tool
        result = k8s_exec_tool._run("echo 'Hello from the sandbox!'")

        # 4. Assertions
        assert result == "Hello from the sandbox!\n"

        # Ensure the SDK was called with the exact string
        mock_sandbox.commands.run.assert_called_once_with("echo 'Hello from the sandbox!'")

        # Ensure the sandbox was safely released
        mock_release_sandbox.assert_called_once_with(mock_sandbox, True)


def test_run_failure(k8s_exec_tool, mock_sandbox):
    """Test that a failing command returns the formatted error string."""

    # 1. Setup the mock response to simulate a failed command (exit_code != 0)
    mock_response = MagicMock()
    mock_response.exit_code = 127
    mock_response.stdout = ""
    mock_response.stderr = "bash: invalid_command: command not found"
    mock_sandbox.commands.run.return_value = mock_response

    with patch.object(k8s_exec_tool, '_get_sandbox', return_value=(mock_sandbox, False)), \
         patch.object(k8s_exec_tool, '_release_sandbox') as mock_release_sandbox:

        # 2. Execute the tool
        result = k8s_exec_tool._run("invalid_command")

        # 3. Assertions
        assert "Command execution failed (Exit Code 127):" in result
        assert "bash: invalid_command: command not found" in result

        mock_sandbox.commands.run.assert_called_once_with("invalid_command")
        mock_release_sandbox.assert_called_once_with(mock_sandbox, False)


def test_run_finally_executes_on_exception(k8s_exec_tool, mock_sandbox):
    """Test that the sandbox is safely released even if the SDK crashes."""

    # 1. Simulate the SDK throwing an unhandled exception (e.g., network timeout)
    mock_sandbox.commands.run.side_effect = Exception("SDK Connection Timeout")

    with patch.object(k8s_exec_tool, '_get_sandbox', return_value=(mock_sandbox, True)), \
         patch.object(k8s_exec_tool, '_release_sandbox') as mock_release_sandbox:

        # 2. Execute the tool and assert it raises the error up the chain
        with pytest.raises(Exception, match="SDK Connection Timeout"):
            k8s_exec_tool._run("ls -la")

        # 3. Crucial Assertion: The finally block MUST still fire to prevent cluster resource leaks
        mock_release_sandbox.assert_called_once_with(mock_sandbox, True)
