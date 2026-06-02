import base64
import pytest
from unittest.mock import MagicMock, patch

from crewai_tools.tools.k8s_agent_sandbox.python_tool import K8sPythonTool


@pytest.fixture
def k8s_python_tool():
    """Fixture to provide a fresh instance of the tool for each test."""
    return K8sPythonTool(template="template")


@pytest.fixture
def mock_sandbox():
    """Fixture to provide a mocked sandbox object."""
    return MagicMock()


def test_run_success(k8s_python_tool, mock_sandbox):
    """Test that a successfully executed Python script returns stdout."""

    # 1. Setup the mock response
    mock_response = MagicMock()
    mock_response.exit_code = 0
    mock_response.stdout = "Hello from Python!\n"
    mock_response.stderr = ""
    mock_sandbox.commands.run.return_value = mock_response

    # 2. Define the input code and calculate what the expected safe command should be
    input_code = "print('Hello from Python!')"
    expected_encoded = base64.b64encode(input_code.encode('utf-8')).decode('utf-8')
    expected_command = f'python -c "import base64; exec(base64.b64decode(\'{expected_encoded}\').decode(\'utf-8\'))"'

    # 3. Mock the lifecycle methods and execute
    with patch.object(k8s_python_tool, '_get_sandbox', return_value=(mock_sandbox, True)), \
         patch.object(k8s_python_tool, '_release_sandbox') as mock_release_sandbox:

        result = k8s_python_tool._run(input_code)

        # 4. Assertions
        assert result == "Hello from Python!\n"

        # Verify the sandbox received the exact base64 wrapped command
        mock_sandbox.commands.run.assert_called_once_with(expected_command)
        mock_release_sandbox.assert_called_once_with(mock_sandbox, True)


def test_run_failure(k8s_python_tool, mock_sandbox):
    """Test that a failing Python script returns the formatted error string."""

    # 1. Setup the mock response for a syntax error
    mock_response = MagicMock()
    mock_response.exit_code = 1
    mock_response.stdout = ""
    mock_response.stderr = "SyntaxError: invalid syntax"
    mock_sandbox.commands.run.return_value = mock_response

    with patch.object(k8s_python_tool, '_get_sandbox', return_value=(mock_sandbox, False)), \
         patch.object(k8s_python_tool, '_release_sandbox') as mock_release_sandbox:

        # 2. Execute a broken script
        result = k8s_python_tool._run("print('Missing quote)")

        # 3. Assertions
        assert "Python execution failed (Exit Code 1):" in result
        assert "SyntaxError: invalid syntax" in result

        mock_sandbox.commands.run.assert_called_once()
        mock_release_sandbox.assert_called_once_with(mock_sandbox, False)


def test_run_finally_executes_on_exception(k8s_python_tool, mock_sandbox):
    """Test that the sandbox is safely released even if the SDK crashes."""

    # 1. Simulate the SDK throwing an unhandled exception
    mock_sandbox.commands.run.side_effect = Exception("SDK Connection Timeout")

    with patch.object(k8s_python_tool, '_get_sandbox', return_value=(mock_sandbox, True)), \
         patch.object(k8s_python_tool, '_release_sandbox') as mock_release_sandbox:

        # 2. Execute the tool and assert it raises the error
        with pytest.raises(Exception, match="SDK Connection Timeout"):
            k8s_python_tool._run("x = 10")

        # 3. Ensure the finally block fires
        mock_release_sandbox.assert_called_once_with(mock_sandbox, True)
