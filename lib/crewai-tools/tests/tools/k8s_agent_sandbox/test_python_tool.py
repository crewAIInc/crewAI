import pytest

from k8s_agent_sandbox.models import ExecutionResult

from crewai_tools.tools.k8s_agent_sandbox.python_tool import K8sAgentSandboxPythonTool


@pytest.fixture
def k8s_python_tool(sample_toolset):
    return K8sAgentSandboxPythonTool(toolset=sample_toolset)


@pytest.mark.parametrize("exit_code", [0, 127])
def test_run(k8s_python_tool, mock_sandbox, exit_code):
    mock_sandbox.commands.run.return_value = ExecutionResult(
        exit_code=exit_code,
        stdout="some-output",
        stderr="some-logs",
    )

    result = k8s_python_tool.run(code="some-code", timeout=120)

    assert result == {
        "exit_code": exit_code,
        "stdout": "some-output",
        "stderr": "some-logs",
    }

    mock_sandbox.files.write.assert_called_once_with("main.py", "some-code", timeout=120)

    assert mock_sandbox.commands.run.call_args.args == ("python3 main.py",)
    assert 0 <= mock_sandbox.commands.run.call_args.kwargs["timeout"] <= 120
