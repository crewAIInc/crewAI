import pytest
from k8s_agent_sandbox.models import ExecutionResult

from crewai_tools.tools.k8s_agent_sandbox.exec_tool import K8sAgentSandboxExecTool
from crewai_tools.tools.k8s_agent_sandbox.toolset import K8sAgentSandboxToolset


@pytest.fixture
def k8s_exec_tool(sample_toolset: K8sAgentSandboxToolset):
    return K8sAgentSandboxExecTool(toolset=sample_toolset)


@pytest.mark.parametrize("exit_code", [0, 127])
@pytest.mark.usefixtures("mock_client_returns_mock_sandbox_in_create_sandbox")
def test_run_success(exit_code, k8s_exec_tool, mock_sandbox):

    mock_sandbox.commands.run.return_value = ExecutionResult(
        exit_code=exit_code,
        stdout="some-output",
        stderr="some-logs",
    )

    result = k8s_exec_tool.run("some-command", timeout=120)

    assert result == {
        "exit_code": exit_code,
        "stdout": "some-output",
        "stderr": "some-logs",
    }

    mock_sandbox.commands.run.assert_called_once_with("some-command", timeout=120)
