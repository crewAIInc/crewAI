import base64
import pytest

from k8s_agent_sandbox.models import ExecutionResult

from crewai_tools.tools.k8s_agent_sandbox.python_tool import (
    K8sAgentSandboxPythonTool,
    K8sAgentSandboxPythonToolOutput,
)


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

    assert result == K8sAgentSandboxPythonToolOutput(
        exit_code=exit_code,
        stdout="some-output",
        stderr="some-logs",
    )

    assert mock_sandbox.files.write.called

    assert mock_sandbox.commands.run.call_args.args[0].startswith("python3 /tmp",)
    assert 0 <= mock_sandbox.commands.run.call_args.kwargs["timeout"] <= 120
