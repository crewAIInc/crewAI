import shlex

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

    written_path, written_content = mock_sandbox.files.write.call_args.args
    assert written_content == b"some-code"

    # The code that was written is the code that gets executed.
    assert mock_sandbox.commands.run.call_args.args[0].startswith(
        f"python3 {written_path};"
    )
    assert 0 <= mock_sandbox.commands.run.call_args.kwargs["timeout"] <= 120


def test_staged_code_removed_when_the_command_raises(k8s_python_tool, mock_sandbox):
    mock_sandbox.commands.run.side_effect = [
        RuntimeError("connection lost"),
        ExecutionResult(exit_code=0, stdout="", stderr=""),
    ]

    with pytest.raises(RuntimeError, match="connection lost"):
        k8s_python_tool.run(code="some-code", timeout=120)

    written_path = mock_sandbox.files.write.call_args.args[0]
    assert mock_sandbox.commands.run.call_args.args[0] == (
        f"rm -f -- {shlex.quote(written_path)}"
    )
