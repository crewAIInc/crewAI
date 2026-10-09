from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from crewai_tools import SmolExecTool, SmolFileTool, SmolPythonTool
from crewai_tools.tools.smol_sandbox_tool.smol_base_tool import SmolBaseTool


@pytest.fixture
def sdk(monkeypatch):
    machine = MagicMock()
    machine.id = "mach-123"
    machine.state.return_value = "running"
    machine.exec.return_value = SimpleNamespace(
        exit_code=0,
        stdout="hello\n",
        stderr="",
        stdout_truncated=False,
        stderr_truncated=False,
    )
    machine.read_file.return_value = b"hello\n"
    client = SimpleNamespace(
        Machine=SimpleNamespace(
            create=MagicMock(return_value=machine),
            connect=MagicMock(return_value=machine),
        ),
        MachineConfig=lambda **kw: SimpleNamespace(**kw),
        ConnectOptions=lambda **kw: SimpleNamespace(**kw),
        ExecOptions=lambda **kw: SimpleNamespace(**kw),
    )
    monkeypatch.setattr(SmolBaseTool, "_sdk", staticmethod(lambda: client))
    return client, machine


def test_local_shell_exec_does_not_select_cloud_from_environment(sdk, monkeypatch):
    client, machine = sdk
    monkeypatch.setenv("SMOL_CLOUD_TOKEN", "unused-token")
    tool = SmolExecTool()
    result = tool.run(command="echo hello", cwd="/workspace", timeout=5)

    assert result["stdout"] == "hello\n"
    assert result["exit_code"] == 0
    assert client.Machine.create.call_args.args[1].target == "local"
    assert client.Machine.create.call_args.args[0].network is True
    assert machine.exec.call_args.args[0] == ["sh", "-lc", "echo hello"]
    assert machine.exec.call_args.args[1].workdir == "/workspace"
    machine.delete.assert_called_once_with()


def test_persistent_shell_and_attached_files_share_vm_without_double_delete(sdk):
    client, machine = sdk
    shell = SmolExecTool(persistent=True, network=False)
    try:
        shell.run(command="echo first")
        shell.run(command="echo second")
        assert client.Machine.create.call_count == 1
        assert client.Machine.create.call_args.args[0].network is False
        assert client.Machine.create.call_args.args[0].persistent is True
        assert shell.active_machine_id == "mach-123"
        files = SmolFileTool(machine_id=shell.active_machine_id)
        assert files.run(action="write", path="/workspace/a", content="hi") == (
            "Wrote 2 bytes to /workspace/a"
        )
        assert client.Machine.connect.call_count == 1
        assert client.Machine.connect.call_args.args[0] == "mach-123"
        assert client.Machine.connect.call_args.args[1].target == "local"
        machine.write_file.assert_called_once_with("/workspace/a", b"hi")
        files.close()
        machine.delete.assert_not_called()
    finally:
        shell.close()
        shell.close()
    machine.delete.assert_called_once_with()


def test_ephemeral_machine_deleted_when_command_fails(sdk):
    _, machine = sdk
    machine.exec.side_effect = RuntimeError("command failed")
    with pytest.raises(RuntimeError, match="command failed"):
        SmolExecTool().run(command="false")
    machine.delete.assert_called_once_with()


def test_cloud_attachment_starts_stopped_vm_and_preserves_ownership(sdk):
    client, machine = sdk
    machine.state.return_value = "stopped"
    tool = SmolExecTool(target="cloud", machine_id="mach-123", api_key="test-token")
    assert tool.run(command="echo hello")["exit_code"] == 0
    options = client.Machine.connect.call_args.args[1]
    assert options.target == "cloud" and options.api_key == "test-token"
    machine.start.assert_called_once_with()
    machine.wait_until_ready.assert_called_once_with()
    client.Machine.create.assert_not_called()
    machine.delete.assert_not_called()


def test_python_argv_is_passed_without_shell_interpolation(sdk):
    _, machine = sdk
    tool = SmolPythonTool()
    result = tool.run(code="import sys; print(sys.argv[1])", argv=["; echo unsafe"])
    assert result["stdout"] == "hello\n"
    assert machine.exec.call_args.args[0] == [
        "python", "-c", "import sys; print(sys.argv[1])", "; echo unsafe"
    ]
    machine.delete.assert_called_once_with()


def test_file_data_and_validation(sdk):
    client, machine = sdk
    tool = SmolFileTool()
    assert tool.run(action="read", path="/workspace/a") == "hello\n"
    assert tool.run(action="read", path="/workspace/a", binary=True) == "aGVsbG8K"
    assert tool.run(action="write", path="/workspace/a", content="AAE=", binary=True) == (
        "Wrote 2 bytes to /workspace/a"
    )
    machine.write_file.assert_called_with("/workspace/a", b"\x00\x01")
    with pytest.raises(ValueError, match="absolute"):
        tool.run(action="read", path="relative/path")
    with pytest.raises(ValueError, match="valid base64"):
        tool.run(action="write", path="/workspace/a", content="!!!", binary=True)
    assert client.Machine.create.call_count == 4
    assert machine.delete.call_count == 4
