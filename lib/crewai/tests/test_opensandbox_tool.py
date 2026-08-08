"""Tests for ``OpenSandboxTool``.

These tests mock the underlying ``opensandbox`` SDK so the suite runs
without a real OpenSandbox server. The mocks live inside the
``crewai.tools.opensandbox_tool`` module so the locally imported names
(``Sandbox``, ``ConnectionConfig``, ``WriteEntry``) are intercepted.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai.tools.opensandbox_tool import OpenSandboxTool


def _stub_opensandbox_modules() -> None:
    """Install stub ``opensandbox`` submodules so deferred imports succeed.

    The tool imports ``opensandbox`` lazily inside its async helpers; tests
    patch the symbols on those modules directly. We pre-create the module
    objects so ``patch`` can find an attribute to replace.
    """
    pkg = sys.modules.setdefault("opensandbox", types.ModuleType("opensandbox"))
    config_pkg = sys.modules.setdefault(
        "opensandbox.config", types.ModuleType("opensandbox.config")
    )
    connection_mod = sys.modules.setdefault(
        "opensandbox.config.connection",
        types.ModuleType("opensandbox.config.connection"),
    )
    models_mod = sys.modules.setdefault(
        "opensandbox.models", types.ModuleType("opensandbox.models")
    )

    if not hasattr(pkg, "Sandbox"):
        pkg.Sandbox = MagicMock()
    if not hasattr(connection_mod, "ConnectionConfig"):
        connection_mod.ConnectionConfig = MagicMock()
    if not hasattr(config_pkg, "connection"):
        config_pkg.connection = connection_mod
    if not hasattr(models_mod, "WriteEntry"):
        models_mod.WriteEntry = MagicMock()


_stub_opensandbox_modules()


def _make_execution(stdout: str = "", stderr: str = "", error: str | None = None) -> SimpleNamespace:
    stdout_items = [SimpleNamespace(text=stdout)] if stdout else []
    stderr_items = [SimpleNamespace(text=stderr)] if stderr else []
    return SimpleNamespace(
        logs=SimpleNamespace(stdout=stdout_items, stderr=stderr_items),
        error=error,
    )


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("OPENSANDBOX_DOMAIN", "localhost:8080")
    monkeypatch.setenv("OPENSANDBOX_PROTOCOL", "http")
    monkeypatch.setenv("OPENSANDBOX_IMAGE", "python:3.12")
    monkeypatch.setenv("OPENSANDBOX_TIMEOUT_MINUTES", "30")
    monkeypatch.delenv("OPENSANDBOX_API_KEY", raising=False)


@pytest.fixture
def fake_sandbox():
    sandbox = MagicMock()
    sandbox.commands.run = AsyncMock(return_value=_make_execution(stdout="hello\n"))
    sandbox.files.read_file = AsyncMock(return_value="file contents")
    sandbox.files.write_files = AsyncMock(return_value=None)
    sandbox.kill = AsyncMock(return_value=None)
    return sandbox


@pytest.fixture
def patched_sandbox_create(fake_sandbox):
    with patch("opensandbox.Sandbox") as sandbox_cls:
        sandbox_cls.create = AsyncMock(return_value=fake_sandbox)
        yield sandbox_cls, fake_sandbox


def test_missing_domain_raises_value_error(monkeypatch):
    monkeypatch.delenv("OPENSANDBOX_DOMAIN", raising=False)
    tool = OpenSandboxTool()
    with pytest.raises(ValueError, match="OPENSANDBOX_DOMAIN is not set"):
        tool._build_connection_config()


def test_run_command_returns_stdout(env, patched_sandbox_create):
    _, fake = patched_sandbox_create
    tool = OpenSandboxTool()
    result = tool.run(action="run_command", command="echo hello")
    assert "hello" in result
    fake.commands.run.assert_awaited_once_with("echo hello")


def test_run_command_includes_stderr_and_error(env, fake_sandbox):
    fake_sandbox.commands.run = AsyncMock(
        return_value=_make_execution(stdout="ok", stderr="warn", error="boom")
    )
    with patch("opensandbox.Sandbox") as sandbox_cls:
        sandbox_cls.create = AsyncMock(return_value=fake_sandbox)
        tool = OpenSandboxTool()
        result = tool.run(action="run_command", command="do_thing")
    assert "ok" in result
    assert "stderr:\nwarn" in result
    assert "error: boom" in result


def test_run_command_requires_command(env):
    tool = OpenSandboxTool()
    result = tool.run(action="run_command")
    assert "command" in result.lower()
    assert "required" in result.lower()


def test_read_file(env, patched_sandbox_create):
    _, fake = patched_sandbox_create
    tool = OpenSandboxTool()
    result = tool.run(action="read_file", path="/tmp/foo.txt")
    assert result == "file contents"
    fake.files.read_file.assert_awaited_once_with("/tmp/foo.txt")


def test_read_file_requires_path(env):
    tool = OpenSandboxTool()
    result = tool.run(action="read_file")
    assert "path" in result.lower()
    assert "required" in result.lower()


def test_write_file(env, patched_sandbox_create):
    _, fake = patched_sandbox_create
    with patch("opensandbox.models.WriteEntry") as write_entry:
        write_entry.side_effect = lambda **kwargs: kwargs
        tool = OpenSandboxTool()
        result = tool.run(
            action="write_file", path="/tmp/foo.txt", content="hello"
        )
    assert "Wrote" in result
    assert "/tmp/foo.txt" in result
    fake.files.write_files.assert_awaited_once()
    entries = fake.files.write_files.await_args.args[0]
    assert entries[0]["path"] == "/tmp/foo.txt"
    assert entries[0]["data"] == "hello"


def test_write_file_requires_path_and_content(env):
    tool = OpenSandboxTool()
    no_path = tool.run(action="write_file", content="x")
    assert "path" in no_path.lower()
    no_content = tool.run(action="write_file", path="/tmp/x")
    assert "content" in no_content.lower()


def test_kill_when_no_sandbox(env):
    tool = OpenSandboxTool()
    result = tool.run(action="kill")
    assert result == "No sandbox to kill."


def test_kill_after_use(env, patched_sandbox_create):
    _, fake = patched_sandbox_create
    tool = OpenSandboxTool()
    tool.run(action="run_command", command="echo hi")
    result = tool.run(action="kill")
    assert "killed" in result.lower()
    fake.kill.assert_awaited_once()
    assert tool._sandbox is None


def test_sandbox_reused_across_calls(env, patched_sandbox_create):
    sandbox_cls, fake = patched_sandbox_create
    tool = OpenSandboxTool()
    tool.run(action="run_command", command="echo 1")
    tool.run(action="read_file", path="/tmp/x")
    assert sandbox_cls.create.await_count == 1


def test_run_command_wraps_sdk_exception(env, fake_sandbox):
    fake_sandbox.commands.run = AsyncMock(side_effect=RuntimeError("network down"))
    with patch("opensandbox.Sandbox") as sandbox_cls:
        sandbox_cls.create = AsyncMock(return_value=fake_sandbox)
        tool = OpenSandboxTool()
        result = tool.run(action="run_command", command="echo hi")
    assert "OpenSandbox error" in result
    assert "network down" in result


def test_unknown_action(env):
    tool = OpenSandboxTool()
    # Bypass schema validation by calling _run directly with an invalid action.
    result = tool._run(action="not_a_real_action")
    assert "unknown action" in result.lower()
