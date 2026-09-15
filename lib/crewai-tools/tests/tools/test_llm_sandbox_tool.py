from unittest.mock import MagicMock, patch

import pytest

from crewai_tools import DEFAULT_RUNTIME_CONFIGS, LLMSandboxTool


def _mock_session(exit_code=0, stdout="", stderr=""):
    result = MagicMock(exit_code=exit_code, stdout=stdout, stderr=stderr)
    session = MagicMock()
    session.run.return_value = result
    ctx = MagicMock()
    ctx.__enter__.return_value = session
    ctx.__exit__.return_value = False
    return ctx, session


def test_tool_metadata():
    tool = LLMSandboxTool()
    assert tool.name == "LLM Sandbox"
    assert "llm-sandbox" in tool.package_dependencies


def test_schema_exposes_only_code():
    # A package-installation argument here would let the model choose arbitrary
    # PyPI packages, which executes setup.py at install time.
    props = LLMSandboxTool().args_schema.model_json_schema()["properties"]
    assert set(props) == {"code"}


def test_defaults_are_hardened():
    assert DEFAULT_RUNTIME_CONFIGS["network_mode"] == "none"
    assert DEFAULT_RUNTIME_CONFIGS["cap_drop"] == ["ALL"]
    assert DEFAULT_RUNTIME_CONFIGS["security_opt"] == ["no-new-privileges:true"]
    assert "mem_limit" in DEFAULT_RUNTIME_CONFIGS
    assert "pids_limit" in DEFAULT_RUNTIME_CONFIGS


def test_defaults_omit_unsupported_flags():
    # read_only makes Docker reject the code copy llm-sandbox performs.
    assert "read_only" not in DEFAULT_RUNTIME_CONFIGS
    # cap_drop=ALL alone leaves that copied file unreadable.
    assert DEFAULT_RUNTIME_CONFIGS["cap_add"] == ["DAC_OVERRIDE"]


@patch("llm_sandbox.SandboxSession")
def test_run_returns_stdout(mock_cls):
    ctx, _ = _mock_session(stdout="42\n")
    mock_cls.return_value = ctx
    assert LLMSandboxTool()._run(code="print(6 * 7)") == "42"


@patch("llm_sandbox.SandboxSession")
def test_run_reports_failure(mock_cls):
    ctx, _ = _mock_session(exit_code=1, stderr="Traceback: boom")
    mock_cls.return_value = ctx
    out = LLMSandboxTool()._run(code="raise ValueError")
    assert out.startswith("exit 1")
    assert "boom" in out


@patch("llm_sandbox.SandboxSession")
def test_run_handles_empty_output(mock_cls):
    ctx, _ = _mock_session(stdout="  ")
    mock_cls.return_value = ctx
    assert LLMSandboxTool()._run(code="x = 1") == "(no output)"


@patch("llm_sandbox.SandboxSession")
def test_hardening_reaches_the_session(mock_cls):
    ctx, _ = _mock_session(stdout="ok")
    mock_cls.return_value = ctx
    LLMSandboxTool()._run(code="print('ok')")
    kwargs = mock_cls.call_args.kwargs
    assert kwargs["runtime_configs"] == DEFAULT_RUNTIME_CONFIGS
    # Without this the image is deleted on close and re-pulled every call.
    assert kwargs["keep_template"] is True


@patch("llm_sandbox.SandboxSession")
def test_config_is_forwarded(mock_cls):
    ctx, session = _mock_session(stdout="ok")
    mock_cls.return_value = ctx
    LLMSandboxTool(lang="ruby", backend="podman", timeout=5.0)._run(code="puts 1")
    kwargs = mock_cls.call_args.kwargs
    assert kwargs["lang"] == "ruby"
    assert kwargs["backend"] == "podman"
    assert session.run.call_args.kwargs["timeout"] == 5.0


@patch("llm_sandbox.SandboxSession")
def test_sandbox_errors_do_not_leak_host_detail(mock_cls):
    from llm_sandbox.exceptions import SandboxError

    mock_cls.side_effect = SandboxError("unix:///Users/someone/.docker/run/docker.sock")
    out = LLMSandboxTool()._run(code="print(1)")
    assert out == "sandbox error: execution environment unavailable"
    assert "docker.sock" not in out


def test_image_is_omitted_when_unset():
    assert "image" not in LLMSandboxTool()._session_kwargs()
    assert LLMSandboxTool(image="custom:1.0")._session_kwargs()["image"] == "custom:1.0"
