"""Tests for OpenSandboxTool."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from crewai_tools.tools.open_sandbox_tool.open_sandbox_tool import (
    OpenSandboxTool,
    OpenSandboxToolSchema,
)


# --- Schema Tests ---


class TestOpenSandboxToolSchema:
    def test_schema_requires_code(self):
        with pytest.raises(Exception):
            OpenSandboxToolSchema()

    def test_schema_with_code_only(self):
        schema = OpenSandboxToolSchema(code="print('hello')")
        assert schema.code == "print('hello')"
        assert schema.libraries == []

    def test_schema_with_libraries(self):
        schema = OpenSandboxToolSchema(
            code="import pandas", libraries=["pandas", "numpy"]
        )
        assert schema.libraries == ["pandas", "numpy"]


# --- Tool Instantiation Tests ---


class TestOpenSandboxToolInit:
    def test_default_values(self):
        tool = OpenSandboxTool()
        assert tool.name == "Open Sandbox Code Interpreter"
        assert tool.opensandbox_domain == "localhost:8080"
        assert tool.opensandbox_protocol == "http"
        assert tool.opensandbox_image == "opensandbox/code-interpreter:v1.0.2"
        assert tool.timeout == 300
        assert tool.resource is None

    def test_custom_values(self):
        tool = OpenSandboxTool(
            opensandbox_api_key="test-key",
            opensandbox_domain="sandbox.example.com",
            opensandbox_protocol="https",
            opensandbox_image="custom-image:latest",
            timeout=600,
            resource={"cpu": "2", "memory": "4Gi"},
        )
        assert tool.opensandbox_api_key == "test-key"
        assert tool.opensandbox_domain == "sandbox.example.com"
        assert tool.opensandbox_protocol == "https"
        assert tool.opensandbox_image == "custom-image:latest"
        assert tool.timeout == 600
        assert tool.resource == {"cpu": "2", "memory": "4Gi"}

    def test_api_key_from_env(self):
        tool = OpenSandboxTool()
        with patch.dict("os.environ", {"OPENSANDBOX_API_KEY": "env-key"}):
            assert tool._get_api_key() == "env-key"

    def test_api_key_param_overrides_env(self):
        tool = OpenSandboxTool(opensandbox_api_key="param-key")
        with patch.dict("os.environ", {"OPENSANDBOX_API_KEY": "env-key"}):
            assert tool._get_api_key() == "param-key"


# --- Helpers for mocking OpenSandbox SDK ---


def _make_output_message(text: str, is_error: bool = False):
    msg = MagicMock()
    msg.text = text
    msg.is_error = is_error
    return msg


def _make_execution_result(text: str | None = None):
    result = MagicMock()
    result.text = text
    return result


def _make_execution(
    stdout: list[str] | None = None,
    stderr: list[str] | None = None,
    results: list[str | None] | None = None,
    error: dict | None = None,
):
    execution = MagicMock()
    execution.logs.stdout = [_make_output_message(t) for t in (stdout or [])]
    execution.logs.stderr = [_make_output_message(t, is_error=True) for t in (stderr or [])]
    execution.result = [_make_execution_result(t) for t in (results or [])]

    if error:
        execution.error = MagicMock()
        execution.error.name = error.get("name", "Error")
        execution.error.value = error.get("value", "")
        execution.error.traceback = error.get("traceback", [])
    else:
        execution.error = None

    return execution


def _make_command_result(error=None, stderr_text=""):
    result = MagicMock()
    result.error = error
    result.logs.stderr = [_make_output_message(stderr_text)] if stderr_text else []
    return result


# --- Execution Tests ---


class TestOpenSandboxToolRun:
    @pytest.mark.asyncio
    async def test_missing_import_returns_error(self):
        tool = OpenSandboxTool()
        with patch.dict("sys.modules", {"opensandbox": None, "code_interpreter": None}):
            with patch(
                "crewai_tools.tools.open_sandbox_tool.open_sandbox_tool.OpenSandboxTool._arun"
            ) as mock_arun:
                mock_arun.return_value = (
                    "Error: Missing required packages. Install with:\n"
                    "  pip install opensandbox opensandbox-code-interpreter\n"
                    "Details: No module named 'opensandbox'"
                )
                result = await mock_arun(code="print('hi')")
                assert "Missing required packages" in result

    @pytest.mark.asyncio
    async def test_successful_execution(self):
        tool = OpenSandboxTool()
        execution = _make_execution(stdout=["Hello, World!\n"])

        mock_sandbox = AsyncMock()
        mock_interpreter = AsyncMock()
        mock_interpreter.codes.run = AsyncMock(return_value=execution)

        tool._sandbox = mock_sandbox
        tool._interpreter = mock_interpreter

        with patch(
            "crewai_tools.tools.open_sandbox_tool.open_sandbox_tool.OpenSandboxTool._arun",
            wraps=tool._arun,
        ):
            # Mock the imports inside _arun
            mock_modules = {
                "code_interpreter": MagicMock(),
                "code_interpreter.models.code": MagicMock(),
                "opensandbox": MagicMock(),
                "opensandbox.config": MagicMock(),
            }
            with patch.dict("sys.modules", mock_modules):
                result = await tool._arun(code="print('Hello, World!')")

        assert "[stdout]" in result
        assert "Hello, World!" in result

    @pytest.mark.asyncio
    async def test_execution_with_error(self):
        tool = OpenSandboxTool()
        execution = _make_execution(
            error={
                "name": "NameError",
                "value": "name 'foo' is not defined",
                "traceback": ["  File \"<stdin>\", line 1"],
            }
        )

        mock_sandbox = AsyncMock()
        mock_interpreter = AsyncMock()
        mock_interpreter.codes.run = AsyncMock(return_value=execution)

        tool._sandbox = mock_sandbox
        tool._interpreter = mock_interpreter

        mock_modules = {
            "code_interpreter": MagicMock(),
            "code_interpreter.models.code": MagicMock(),
            "opensandbox": MagicMock(),
            "opensandbox.config": MagicMock(),
        }
        with patch.dict("sys.modules", mock_modules):
            result = await tool._arun(code="print(foo)")

        assert "NameError" in result
        assert "name 'foo' is not defined" in result

    @pytest.mark.asyncio
    async def test_no_output(self):
        tool = OpenSandboxTool()
        execution = _make_execution()

        mock_sandbox = AsyncMock()
        mock_interpreter = AsyncMock()
        mock_interpreter.codes.run = AsyncMock(return_value=execution)

        tool._sandbox = mock_sandbox
        tool._interpreter = mock_interpreter

        mock_modules = {
            "code_interpreter": MagicMock(),
            "code_interpreter.models.code": MagicMock(),
            "opensandbox": MagicMock(),
            "opensandbox.config": MagicMock(),
        }
        with patch.dict("sys.modules", mock_modules):
            result = await tool._arun(code="x = 1")

        assert result == "Code executed successfully (no output)."


# --- Library Installation Tests ---


class TestOpenSandboxToolLibraries:
    @pytest.mark.asyncio
    async def test_install_libraries(self):
        tool = OpenSandboxTool()

        mock_sandbox = AsyncMock()
        mock_sandbox.commands.run = AsyncMock(
            return_value=_make_command_result()
        )
        mock_interpreter = AsyncMock()
        mock_interpreter.codes.run = AsyncMock(
            return_value=_make_execution(stdout=["done\n"])
        )

        tool._sandbox = mock_sandbox
        tool._interpreter = mock_interpreter
        tool._installed_libraries = set()

        mock_modules = {
            "code_interpreter": MagicMock(),
            "code_interpreter.models.code": MagicMock(),
            "opensandbox": MagicMock(),
            "opensandbox.config": MagicMock(),
        }
        with patch.dict("sys.modules", mock_modules):
            result = await tool._arun(
                code="import pandas; print('done')",
                libraries=["pandas"],
            )

        mock_sandbox.commands.run.assert_called_once_with("pip install pandas")
        assert "pandas" in tool._installed_libraries
        assert "[stdout]" in result

    @pytest.mark.asyncio
    async def test_skip_already_installed_libraries(self):
        tool = OpenSandboxTool()

        mock_sandbox = AsyncMock()
        mock_interpreter = AsyncMock()
        mock_interpreter.codes.run = AsyncMock(
            return_value=_make_execution(stdout=["ok\n"])
        )

        tool._sandbox = mock_sandbox
        tool._interpreter = mock_interpreter
        tool._installed_libraries = {"pandas"}

        mock_modules = {
            "code_interpreter": MagicMock(),
            "code_interpreter.models.code": MagicMock(),
            "opensandbox": MagicMock(),
            "opensandbox.config": MagicMock(),
        }
        with patch.dict("sys.modules", mock_modules):
            await tool._arun(
                code="import pandas",
                libraries=["pandas"],
            )

        # Should not call pip install since pandas is already installed
        mock_sandbox.commands.run.assert_not_called()


# --- Output Formatting Tests ---


class TestFormatExecutionResult:
    def test_stdout_only(self):
        tool = OpenSandboxTool()
        execution = _make_execution(stdout=["hello\n"])
        result = tool._format_execution_result(execution)
        assert result == "[stdout]\nhello\n"

    def test_stderr_only(self):
        tool = OpenSandboxTool()
        execution = _make_execution(stderr=["warning: something\n"])
        result = tool._format_execution_result(execution)
        assert result == "[stderr]\nwarning: something\n"

    def test_result_only(self):
        tool = OpenSandboxTool()
        execution = _make_execution(results=["42"])
        result = tool._format_execution_result(execution)
        assert result == "[result]\n42"

    def test_error_with_traceback(self):
        tool = OpenSandboxTool()
        execution = _make_execution(
            error={
                "name": "ValueError",
                "value": "bad value",
                "traceback": ["line 1", "line 2"],
            }
        )
        result = tool._format_execution_result(execution)
        assert "[error] ValueError: bad value" in result
        assert "line 1\nline 2" in result

    def test_combined_output(self):
        tool = OpenSandboxTool()
        execution = _make_execution(
            stdout=["out\n"],
            stderr=["err\n"],
            results=["res"],
        )
        result = tool._format_execution_result(execution)
        assert "[stdout]" in result
        assert "[stderr]" in result
        assert "[result]" in result

    def test_empty_execution(self):
        tool = OpenSandboxTool()
        execution = _make_execution()
        result = tool._format_execution_result(execution)
        assert result == "Code executed successfully (no output)."


# --- Cleanup Tests ---


class TestOpenSandboxToolCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_kills_sandbox(self):
        tool = OpenSandboxTool()
        mock_sandbox = AsyncMock()
        tool._sandbox = mock_sandbox
        tool._interpreter = AsyncMock()
        tool._installed_libraries = {"pandas"}

        await tool.cleanup()

        mock_sandbox.kill.assert_called_once()
        mock_sandbox.close.assert_called_once()
        assert tool._sandbox is None
        assert tool._interpreter is None
        assert tool._installed_libraries == set()

    @pytest.mark.asyncio
    async def test_cleanup_handles_errors(self):
        tool = OpenSandboxTool()
        mock_sandbox = AsyncMock()
        mock_sandbox.kill.side_effect = Exception("connection lost")
        tool._sandbox = mock_sandbox
        tool._interpreter = AsyncMock()

        # Should not raise
        await tool.cleanup()
        assert tool._sandbox is None

    @pytest.mark.asyncio
    async def test_cleanup_when_no_sandbox(self):
        tool = OpenSandboxTool()
        # Should not raise
        await tool.cleanup()


# --- Sandbox Creation Tests ---


class TestOpenSandboxToolCreate:
    @pytest.mark.asyncio
    async def test_creates_sandbox_on_first_run(self):
        tool = OpenSandboxTool(
            opensandbox_api_key="test-key",
            opensandbox_domain="test.example.com",
            opensandbox_protocol="https",
            timeout=120,
            resource={"cpu": "2", "memory": "4Gi"},
        )

        mock_sandbox = AsyncMock()
        mock_interpreter = AsyncMock()
        mock_interpreter.codes.run = AsyncMock(
            return_value=_make_execution(stdout=["hi\n"])
        )

        mock_sandbox_cls = AsyncMock()
        mock_sandbox_cls.create = AsyncMock(return_value=mock_sandbox)

        mock_interpreter_cls = AsyncMock()
        mock_interpreter_cls.create = AsyncMock(return_value=mock_interpreter)

        mock_config_cls = MagicMock()

        await tool._create_sandbox(mock_sandbox_cls, mock_interpreter_cls, mock_config_cls)

        mock_config_cls.assert_called_once_with(
            domain="test.example.com",
            protocol="https",
            api_key="test-key",
        )
        mock_sandbox_cls.create.assert_called_once()
        mock_interpreter_cls.create.assert_called_once_with(sandbox=mock_sandbox)
        assert tool._sandbox is mock_sandbox
        assert tool._interpreter is mock_interpreter
