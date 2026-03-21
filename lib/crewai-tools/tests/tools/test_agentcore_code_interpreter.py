"""Unit tests for AWS Bedrock AgentCore Code Interpreter toolkit."""

from unittest.mock import MagicMock, patch

import pytest

from crewai_tools.aws.bedrock.code_interpreter.code_interpreter_toolkit import (
    ClearContextTool,
    CodeInterpreterToolkit,
    DownloadFileTool,
    DownloadFilesTool,
    ExecuteCodeTool,
    ExecuteCommandTool,
    InstallPackagesTool,
    UploadFileTool,
    UploadFilesTool,
    create_code_interpreter_toolkit,
    extract_output_from_stream,
)


# --- Helpers ---


def make_stream_response(text: str) -> dict:
    """Build a mock code interpreter stream response."""
    return {
        "stream": [
            {
                "result": {
                    "content": [{"type": "text", "text": text}]
                }
            }
        ]
    }


def make_resource_response(path: str, text: str) -> dict:
    """Build a mock code interpreter stream response with a file resource."""
    return {
        "stream": [
            {
                "result": {
                    "content": [
                        {
                            "type": "resource",
                            "resource": {"uri": f"file://{path}", "text": text},
                        }
                    ]
                }
            }
        ]
    }


# --- extract_output_from_stream ---


class TestExtractOutput:
    def test_text_content(self):
        response = make_stream_response("hello world")
        assert extract_output_from_stream(response) == "hello world"

    def test_resource_content(self):
        response = make_resource_response("/tmp/data.csv", "a,b\n1,2")
        output = extract_output_from_stream(response)
        assert "data.csv" in output
        assert "a,b" in output

    def test_empty_stream(self):
        assert extract_output_from_stream({"stream": []}) == ""

    def test_resource_without_text_key(self):
        """Verify JSON fallback when resource has no 'text' key."""
        response = {
            "stream": [
                {
                    "result": {
                        "content": [
                            {
                                "type": "resource",
                                "resource": {"uri": "file:///tmp/out.bin", "size": 1024},
                            }
                        ]
                    }
                }
            ]
        }
        output = extract_output_from_stream(response)
        assert "out.bin" in output
        assert "1024" in output

    def test_multiple_events_joined(self):
        """Verify multiple stream events are joined with newlines."""
        response = {
            "stream": [
                {"result": {"content": [{"type": "text", "text": "line1"}]}},
                {"result": {"content": [{"type": "text", "text": "line2"}]}},
            ]
        }
        output = extract_output_from_stream(response)
        assert "line1" in output
        assert "line2" in output


# --- CodeInterpreterToolkit ---


class TestCodeInterpreterToolkit:
    def test_integration_source_set(self):
        """Verify integration_source='crewai' is passed to CodeInterpreter."""
        MockCI = MagicMock()
        mock_ci = MagicMock()
        mock_ci.session_id = "sess-123"
        MockCI.return_value = mock_ci

        # Pre-seed sys.modules so the lazy import resolves to our mock
        import sys
        fake_module = MagicMock()
        fake_module.CodeInterpreter = MockCI
        with patch.dict(sys.modules, {"bedrock_agentcore.tools.code_interpreter_client": fake_module}):
            toolkit = CodeInterpreterToolkit(region="us-east-1")
            toolkit._get_or_create_interpreter("default")

        MockCI.assert_called_once_with(
            region="us-east-1",
            integration_source="crewai",
        )
        mock_ci.start.assert_called_once_with()

    def test_identifier_passed_to_start(self):
        """Verify custom identifier is passed to start()."""
        MockCI = MagicMock()
        mock_ci = MagicMock()
        mock_ci.session_id = "sess-123"
        MockCI.return_value = mock_ci

        import sys
        fake_module = MagicMock()
        fake_module.CodeInterpreter = MockCI
        with patch.dict(sys.modules, {"bedrock_agentcore.tools.code_interpreter_client": fake_module}):
            toolkit = CodeInterpreterToolkit(
                region="us-west-2", identifier="my-custom-id"
            )
            toolkit._get_or_create_interpreter("default")

        mock_ci.start.assert_called_once_with(identifier="my-custom-id")

    def test_no_identifier_no_kwarg(self):
        """Verify start() called without identifier when not set."""
        MockCI = MagicMock()
        mock_ci = MagicMock()
        mock_ci.session_id = "sess-123"
        MockCI.return_value = mock_ci

        import sys
        fake_module = MagicMock()
        fake_module.CodeInterpreter = MockCI
        with patch.dict(sys.modules, {"bedrock_agentcore.tools.code_interpreter_client": fake_module}):
            toolkit = CodeInterpreterToolkit(region="us-west-2")
            toolkit._get_or_create_interpreter("default")

        mock_ci.start.assert_called_once_with()

    def test_interpreter_reused_for_same_thread(self):
        """Verify same interpreter returned for repeated calls with same thread_id."""
        MockCI = MagicMock()
        mock_ci = MagicMock()
        mock_ci.session_id = "sess-123"
        MockCI.return_value = mock_ci

        import sys
        fake_module = MagicMock()
        fake_module.CodeInterpreter = MockCI
        with patch.dict(sys.modules, {"bedrock_agentcore.tools.code_interpreter_client": fake_module}):
            toolkit = CodeInterpreterToolkit(region="us-west-2")
            ci1 = toolkit._get_or_create_interpreter("thread-a")
            ci2 = toolkit._get_or_create_interpreter("thread-a")

        assert ci1 is ci2
        assert MockCI.call_count == 1

    def test_tool_count(self):
        """Verify all 15 tools are registered."""
        toolkit = CodeInterpreterToolkit(region="us-west-2")
        assert len(toolkit.tools) == 15

    def test_tool_names(self):
        """Verify expected tool names are present."""
        toolkit = CodeInterpreterToolkit(region="us-west-2")
        names = {t.name for t in toolkit.tools}
        expected = {
            "execute_code",
            "execute_command",
            "list_files",
            "read_files",
            "write_files",
            "delete_files",
            "start_command_execution",
            "get_task",
            "stop_task",
            "upload_file",
            "upload_files",
            "install_packages",
            "download_file",
            "download_files",
            "clear_context",
        }
        assert names == expected

    def test_session_timeout_passed_to_start(self):
        """Verify session_timeout_seconds is passed to start()."""
        MockCI = MagicMock()
        mock_ci = MagicMock()
        mock_ci.session_id = "sess-123"
        MockCI.return_value = mock_ci

        import sys
        fake_module = MagicMock()
        fake_module.CodeInterpreter = MockCI
        with patch.dict(sys.modules, {"bedrock_agentcore.tools.code_interpreter_client": fake_module}):
            toolkit = CodeInterpreterToolkit(
                region="us-west-2", session_timeout_seconds=1800
            )
            toolkit._get_or_create_interpreter("default")

        mock_ci.start.assert_called_once_with(session_timeout_seconds=1800)

    def test_session_timeout_with_identifier_passed_to_start(self):
        """Verify both identifier and session_timeout_seconds are passed to start()."""
        MockCI = MagicMock()
        mock_ci = MagicMock()
        mock_ci.session_id = "sess-123"
        MockCI.return_value = mock_ci

        import sys
        fake_module = MagicMock()
        fake_module.CodeInterpreter = MockCI
        with patch.dict(sys.modules, {"bedrock_agentcore.tools.code_interpreter_client": fake_module}):
            toolkit = CodeInterpreterToolkit(
                region="us-west-2",
                identifier="custom-id",
                session_timeout_seconds=3600,
            )
            toolkit._get_or_create_interpreter("default")

        mock_ci.start.assert_called_once_with(
            identifier="custom-id", session_timeout_seconds=3600
        )

    def test_create_code_interpreter_toolkit_passes_identifier(self):
        """Verify factory function passes identifier through."""
        toolkit, tools = create_code_interpreter_toolkit(
            region="us-east-1", identifier="vpc-id"
        )
        assert toolkit.identifier == "vpc-id"
        assert len(tools) == 15

    def test_create_code_interpreter_toolkit_passes_timeout(self):
        """Verify factory function passes session_timeout_seconds through."""
        toolkit, tools = create_code_interpreter_toolkit(
            region="us-east-1", session_timeout_seconds=1800
        )
        assert toolkit.session_timeout_seconds == 1800
        assert len(tools) == 15


# --- New Tool Classes ---


class TestUploadFileTool:
    def test_run_success(self):
        mock_toolkit = MagicMock()
        mock_ci = MagicMock()
        mock_ci.upload_file.return_value = make_stream_response("File uploaded")
        mock_toolkit._get_or_create_interpreter.return_value = mock_ci

        tool = UploadFileTool(mock_toolkit)
        result = tool._run(path="data.csv", content="a,b\n1,2", description="test data")

        mock_ci.upload_file.assert_called_once_with(
            path="data.csv", content="a,b\n1,2", description="test data"
        )
        assert "File uploaded" in result

    def test_run_error(self):
        mock_toolkit = MagicMock()
        mock_toolkit._get_or_create_interpreter.side_effect = RuntimeError("boom")

        tool = UploadFileTool(mock_toolkit)
        result = tool._run(path="x.py", content="code")
        assert "Error uploading file" in result


class TestUploadFilesTool:
    def test_run_success(self):
        mock_toolkit = MagicMock()
        mock_ci = MagicMock()
        mock_ci.upload_files.return_value = make_stream_response("2 files uploaded")
        mock_toolkit._get_or_create_interpreter.return_value = mock_ci

        tool = UploadFilesTool(mock_toolkit)
        files = [
            {"path": "a.py", "content": "x=1"},
            {"path": "b.py", "content": "y=2"},
        ]
        result = tool._run(files=files)

        mock_ci.upload_files.assert_called_once_with(files=files)
        assert "2 files uploaded" in result

    def test_run_error(self):
        mock_toolkit = MagicMock()
        mock_toolkit._get_or_create_interpreter.side_effect = RuntimeError("boom")

        tool = UploadFilesTool(mock_toolkit)
        result = tool._run(files=[{"path": "a.py", "content": "x=1"}])
        assert "Error uploading files" in result


class TestInstallPackagesTool:
    def test_run_success(self):
        mock_toolkit = MagicMock()
        mock_ci = MagicMock()
        mock_ci.install_packages.return_value = make_stream_response(
            "Successfully installed pandas numpy"
        )
        mock_toolkit._get_or_create_interpreter.return_value = mock_ci

        tool = InstallPackagesTool(mock_toolkit)
        result = tool._run(packages=["pandas", "numpy"], upgrade=True)

        mock_ci.install_packages.assert_called_once_with(
            packages=["pandas", "numpy"], upgrade=True
        )
        assert "Successfully installed" in result

    def test_run_default_upgrade_false(self):
        mock_toolkit = MagicMock()
        mock_ci = MagicMock()
        mock_ci.install_packages.return_value = make_stream_response("installed")
        mock_toolkit._get_or_create_interpreter.return_value = mock_ci

        tool = InstallPackagesTool(mock_toolkit)
        tool._run(packages=["requests"])

        mock_ci.install_packages.assert_called_once_with(
            packages=["requests"], upgrade=False
        )

    def test_run_error(self):
        mock_toolkit = MagicMock()
        mock_ci = MagicMock()
        mock_ci.install_packages.side_effect = ValueError("bad package name")
        mock_toolkit._get_or_create_interpreter.return_value = mock_ci

        tool = InstallPackagesTool(mock_toolkit)
        result = tool._run(packages=["bad;pkg"])
        assert "Error installing packages" in result


class TestDownloadFileTool:
    def test_run_text_file(self):
        mock_toolkit = MagicMock()
        mock_ci = MagicMock()
        mock_ci.download_file.return_value = "file contents here"
        mock_toolkit._get_or_create_interpreter.return_value = mock_ci

        tool = DownloadFileTool(mock_toolkit)
        result = tool._run(path="output.txt")

        mock_ci.download_file.assert_called_once_with(path="output.txt")
        assert result == "file contents here"

    def test_run_binary_file(self):
        mock_toolkit = MagicMock()
        mock_ci = MagicMock()
        mock_ci.download_file.return_value = b"\x89PNG binary data"
        mock_toolkit._get_or_create_interpreter.return_value = mock_ci

        tool = DownloadFileTool(mock_toolkit)
        result = tool._run(path="image.png")
        assert "Binary file" in result
        assert "16 bytes" in result

    def test_run_error(self):
        mock_toolkit = MagicMock()
        mock_ci = MagicMock()
        mock_ci.download_file.side_effect = RuntimeError("not found")
        mock_toolkit._get_or_create_interpreter.return_value = mock_ci

        tool = DownloadFileTool(mock_toolkit)
        result = tool._run(path="missing.txt")
        assert "Error downloading file" in result


class TestDownloadFilesTool:
    def test_run_success(self):
        mock_toolkit = MagicMock()
        mock_ci = MagicMock()
        mock_ci.download_files.return_value = {
            "data.csv": "a,b\n1,2",
            "img.png": b"\x89PNG",
        }
        mock_toolkit._get_or_create_interpreter.return_value = mock_ci

        tool = DownloadFilesTool(mock_toolkit)
        result = tool._run(paths=["data.csv", "img.png"])

        assert "data.csv" in result
        assert "a,b" in result
        assert "Binary" in result

    def test_run_empty(self):
        mock_toolkit = MagicMock()
        mock_ci = MagicMock()
        mock_ci.download_files.return_value = {}
        mock_toolkit._get_or_create_interpreter.return_value = mock_ci

        tool = DownloadFilesTool(mock_toolkit)
        result = tool._run(paths=["missing.txt"])
        assert result == "No files found"

    def test_run_error(self):
        mock_toolkit = MagicMock()
        mock_ci = MagicMock()
        mock_ci.download_files.side_effect = RuntimeError("download failed")
        mock_toolkit._get_or_create_interpreter.return_value = mock_ci

        tool = DownloadFilesTool(mock_toolkit)
        result = tool._run(paths=["a.txt"])
        assert "Error downloading files" in result


class TestClearContextTool:
    def test_run_success(self):
        mock_toolkit = MagicMock()
        mock_ci = MagicMock()
        mock_toolkit._get_or_create_interpreter.return_value = mock_ci

        tool = ClearContextTool(mock_toolkit)
        result = tool._run()

        mock_ci.clear_context.assert_called_once()
        assert "cleared successfully" in result

    def test_run_error(self):
        mock_toolkit = MagicMock()
        mock_ci = MagicMock()
        mock_ci.clear_context.side_effect = RuntimeError("fail")
        mock_toolkit._get_or_create_interpreter.return_value = mock_ci

        tool = ClearContextTool(mock_toolkit)
        result = tool._run()
        assert "Error clearing context" in result


# --- Existing Tool Classes (regression) ---


class TestExecuteCodeTool:
    def test_run_success(self):
        mock_toolkit = MagicMock()
        mock_ci = MagicMock()
        mock_ci.invoke.return_value = make_stream_response("42")
        mock_toolkit._get_or_create_interpreter.return_value = mock_ci

        tool = ExecuteCodeTool(mock_toolkit)
        result = tool._run(code="print(42)")

        mock_ci.invoke.assert_called_once_with(
            method="executeCode",
            params={"code": "print(42)", "language": "python", "clearContext": False},
        )
        assert "42" in result


class TestExecuteCommandTool:
    def test_run_success(self):
        mock_toolkit = MagicMock()
        mock_ci = MagicMock()
        mock_ci.invoke.return_value = make_stream_response("Python 3.12.0")
        mock_toolkit._get_or_create_interpreter.return_value = mock_ci

        tool = ExecuteCommandTool(mock_toolkit)
        result = tool._run(command="python --version")

        mock_ci.invoke.assert_called_once_with(
            method="executeCommand", params={"command": "python --version"}
        )
        assert "Python 3.12.0" in result
