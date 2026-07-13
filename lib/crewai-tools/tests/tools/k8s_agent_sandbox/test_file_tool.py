import base64
import shlex
import time

import pytest
from k8s_agent_sandbox.models import (
    ExecutionResult,
    FileEntry,
)

from crewai_tools.tools.k8s_agent_sandbox.file_tool import (
    K8sAgentSandboxFileTool,
    K8sAgentSandboxFileToolOutput,
    FileEntryModel,
)


@pytest.fixture
def k8s_file_tool(sample_toolset):
    return K8sAgentSandboxFileTool(toolset=sample_toolset)


class TestFileToolReadAction:
    @pytest.mark.parametrize("binary", [False, True])
    def test_success(self, k8s_file_tool, mock_sandbox, binary):
        content = b"some-content"
        mock_sandbox.files.read.return_value = content

        result = k8s_file_tool.run(
            action="read", path="some/path", binary=binary, timeout=120
        )

        if binary:
            assert result == K8sAgentSandboxFileToolOutput(
                content=base64.b64encode(content).decode("ascii"),
                path="some/path",
                encoding="base64",
            )
        else:
            assert result == K8sAgentSandboxFileToolOutput(
                content="some-content",
                path="some/path",
                encoding="utf-8",
            )

        mock_sandbox.files.read.assert_called_once_with("some/path", timeout=120)

    def test_invalid_utf(self, k8s_file_tool, mock_sandbox):
        content = b"Some malformed \xff content"
        mock_sandbox.files.read.return_value = content
        result = k8s_file_tool.run(action="read", path="some/path")

        assert result == K8sAgentSandboxFileToolOutput(
            content=base64.b64encode(content).decode("ascii"),
            path="some/path",
            encoding="base64",
            note="File was not valid utf-8; returned as base64.",
        )


class TestFileToolWriteAction:
    @pytest.mark.parametrize("binary", [False, True])
    def test_success(self, k8s_file_tool, mock_sandbox, binary):
        mock_sandbox.commands.run.return_value = ExecutionResult(
            exit_code=0,
            stdout="",
            stderr="",
        )

        content_to_write = "Hello World!"

        if binary:
            content = base64.b64encode(content_to_write.encode("ascii"))
        else:
            content = content_to_write

        result = k8s_file_tool.run(
            action="write",
            path="parent/file.txt",
            content=content,
            binary=binary,
            timeout=120,
        )

        assert result == K8sAgentSandboxFileToolOutput(
            bytes=12,
            path="parent/file.txt",
            status="written",
        )

        mock_sandbox.commands.run.assert_called_once_with(
            "mkdir -p parent", timeout=120
        )

        assert mock_sandbox.files.write.call_args.args == (
            "parent/file.txt",
            b"Hello World!",
        )
        assert 0 <= mock_sandbox.files.write.call_args.kwargs["timeout"] <= 120

    def test_missing_content(self, k8s_file_tool, mock_sandbox):
        mock_sandbox.commands.run.return_value = ExecutionResult(
            exit_code=0,
            stdout="",
            stderr="",
        )

        k8s_file_tool.run(
            action="write",
            path="parent/file.txt",
        )

        assert mock_sandbox.files.write.call_args.args == ("parent/file.txt", b"")

    def test_mkdir_parent_error(self, k8s_file_tool, mock_sandbox):
        mock_sandbox.commands.run.return_value = ExecutionResult(
            exit_code=1,
            stdout="",
            stderr="some parent dir creation error",
        )

        content = "Hello World!"

        with pytest.raises(Exception, match="some parent dir creation error"):
            k8s_file_tool.run(
                action="write",
                path="parent/file.txt",
                content=content,
            )


class TestFileToolAppendAction:
    @pytest.mark.parametrize("binary", [False, True])
    def test_success(self, k8s_file_tool, mock_sandbox, binary):
        mock_sandbox.commands.run.return_value = ExecutionResult(
            exit_code=0,
            stdout="",
            stderr="",
        )

        mock_sandbox.files.read.return_value = b"Hello"

        content_to_append = " World"

        if binary:
            content = base64.b64encode(content_to_append.encode("ascii"))
        else:
            content = content_to_append

        result = k8s_file_tool.run(
            action="append",
            path="parent/file.txt",
            content=content,
            binary=binary,
            timeout=120,
        )

        assert result == K8sAgentSandboxFileToolOutput(
            path="parent/file.txt",
            status="appended",
            appended_bytes=6,
            total_bytes=11,
        )

        mock_sandbox.files.read.assert_called_once()
        assert mock_sandbox.files.read.call_args.args == ("parent/file.txt",)
        assert mock_sandbox.files.read.call_args.kwargs["timeout"] == 120

        mock_sandbox.files.write.assert_called_once()
        assert mock_sandbox.files.write.call_args.args == (
            "parent/file.txt",
            b"Hello World",
        )
        assert 0 <= mock_sandbox.files.write.call_args.kwargs["timeout"] <= 120


class TestFileToolListAction:
    def test_success(self, k8s_file_tool, mock_sandbox):
        modification_time = time.time()
        mock_sandbox.files.list.return_value = [
            FileEntry(
                name="test.txt", size=1000000, type="file", mod_time=modification_time
            ),
            FileEntry(
                name="subdir", size=4096, type="directory", mod_time=modification_time
            ),
        ]

        result = k8s_file_tool.run(action="list", path="some/directory")

        assert result == K8sAgentSandboxFileToolOutput(
            entries=[
                FileEntryModel(
                    mod_time=modification_time,
                    name="test.txt",
                    size=1000000,
                    type="file",
                ),
                FileEntryModel(
                    mod_time=modification_time,
                    name="subdir",
                    size=4096,
                    type="directory",
                ),
            ],
            path="some/directory",
        )


class TestFileToolDeleteAction:
    def test_success(self, k8s_file_tool, mock_sandbox):
        file_path = "parent/file.txt"
        mock_sandbox.commands.run.return_value = ExecutionResult(
            exit_code=0, stdout="", stderr=""
        )
        result = k8s_file_tool.run(action="delete", path=file_path)

        assert result == K8sAgentSandboxFileToolOutput(
            path=file_path,
            status="deleted",
        )

        mock_sandbox.commands.run.assert_called_once()
        assert (
            mock_sandbox.commands.run.call_args.args[0]
            == f"rm -r {shlex.quote(file_path)}"
        )

    @pytest.mark.parametrize(
        "path,expect_error",
        [
            ("/", True),
            ("/usr", True),
            ("/etc", True),
            ("/app", True),
            ("/app/some-path", False),
        ]
    )
    def test_delete_root_error(
        self,
        k8s_file_tool,
        mock_sandbox,
        path,
        expect_error,
    ):
        mock_sandbox.commands.run.return_value = ExecutionResult(
            exit_code=0, stdout="", stderr=""
        )

        if expect_error:
          with pytest.raises(RuntimeError, match="cannot be deleted"):
              k8s_file_tool.run(action="delete", path=path)
        else:
              k8s_file_tool.run(action="delete", path=path)


class TestFileToolMkdirAction:
    def test_success(self, k8s_file_tool, mock_sandbox):
        mock_sandbox.commands.run.return_value = ExecutionResult(
            exit_code=0, stdout="", stderr=""
        )

        result = k8s_file_tool.run(action="mkdir", path="parent/subfolder", timeout=120)

        assert result == K8sAgentSandboxFileToolOutput(
            path="parent/subfolder",
            status="created",
        )

        mock_sandbox.commands.run.assert_called_once_with(
            f"mkdir -p {shlex.quote('parent/subfolder')}",
            timeout=120,
        )


class TestFileToolExistsAction:
    def test_success(self, k8s_file_tool, mock_sandbox):
        mock_sandbox.files.exists.return_value = True

        result = k8s_file_tool.run(action="exists", path="parent/file.txt", timeout=120)

        assert result == K8sAgentSandboxFileToolOutput(exists=True, path="parent/file.txt")

        mock_sandbox.files.exists.assert_called_once_with(
            "parent/file.txt", timeout=120
        )
