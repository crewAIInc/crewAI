from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from crewai_tools.tools.vetto_sandbox_tool import (
    VettoBaseTool,
    VettoExecTool,
    VettoFileTool,
    VettoPythonTool,
)


class TestVettoSandboxTools(unittest.TestCase):
    """Test suite for Vetto sandbox tools and containment invariants."""

    def setUp(self):
        """Set up temporary directory and workspace for tests."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.temp_dir.name).resolve()

    def tearDown(self):
        """Clean up temporary directory after tests."""
        self.temp_dir.cleanup()

    def test_instantiation_defaults(self):
        """Verify default configuration attributes of VettoBaseTool."""
        tool = VettoBaseTool(working_dir=str(self.workspace))
        self.assertEqual(tool.net, "off")
        self.assertEqual(tool.timeout, 120)
        self.assertIsNone(tool.memory_limit)
        self.assertFalse(tool.allow_fallback)

    def test_build_command_with_vetto_binary(self):
        """Verify command construction with flags when vetto binary exists."""
        tool = VettoExecTool(
            working_dir=str(self.workspace),
            net="allowlist",
            allowed_domains=["api.anthropic.com", "pypi.org"],
            allow_write=["/tmp/extra_write"],
            allow_read=["/etc/ssl/certs"],
            timeout=45,
            memory_limit="256MB",
            vetto_binary="/bin/vetto",
        )
        with patch.object(tool, "_resolve_vetto_binary", return_value="/bin/vetto"):
            cmd = tool._build_command(["ls", "-la"], cwd=str(self.workspace), timeout=45)

            self.assertEqual(cmd[0], "/bin/vetto")
            self.assertEqual(cmd[1], "run")
            self.assertIn("--net=allowlist", cmd)
            self.assertIn("--timeout", cmd)
            self.assertIn("45", cmd)
            self.assertIn("--memory", cmd)
            self.assertIn("256MB", cmd)
            self.assertIn("--allow-domain", cmd)
            self.assertIn("api.anthropic.com", cmd)
            self.assertIn("pypi.org", cmd)
            self.assertIn("--", cmd)
            self.assertEqual(cmd[-2:], ["ls", "-la"])

    def test_missing_binary_raises_without_fallback(self):
        """Verify RuntimeError is raised when binary is missing and fallback disabled."""
        tool = VettoExecTool(working_dir=str(self.workspace), allow_fallback=False)
        with patch.object(tool, "_resolve_vetto_binary", return_value=None):
            with self.assertRaises(RuntimeError) as ctx:
                tool._build_command(["echo", "hi"])
            self.assertIn("Vetto binary not found", str(ctx.exception))

    def test_missing_binary_passes_with_fallback(self):
        """Verify un-sandboxed command returned when fallback enabled."""
        tool = VettoExecTool(working_dir=str(self.workspace), allow_fallback=True)
        with patch.object(tool, "_resolve_vetto_binary", return_value=None):
            cmd = tool._build_command(["echo", "hi"])
            self.assertEqual(cmd, ["echo", "hi"])

    def test_exec_tool_execution_mocked(self):
        """Verify mocked execution of VettoExecTool returns proper schema."""
        tool = VettoExecTool(working_dir=str(self.workspace), allow_fallback=True)
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_proc.communicate.return_value = (b"output data", b"")
            mock_proc.returncode = 0
            mock_popen.return_value = mock_proc

            result = tool._run("echo hello")
            self.assertEqual(result["exit_code"], 0)
            self.assertEqual(result["stdout"], "output data")
            self.assertFalse(result["timed_out"])

    def test_python_tool_execution_mocked(self):
        """Verify mocked execution of VettoPythonTool returns proper schema."""
        tool = VettoPythonTool(working_dir=str(self.workspace), allow_fallback=True)
        with patch.object(tool, "_execute_subprocess") as mock_exec:
            mock_exec.return_value = {
                "exit_code": 0,
                "stdout": "42\n",
                "stderr": "",
                "timed_out": False,
                "elapsed_seconds": 0.05,
            }
            res = tool._run("print(42)")
            self.assertEqual(res["exit_code"], 0)
            self.assertEqual(res["stdout"], "42\n")
            self.assertTrue(mock_exec.called)

    def test_file_tool_write_read_append(self):
        """Verify write, read, and append operations in VettoFileTool."""
        tool = VettoFileTool(working_dir=str(self.workspace))

        # Write
        res_w = tool._run(action="write", path="test.txt", content="line 1\n")
        self.assertEqual(res_w["status"], "written")
        self.assertTrue((self.workspace / "test.txt").exists())

        # Read
        res_r = tool._run(action="read", path="test.txt")
        self.assertEqual(res_r["content"], "line 1\n")

        # Append
        res_a = tool._run(action="append", path="test.txt", content="line 2\n")
        self.assertEqual(res_a["status"], "appended")

        # Read updated
        res_r2 = tool._run(action="read", path="test.txt")
        self.assertEqual(res_r2["content"], "line 1\nline 2\n")

    def test_file_tool_traversal_rejection(self):
        """Verify path traversal outside workspace is blocked fail-closed."""
        tool = VettoFileTool(working_dir=str(self.workspace))

        # Traversal attempt via relative path
        with self.assertRaises(PermissionError):
            tool._resolve_safe_path("../../../etc/passwd")

        # Traversal attempt via absolute path outside workspace
        outside_path = str(Path("/etc/shadow"))
        with self.assertRaises(PermissionError):
            tool._resolve_safe_path(outside_path)

    def test_file_tool_management_actions(self):
        """Verify directory creation, exists, search, find, and delete actions."""
        tool = VettoFileTool(working_dir=str(self.workspace))

        # mkdir
        res_mkdir = tool._run(action="mkdir", path="subdir")
        self.assertEqual(res_mkdir["status"], "created")
        self.assertTrue((self.workspace / "subdir").is_dir())

        # exists
        res_exists = tool._run(action="exists", path="subdir")
        self.assertTrue(res_exists["exists"])
        self.assertTrue(res_exists["is_dir"])

        # write in subdir
        tool._run(action="write", path="subdir/subfile.py", content="needle = 123\n")

        # list
        res_list = tool._run(action="list", path="subdir")
        self.assertEqual(len(res_list["entries"]), 1)
        self.assertEqual(res_list["entries"][0]["name"], "subfile.py")

        # find (content search)
        res_find = tool._run(action="find", path="subdir", pattern="needle")
        self.assertEqual(len(res_find["matches"]), 1)
        self.assertIn("needle = 123", res_find["matches"][0]["text"])

        # search (filename glob)
        res_search = tool._run(action="search", path="subdir", pattern="*.py")
        self.assertEqual(len(res_search["results"]), 1)

        # delete
        res_del = tool._run(action="delete", path="subdir", recursive=True)
        self.assertEqual(res_del["status"], "deleted")
        self.assertFalse((self.workspace / "subdir").exists())

    def test_exec_tool_cwd_outside_workspace_rejected(self):
        """Verify execution fails when cwd escapes configured workspace."""
        tool = VettoExecTool(working_dir=str(self.workspace), allow_fallback=True)
        with self.assertRaises(PermissionError):
            tool._run("ls", cwd=str(self.workspace.parent))

    def test_file_tool_root_deletion_rejected(self):
        """Verify deletion or moving of root workspace is rejected fail-closed."""
        tool = VettoFileTool(working_dir=str(self.workspace))
        with self.assertRaises(PermissionError):
            tool._run(action="delete", path=str(self.workspace), recursive=True)
        with self.assertRaises(PermissionError):
            tool._run(action="move", path=str(self.workspace), destination=str(self.workspace / "moved"))


if __name__ == "__main__":
    unittest.main()
