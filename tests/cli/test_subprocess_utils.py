import platform
import subprocess
from unittest import mock

import pytest

from crewai.cli.subprocess_utils import run_command


class TestRunCommand:
    """Test the cross-platform subprocess utility."""

    @mock.patch("platform.system")
    @mock.patch("subprocess.run")
    def test_windows_uses_shell_true(self, mock_subprocess_run, mock_platform):
        """Test that Windows uses shell=True with proper command conversion."""
        mock_platform.return_value = "Windows"
        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args="uv run test", returncode=0
        )

        command = ["uv", "run", "test"]
        run_command(command)

        mock_subprocess_run.assert_called_once()
        call_args = mock_subprocess_run.call_args
        
        assert call_args[1]["shell"] is True
        assert isinstance(call_args[0][0], str)
        assert "uv run test" in call_args[0][0]

    @mock.patch("platform.system")
    @mock.patch("subprocess.run")
    def test_unix_uses_shell_false(self, mock_subprocess_run, mock_platform):
        """Test that Unix-like systems use shell=False with list commands."""
        mock_platform.return_value = "Linux"
        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args=["uv", "run", "test"], returncode=0
        )

        command = ["uv", "run", "test"]
        run_command(command)

        mock_subprocess_run.assert_called_once()
        call_args = mock_subprocess_run.call_args
        
        assert call_args[1].get("shell", False) is False
        assert call_args[0][0] == command

    @mock.patch("platform.system")
    @mock.patch("subprocess.run")
    def test_windows_command_escaping(self, mock_subprocess_run, mock_platform):
        """Test that Windows properly escapes command arguments."""
        mock_platform.return_value = "Windows"
        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args="test", returncode=0
        )

        command = ["echo", "hello world", "test&special"]
        run_command(command)

        mock_subprocess_run.assert_called_once()
        call_args = mock_subprocess_run.call_args
        command_str = call_args[0][0]
        
        assert '"hello world"' in command_str or "'hello world'" in command_str

    @mock.patch("platform.system")
    @mock.patch("subprocess.run")
    def test_error_handling_preserved(self, mock_subprocess_run, mock_platform):
        """Test that CalledProcessError is properly raised."""
        mock_platform.return_value = "Windows"
        mock_subprocess_run.side_effect = subprocess.CalledProcessError(1, "test")

        with pytest.raises(subprocess.CalledProcessError):
            run_command(["test"], check=True)

    @mock.patch("platform.system")
    @mock.patch("subprocess.run")
    def test_all_parameters_passed_through(self, mock_subprocess_run, mock_platform):
        """Test that all subprocess parameters are properly passed through."""
        mock_platform.return_value = "Linux"
        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args=["test"], returncode=0
        )

        run_command(
            ["test"],
            capture_output=True,
            text=False,
            check=False,
            cwd="/tmp",
            env={"TEST": "value"},
            timeout=30
        )

        mock_subprocess_run.assert_called_once()
        call_args = mock_subprocess_run.call_args
        
        assert call_args[1]["capture_output"] is True
        assert call_args[1]["text"] is False
        assert call_args[1]["check"] is False
        assert call_args[1]["cwd"] == "/tmp"
        assert call_args[1]["env"] == {"TEST": "value"}
        assert call_args[1]["timeout"] == 30

    @mock.patch("platform.system")
    @mock.patch("subprocess.run")
    def test_macos_uses_shell_false(self, mock_subprocess_run, mock_platform):
        """Test that macOS uses shell=False with list commands."""
        mock_platform.return_value = "Darwin"
        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args=["uv", "run", "test"], returncode=0
        )

        command = ["uv", "run", "test"]
        run_command(command)

        mock_subprocess_run.assert_called_once()
        call_args = mock_subprocess_run.call_args
        
        assert call_args[1].get("shell", False) is False
        assert call_args[0][0] == command

    @mock.patch("platform.system")
    @mock.patch("subprocess.run")
    def test_windows_string_command_passthrough(self, mock_subprocess_run, mock_platform):
        """Test that Windows passes through string commands unchanged."""
        mock_platform.return_value = "Windows"
        mock_subprocess_run.return_value = subprocess.CompletedProcess(
            args="test command", returncode=0
        )

        command_str = "test command with spaces"
        run_command(command_str)

        mock_subprocess_run.assert_called_once()
        call_args = mock_subprocess_run.call_args
        
        assert call_args[0][0] == command_str
        assert call_args[1]["shell"] is True
