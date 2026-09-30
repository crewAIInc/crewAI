from __future__ import annotations

import logging
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import ConfigDict, Field


logger = logging.getLogger(__name__)


class VettoBaseTool(BaseTool):
    """Shared base for tools executing inside a Vetto sandbox.

    Vetto provides daemon-less, rootless kernel sandbox boundaries
    (Landlock LSM ABI 1-6, namespaces, cgroups v2, macOS Seatbelt,
    Windows LPAC) with sub-4ms startup overhead.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    working_dir: str | None = Field(
        default=None,
        description="Root workspace directory for filesystem sandbox containment.",
    )
    net: str = Field(
        default="off",
        description="Network isolation mode: 'off' (full isolation), 'allowlist', or 'host'.",
    )
    allowed_domains: list[str] | None = Field(
        default=None,
        description="List of permitted domain names when net='allowlist'.",
    )
    allow_write: list[str] | None = Field(
        default=None,
        description="Additional file or directory paths permitted for write access.",
    )
    allow_read: list[str] | None = Field(
        default=None,
        description="Additional file or directory paths permitted for read-only access.",
    )
    timeout: int | None = Field(
        default=120,
        description="Default execution timeout in seconds.",
    )
    memory_limit: str | None = Field(
        default=None,
        description="Optional cgroups memory limit ceiling (e.g. '512MB', '1GB').",
    )
    vetto_binary: str | None = Field(
        default=None,
        description="Explicit path to the vetto binary. Defaults to autodetecting in PATH.",
    )
    allow_fallback: bool = Field(
        default=False,
        description="If True, fall back to process-group isolated execution when vetto is not found.",
    )

    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="VETTO_PATH",
                description="Optional path override for the vetto sandbox executable",
                required=False,
            ),
        ]
    )

    def _resolve_vetto_binary(self) -> str | None:
        """Resolve path to the vetto binary or None if not installed.

        Returns:
            Resolved executable path string or None if not found.
        """
        if self.vetto_binary and os.path.isfile(self.vetto_binary) and os.access(self.vetto_binary, os.X_OK):
            return self.vetto_binary

        env_path = os.getenv("VETTO_PATH")
        if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
            return env_path

        found = shutil.which("vetto")
        if found:
            return found

        candidates = [
            os.path.expanduser("~/.cargo/bin/vetto"),
            "/usr/local/bin/vetto",
            "/usr/bin/vetto",
        ]
        for path in candidates:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path

        return None

    def _build_command(
        self,
        command_args: list[str],
        cwd: str | None = None,
        timeout: int | None = None,
    ) -> list[str]:
        """Construct the sandboxed execution command.

        Args:
            command_args: Argument vector to run inside the sandbox.
            cwd: Working directory path for the process.
            timeout: Execution timeout in seconds.

        Returns:
            Final command argument vector prefixed with vetto CLI parameters.

        Raises:
            RuntimeError: If vetto binary is missing and allow_fallback is False.
        """
        vetto_bin = self._resolve_vetto_binary()
        if not vetto_bin:
            if not self.allow_fallback:
                raise RuntimeError(
                    "Vetto binary not found. Install via 'cargo install vetto' "
                    "or 'npm install -g @shledery/vetto', set VETTO_PATH, or "
                    "configure allow_fallback=True."
                )
            return command_args

        effective_cwd = cwd or self.working_dir or os.getcwd()
        resolved_cwd = str(Path(effective_cwd).resolve())

        run_args = [vetto_bin, "run", f"--net={self.net}"]

        effective_timeout = timeout if timeout is not None else self.timeout
        if effective_timeout:
            run_args.extend(["--timeout", str(effective_timeout)])

        if self.memory_limit:
            run_args.extend(["--memory", self.memory_limit])

        # Ensure cwd is permitted for read/write
        run_args.extend(["--allow-write", resolved_cwd])

        if self.allow_write:
            for p in self.allow_write:
                run_args.extend(["--allow-write", str(Path(p).resolve())])

        if self.allow_read:
            for p in self.allow_read:
                run_args.extend(["--allow-read", str(Path(p).resolve())])

        if self.net == "allowlist" and self.allowed_domains:
            for domain in self.allowed_domains:
                run_args.extend(["--allow-domain", domain])

        run_args.append("--")
        run_args.extend(command_args)
        return run_args

    def _execute_subprocess(
        self,
        command_args: list[str],
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> dict[str, Any]:
        """Execute command within sandbox process boundary.

        Args:
            command_args: Command and arguments to execute.
            cwd: Working directory for process execution.
            env: Optional environment variables dictionary.
            timeout: Execution timeout in seconds.

        Returns:
            Dictionary with exit_code, stdout, stderr, timed_out flag, and elapsed_seconds.
        """
        full_command = self._build_command(command_args, cwd=cwd, timeout=timeout)
        effective_cwd = cwd or self.working_dir or os.getcwd()
        resolved_cwd = str(Path(effective_cwd).resolve())

        exec_env = os.environ.copy()
        if env:
            exec_env.update(env)

        effective_timeout = timeout if timeout is not None else self.timeout

        kwargs: dict[str, Any] = {
            "cwd": resolved_cwd,
            "env": exec_env,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
        }

        # On Unix, run in separate process group for clean signal cleanup
        if hasattr(os, "setsid"):
            kwargs["preexec_fn"] = os.setsid

        start_time = time.monotonic()
        try:
            proc = subprocess.Popen(full_command, **kwargs)
            try:
                stdout_bytes, stderr_bytes = proc.communicate(timeout=effective_timeout)
                elapsed = time.monotonic() - start_time
                return {
                    "exit_code": proc.returncode,
                    "stdout": stdout_bytes.decode("utf-8", errors="replace"),
                    "stderr": stderr_bytes.decode("utf-8", errors="replace"),
                    "timed_out": False,
                    "elapsed_seconds": round(elapsed, 4),
                }
            except subprocess.TimeoutExpired:
                # Terminate entire process group
                if hasattr(os, "killpg") and hasattr(os, "getpgid"):
                    try:
                        pgid = os.getpgid(proc.pid)
                        os.killpg(pgid, signal.SIGKILL)
                    except OSError:
                        proc.kill()
                else:
                    proc.kill()

                stdout_bytes, stderr_bytes = proc.communicate()
                return {
                    "exit_code": 124,
                    "stdout": stdout_bytes.decode("utf-8", errors="replace") if stdout_bytes else "",
                    "stderr": stderr_bytes.decode("utf-8", errors="replace") if stderr_bytes else "Command timed out",
                    "timed_out": True,
                    "elapsed_seconds": round(time.monotonic() - start_time, 4),
                }
        except Exception as e:
            logger.error("Failed to execute sandboxed command: %s", e)
            return {
                "exit_code": 125,
                "stdout": "",
                "stderr": str(e),
                "timed_out": False,
                "elapsed_seconds": 0.0,
            }
