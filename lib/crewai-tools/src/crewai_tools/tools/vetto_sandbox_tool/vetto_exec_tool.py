from __future__ import annotations

import os
import platform
from builtins import type as type_
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from crewai_tools.tools.vetto_sandbox_tool.vetto_base_tool import VettoBaseTool


class VettoExecToolSchema(BaseModel):
    command: str = Field(
        ...,
        description="Shell command to execute within the sandbox.",
    )
    cwd: str | None = Field(
        default=None,
        description="Working directory to run the command in. Defaults to the configured workspace root.",
    )
    env: dict[str, str] | None = Field(
        default=None,
        description="Optional environment variables to set for the process.",
    )
    timeout: int | None = Field(
        default=None,
        description="Maximum seconds to wait for command execution.",
    )


class VettoExecTool(VettoBaseTool):
    """Run shell commands inside a kernel-level Vetto sandbox."""

    name: str = "Vetto Sandbox Exec"
    description: str = (
        "Execute a shell command inside a kernel-level Vetto sandbox with "
        "Landlock LSM / macOS Seatbelt isolation and zero container overhead. "
        "Returns exit code, stdout, and stderr with fail-closed timeout handling."
    )
    args_schema: type_[BaseModel] = VettoExecToolSchema

    def _run(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> Any:
        """Execute a shell command inside the sandbox.

        Args:
            command: Shell command string to execute.
            cwd: Optional working directory for the command.
            env: Optional environment variables.
            timeout: Optional per-command timeout in seconds.

        Returns:
            Dictionary containing exit_code, stdout, stderr, timed_out flag, and elapsed_seconds.
        """
        sandbox_root = Path(self.working_dir or os.getcwd()).resolve()
        if cwd:
            try:
                resolved_cwd = (sandbox_root / cwd).resolve(strict=False) if not os.path.isabs(cwd) else Path(cwd).resolve(strict=False)
                resolved_cwd.relative_to(sandbox_root)
            except (ValueError, RuntimeError):
                raise PermissionError(
                    f"Execution cwd {cwd} escapes configured workspace boundary {sandbox_root}"
                )
            effective_cwd = str(resolved_cwd)
        else:
            effective_cwd = str(sandbox_root)

        if platform.system() == "Windows":
            shell_cmd = ["cmd.exe", "/c", command]
        else:
            shell_cmd = ["sh", "-c", command]

        return self._execute_subprocess(
            shell_cmd,
            cwd=effective_cwd,
            env=env,
            timeout=timeout,
        )
