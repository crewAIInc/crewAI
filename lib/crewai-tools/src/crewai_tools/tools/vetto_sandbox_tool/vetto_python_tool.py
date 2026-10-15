from __future__ import annotations

import os
import sys
import tempfile
import uuid
from builtins import type as type_
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from crewai_tools.tools.vetto_sandbox_tool.vetto_base_tool import VettoBaseTool


class VettoPythonToolSchema(BaseModel):
    code: str = Field(
        ...,
        description="Python source code to execute inside the sandbox.",
    )
    argv: list[str] | None = Field(
        default=None,
        description="Optional command line arguments passed to sys.argv.",
    )
    env: dict[str, str] | None = Field(
        default=None,
        description="Optional environment variables for the execution context.",
    )
    timeout: int | None = Field(
        default=None,
        description="Maximum seconds to wait for script execution.",
    )


class VettoPythonTool(VettoBaseTool):
    """Run Python code inside a kernel-level Vetto sandbox."""

    name: str = "Vetto Sandbox Python"
    description: str = (
        "Execute Python code inside a kernel-level Vetto sandbox with "
        "filesystem isolation and network enforcement. Returns exit code, "
        "stdout, and stderr."
    )
    args_schema: type_[BaseModel] = VettoPythonToolSchema

    def _run(
        self,
        code: str,
        argv: list[str] | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> Any:
        """Execute a block of Python source code inside the sandbox.

        Args:
            code: Python code string to execute.
            argv: Optional arguments passed to sys.argv.
            env: Optional environment variables.
            timeout: Optional per-execution timeout in seconds.

        Returns:
            Dictionary containing exit_code, stdout, stderr, timed_out flag, and elapsed_seconds.
        """
        effective_cwd = self.working_dir or os.getcwd()
        os.makedirs(effective_cwd, exist_ok=True)

        fd, temp_file_path = tempfile.mkstemp(
            prefix=".vetto_script_",
            suffix=".py",
            dir=effective_cwd,
        )
        script_path = Path(temp_file_path)

        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(code)

            cmd = [sys.executable, str(script_path)]
            if argv:
                cmd.extend(argv)

            return self._execute_subprocess(
                cmd,
                cwd=effective_cwd,
                env=env,
                timeout=timeout,
            )
        finally:
            if script_path.exists():
                try:
                    script_path.unlink()
                except OSError:
                    pass
