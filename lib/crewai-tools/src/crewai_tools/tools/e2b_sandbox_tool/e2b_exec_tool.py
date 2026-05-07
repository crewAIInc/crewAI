from __future__ import annotations

from builtins import type as type_
from typing import Any

from pydantic import BaseModel, Field

from crewai_tools.tools.e2b_sandbox_tool.e2b_base_tool import E2BBaseTool


class E2BExecToolSchema(BaseModel):
    command: str = Field(..., description="Shell command to execute in the sandbox.")
    cwd: str | None = Field(
        default=None,
        description="Working directory to run the command in. Defaults to the sandbox home dir.",
    )
    envs: dict[str, str] | None = Field(
        default=None,
        description="Optional environment variables to set for this command.",
    )
    timeout: float | None = Field(
        default=None,
        description="Maximum seconds to wait for the command to finish.",
    )


class E2BExecTool(E2BBaseTool):
    """Run a shell command inside an E2B sandbox."""

    name: str = "E2B Sandbox Exec"
    description: str = (
        "Execute a shell command inside an E2B sandbox and return the exit "
        "code, stdout, and stderr. Use this to run builds, package installs, "
        "git operations, or any one-off shell command."
    )
    args_schema: type_[BaseModel] = E2BExecToolSchema

    def _run(
        self,
        command: str,
        cwd: str | None = None,
        envs: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        sandbox, should_kill = self._acquire_sandbox()
        try:
            run_kwargs: dict[str, Any] = {}
            if cwd is not None:
                run_kwargs["cwd"] = cwd
            if envs is not None:
                run_kwargs["envs"] = envs
            if timeout is not None:
                run_kwargs["timeout"] = timeout
            result = sandbox.commands.run(command, **run_kwargs)
            return {
                "exit_code": getattr(result, "exit_code", None),
                "stdout": getattr(result, "stdout", None),
                "stderr": getattr(result, "stderr", None),
                "error": getattr(result, "error", None),
            }
        finally:
            self._release_sandbox(sandbox, should_kill)
