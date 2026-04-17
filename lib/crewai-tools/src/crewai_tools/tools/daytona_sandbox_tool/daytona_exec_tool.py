from __future__ import annotations

from builtins import type as type_
from typing import Any

from pydantic import BaseModel, Field

from crewai_tools.tools.daytona_sandbox_tool.daytona_base_tool import DaytonaBaseTool


class DaytonaExecToolSchema(BaseModel):
    command: str = Field(..., description="Shell command to execute in the sandbox.")
    cwd: str | None = Field(
        default=None,
        description="Working directory to run the command in. Defaults to the sandbox work dir.",
    )
    env: dict[str, str] | None = Field(
        default=None,
        description="Optional environment variables to set for this command.",
    )
    timeout: int | None = Field(
        default=None,
        description="Maximum seconds to wait for the command to finish.",
    )


class DaytonaExecTool(DaytonaBaseTool):
    """Run a shell command inside a Daytona sandbox."""

    name: str = "Daytona Sandbox Exec"
    description: str = (
        "Execute a shell command inside a Daytona sandbox and return the exit "
        "code and combined output. Use this to run builds, package installs, "
        "git operations, or any one-off shell command."
    )
    args_schema: type_[BaseModel] = DaytonaExecToolSchema

    def _run(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> Any:
        sandbox, should_delete = self._acquire_sandbox()
        try:
            response = sandbox.process.exec(
                command,
                cwd=cwd,
                env=env,
                timeout=timeout,
            )
            return {
                "exit_code": getattr(response, "exit_code", None),
                "result": getattr(response, "result", None),
                "artifacts": getattr(response, "artifacts", None),
            }
        finally:
            self._release_sandbox(sandbox, should_delete)
