from __future__ import annotations

from builtins import type as type_
from typing import Any

from pydantic import BaseModel, Field

from crewai_tools.tools.boxlite_sandbox_tool.boxlite_base_tool import BoxLiteBaseTool


class BoxLiteExecToolSchema(BaseModel):
    command: str = Field(..., description="Shell command to execute in the box.")
    cwd: str | None = Field(
        default=None,
        description="Working directory to run the command in. Defaults to the box's configured workdir.",
    )
    envs: dict[str, str] | None = Field(
        default=None,
        description="Optional environment variables to set for this command.",
    )
    timeout: float | None = Field(
        default=None,
        description="Maximum seconds to wait for the command to finish.",
    )


class BoxLiteExecTool(BoxLiteBaseTool):
    """Run a shell command inside a BoxLite micro-VM."""

    name: str = "BoxLite Sandbox Exec"
    description: str = (
        "Execute a shell command inside a BoxLite micro-VM and return the exit "
        "code, stdout, and stderr. Use this to run builds, package installs, "
        "git operations, or any one-off shell command."
    )
    args_schema: type_[BaseModel] = BoxLiteExecToolSchema

    def _run(
        self,
        command: str,
        cwd: str | None = None,
        envs: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        box, should_remove = self._acquire_box()
        try:
            result = box.exec(
                "/bin/sh", "-c", command, env=envs, cwd=cwd, timeout=timeout
            )
            return self._result_dict(result)
        finally:
            self._release_box(box, should_remove)
