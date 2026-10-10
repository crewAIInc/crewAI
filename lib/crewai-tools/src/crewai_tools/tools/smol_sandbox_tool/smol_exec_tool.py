from __future__ import annotations

from builtins import type as type_

from pydantic import BaseModel, Field

from crewai_tools.tools.smol_sandbox_tool.smol_base_tool import (
    SmolBaseTool,
    execution_result,
)


class SmolExecToolSchema(BaseModel):
    command: str = Field(..., description="Shell command to execute inside the VM.")
    cwd: str | None = None
    env: dict[str, str] | None = None
    timeout: int | None = Field(default=None, gt=0)


class SmolExecTool(SmolBaseTool):
    """Run shell commands inside a Smol Machines microVM."""

    name: str = "Smol Machines Sandbox Exec"
    description: str = "Run a shell command in an isolated local or cloud VM."
    args_schema: type_[BaseModel] = SmolExecToolSchema

    def _run(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> dict[str, object]:
        sdk = self._sdk()
        with self._machine_session() as machine:
            return execution_result(
                machine.exec(
                    ["sh", "-lc", command],
                    sdk.ExecOptions(env=env, workdir=cwd, timeout=timeout),
                )
            )
