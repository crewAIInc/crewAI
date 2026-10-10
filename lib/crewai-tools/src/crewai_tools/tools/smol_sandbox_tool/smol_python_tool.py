from __future__ import annotations

from builtins import type as type_

from pydantic import BaseModel, Field

from crewai_tools.tools.smol_sandbox_tool.smol_base_tool import (
    SmolBaseTool,
    execution_result,
)


class SmolPythonToolSchema(BaseModel):
    code: str = Field(..., description="Python source to execute inside the VM.")
    argv: list[str] | None = None
    env: dict[str, str] | None = None
    timeout: int | None = Field(default=None, gt=0)


class SmolPythonTool(SmolBaseTool):
    """Execute Python code inside a Smol Machines microVM."""

    name: str = "Smol Machines Sandbox Python"
    description: str = "Run Python code in an isolated local or cloud VM."
    args_schema: type_[BaseModel] = SmolPythonToolSchema

    def _run(
        self,
        code: str,
        argv: list[str] | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> dict[str, object]:
        sdk = self._sdk()
        with self._machine_session() as machine:
            return execution_result(
                machine.exec(
                    ["python", "-c", code, *(argv or [])],
                    sdk.ExecOptions(env=env, timeout=timeout),
                )
            )
