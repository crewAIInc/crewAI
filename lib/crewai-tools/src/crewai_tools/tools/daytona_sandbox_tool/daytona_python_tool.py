from __future__ import annotations

from builtins import type as type_
from typing import Any

from pydantic import BaseModel, Field

from crewai_tools.tools.daytona_sandbox_tool.daytona_base_tool import DaytonaBaseTool


class DaytonaPythonToolSchema(BaseModel):
    code: str = Field(
        ...,
        description="Python source to execute inside the sandbox.",
    )
    argv: list[str] | None = Field(
        default=None,
        description="Optional argv passed to the script (forwarded as params.argv).",
    )
    env: dict[str, str] | None = Field(
        default=None,
        description="Optional environment variables for the run (forwarded as params.env).",
    )
    timeout: int | None = Field(
        default=None,
        description="Maximum seconds to wait for the code to finish.",
    )


class DaytonaPythonTool(DaytonaBaseTool):
    """Run Python source inside a Daytona sandbox."""

    name: str = "Daytona Sandbox Python"
    description: str = (
        "Execute a block of Python code inside a Daytona sandbox and return the "
        "exit code, captured stdout, and any produced artifacts. Use this for "
        "data processing, quick scripts, or analysis that should run in an "
        "isolated environment."
    )
    args_schema: type_[BaseModel] = DaytonaPythonToolSchema

    def _run(
        self,
        code: str,
        argv: list[str] | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> Any:
        sandbox, should_delete = self._acquire_sandbox()
        try:
            params = self._build_code_run_params(argv=argv, env=env)
            response = sandbox.process.code_run(code, params=params, timeout=timeout)
            return {
                "exit_code": getattr(response, "exit_code", None),
                "result": getattr(response, "result", None),
                "artifacts": getattr(response, "artifacts", None),
            }
        finally:
            self._release_sandbox(sandbox, should_delete)

    def _build_code_run_params(
        self,
        argv: list[str] | None,
        env: dict[str, str] | None,
    ) -> Any | None:
        if argv is None and env is None:
            return None
        try:
            from daytona import CodeRunParams
        except ImportError as exc:
            raise ImportError(
                "Could not import daytona.CodeRunParams while building "
                "argv/env for sandbox.process.code_run. This usually means the "
                "installed 'daytona' SDK is too old or incompatible. Upgrade "
                "with: pip install -U 'crewai-tools[daytona]'"
            ) from exc
        kwargs: dict[str, Any] = {}
        if argv is not None:
            kwargs["argv"] = argv
        if env is not None:
            kwargs["env"] = env
        return CodeRunParams(**kwargs)
