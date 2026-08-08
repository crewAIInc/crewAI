from __future__ import annotations

from builtins import type as type_
from datetime import timedelta
from typing import Any

from pydantic import BaseModel, Field

from crewai_tools.tools.open_sandbox_tool.open_sandbox_base_tool import (
    OpenSandboxBaseTool,
)


class OpenSandboxExecToolSchema(BaseModel):
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


class OpenSandboxExecTool(OpenSandboxBaseTool):
    """Run a shell command inside an Open Sandbox sandbox."""

    name: str = "Open Sandbox Exec"
    description: str = (
        "Execute a shell command inside an Open Sandbox sandbox and return the "
        "exit code and combined output. Use this to run builds, package installs, "
        "git operations, or any one-off shell command."
    )
    args_schema: type_[BaseModel] = OpenSandboxExecToolSchema

    def _run(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout: int | None = None,
    ) -> Any:
        sdk = self._import_sdk()
        sandbox, should_kill = self._acquire_sandbox()
        try:
            opts = self._build_run_opts(sdk, cwd=cwd, env=env, timeout=timeout)
            execution = sandbox.commands.run(command, opts=opts)
            return {
                "exit_code": getattr(execution, "exit_code", None),
                "result": getattr(execution, "text", None),
                "artifacts": _collect_artifacts(execution),
            }
        finally:
            self._release_sandbox(sandbox, should_kill)

    @staticmethod
    def _build_run_opts(
        sdk: dict[str, Any],
        *,
        cwd: str | None,
        env: dict[str, str] | None,
        timeout: int | None,
    ) -> Any | None:
        if cwd is None and env is None and timeout is None:
            return None
        kwargs: dict[str, Any] = {}
        if cwd is not None:
            kwargs["working_directory"] = cwd
        if env is not None:
            kwargs["envs"] = env
        if timeout is not None:
            kwargs["timeout"] = timedelta(seconds=timeout)
        return sdk["RunCommandOpts"](**kwargs)


def _collect_artifacts(execution: Any) -> dict[str, Any] | None:
    logs = getattr(execution, "logs", None)
    stderr_msgs = getattr(logs, "stderr", None) if logs is not None else None
    results = getattr(execution, "result", None)
    error = getattr(execution, "error", None)
    if not stderr_msgs and not results and error is None:
        return None
    return {
        "stderr": [getattr(m, "text", str(m)) for m in stderr_msgs or []],
        "results": [getattr(r, "text", None) for r in results or []],
        "error": _serialize_error(error),
    }


def _serialize_error(error: Any) -> dict[str, Any] | None:
    if error is None:
        return None
    return {
        "name": getattr(error, "name", None),
        "value": getattr(error, "value", None),
        "traceback": getattr(error, "traceback", None),
    }
