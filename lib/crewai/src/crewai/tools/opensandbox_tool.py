"""OpenSandbox tool for CrewAI agents.

OpenSandbox (https://open-sandbox.ai) is a self-hosted sandbox platform
for running shell commands and managing files inside isolated containers.
This tool exposes its core operations to CrewAI agents through a single
``OpenSandboxTool`` that lazily creates one sandbox per tool instance and
reuses it across calls until ``kill`` is invoked.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from datetime import timedelta
import os
from typing import Any, Literal

from pydantic import BaseModel, Field, PrivateAttr

from crewai.tools.base_tool import BaseTool, EnvVar


class OpenSandboxToolSchema(BaseModel):
    """Arguments accepted by ``OpenSandboxTool``."""

    action: Literal["run_command", "read_file", "write_file", "kill"] = Field(
        description=(
            "Operation to perform: run_command (execute shell command), "
            "read_file (read file contents), write_file (write file contents), "
            "or kill (terminate the sandbox)."
        ),
    )
    command: str | None = Field(
        default=None,
        description="Shell command to execute. Required when action is 'run_command'.",
    )
    path: str | None = Field(
        default=None,
        description="Absolute file path. Required for 'read_file' and 'write_file'.",
    )
    content: str | None = Field(
        default=None,
        description="File content to write. Required when action is 'write_file'.",
    )


class OpenSandboxTool(BaseTool):
    """Run shell commands and manage files inside an OpenSandbox sandbox."""

    name: str = "OpenSandbox"
    description: str = (
        "Execute commands and manage files in an isolated OpenSandbox container. "
        "Useful for running untrusted code, scripting, file I/O, or any work that "
        "should be isolated from the host. The same sandbox is reused across "
        "calls; invoke action='kill' to release it."
    )
    args_schema: type[BaseModel] = OpenSandboxToolSchema
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="OPENSANDBOX_DOMAIN",
                description="Host:port of the OpenSandbox server (e.g. 'localhost:8080').",
                required=True,
            ),
            EnvVar(
                name="OPENSANDBOX_PROTOCOL",
                description="Protocol used to reach the server: 'http' or 'https'.",
                required=False,
                default="http",
            ),
            EnvVar(
                name="OPENSANDBOX_IMAGE",
                description="Container image to launch (e.g. 'python:3.12').",
                required=False,
                default="python:3.12",
            ),
            EnvVar(
                name="OPENSANDBOX_TIMEOUT_MINUTES",
                description="Sandbox idle timeout in minutes before auto-shutdown.",
                required=False,
                default="30",
            ),
            EnvVar(
                name="OPENSANDBOX_API_KEY",
                description="Optional API key if the OpenSandbox server requires auth.",
                required=False,
                default=None,
            ),
        ]
    )

    _sandbox: Any = PrivateAttr(default=None)

    def _run(self, **kwargs: Any) -> str:
        action = kwargs.get("action")
        command = kwargs.get("command")
        path = kwargs.get("path")
        content = kwargs.get("content")

        if action == "kill":
            return self._run_async(self._kill())
        if action == "run_command":
            if not command:
                return "Error: 'command' is required when action='run_command'."
            return self._run_async(self._run_command(command))
        if action == "read_file":
            if not path:
                return "Error: 'path' is required when action='read_file'."
            return self._run_async(self._read_file(path))
        if action == "write_file":
            if not path:
                return "Error: 'path' is required when action='write_file'."
            if content is None:
                return "Error: 'content' is required when action='write_file'."
            return self._run_async(self._write_file(path, content))
        return f"Error: unknown action '{action}'."

    @staticmethod
    def _run_async(coro: Any) -> str:
        """Run ``coro`` to completion from a sync context, regardless of loop state."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(asyncio.run, coro).result()

    def _build_connection_config(self) -> Any:
        from opensandbox.config.connection import ConnectionConfig

        domain = (os.getenv("OPENSANDBOX_DOMAIN") or "").strip()
        if not domain:
            raise ValueError(
                "OPENSANDBOX_DOMAIN is not set. Configure it to point at your "
                "OpenSandbox server (e.g. 'localhost:8080')."
            )
        protocol = (os.getenv("OPENSANDBOX_PROTOCOL") or "http").strip()
        api_key = os.getenv("OPENSANDBOX_API_KEY") or None
        return ConnectionConfig(domain=domain, protocol=protocol, api_key=api_key)

    async def _ensure_sandbox(self) -> Any:
        if self._sandbox is not None:
            return self._sandbox
        from opensandbox import Sandbox

        image = (os.getenv("OPENSANDBOX_IMAGE") or "python:3.12").strip()
        timeout_minutes = int(os.getenv("OPENSANDBOX_TIMEOUT_MINUTES") or "30")
        connection_config = self._build_connection_config()
        self._sandbox = await Sandbox.create(
            image,
            timeout=timedelta(minutes=timeout_minutes),
            connection_config=connection_config,
        )
        return self._sandbox

    async def _run_command(self, command: str) -> str:
        try:
            sandbox = await self._ensure_sandbox()
            execution = await sandbox.commands.run(command)
        except Exception as exc:
            return f"OpenSandbox error running command: {exc}"

        stdout = "".join(
            getattr(item, "text", "") for item in (execution.logs.stdout or [])
        )
        stderr = "".join(
            getattr(item, "text", "") for item in (execution.logs.stderr or [])
        )
        parts: list[str] = []
        if stdout:
            parts.append(stdout)
        if stderr:
            parts.append(f"stderr:\n{stderr}")
        if getattr(execution, "error", None):
            parts.append(f"error: {execution.error}")
        return "\n".join(parts).strip() or "(no output)"

    async def _read_file(self, path: str) -> str:
        try:
            sandbox = await self._ensure_sandbox()
            return await sandbox.files.read_file(path)
        except Exception as exc:
            return f"OpenSandbox error reading {path}: {exc}"

    async def _write_file(self, path: str, content: str) -> str:
        from opensandbox.models import WriteEntry

        try:
            sandbox = await self._ensure_sandbox()
            await sandbox.files.write_files(
                [WriteEntry(path=path, data=content, mode=0o644)]
            )
        except Exception as exc:
            return f"OpenSandbox error writing {path}: {exc}"
        return f"Wrote {len(content)} bytes to {path}."

    async def _kill(self) -> str:
        if self._sandbox is None:
            return "No sandbox to kill."
        try:
            await self._sandbox.kill()
        except Exception as exc:
            self._sandbox = None
            return f"OpenSandbox error during kill: {exc}"
        self._sandbox = None
        return "Sandbox killed."
