"""OpenSandbox tool for secure Python code execution in isolated containers."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from pydantic import BaseModel, Field, PrivateAttr

from crewai.tools import BaseTool


class OpenSandboxToolSchema(BaseModel):
    """Input schema for OpenSandboxTool."""

    code: str = Field(
        ..., description="Python3 code to execute in the sandbox"
    )
    libraries: list[str] = Field(
        default_factory=list,
        description="Python libraries to install before execution (e.g. ['pandas', 'numpy'])",
    )


class OpenSandboxTool(BaseTool):
    """Securely execute Python code in an isolated OpenSandbox container.

    This tool creates a sandboxed environment using OpenSandbox to safely run
    Python code with full isolation. The sandbox persists across multiple
    invocations, allowing agents to build up state (variables, files, installed
    packages) over the course of a task.

    Requires a running OpenSandbox server. Install dependencies with:
        pip install opensandbox opensandbox-code-interpreter
    """

    name: str = "Open Sandbox Code Interpreter"
    description: str = (
        "Execute Python code securely in an isolated OpenSandbox container. "
        "Use this tool to run Python scripts, install packages, and process data "
        "in a safe sandboxed environment. Variables and state persist across calls."
    )
    args_schema: type[BaseModel] = OpenSandboxToolSchema

    opensandbox_api_key: str | None = Field(
        default=None,
        description="OpenSandbox API key. Falls back to OPENSANDBOX_API_KEY env var.",
    )
    opensandbox_domain: str = Field(
        default="localhost:8080",
        description="OpenSandbox server address.",
    )
    opensandbox_protocol: str = Field(
        default="http",
        description="Protocol for OpenSandbox server (http or https).",
    )
    opensandbox_image: str = Field(
        default="opensandbox/code-interpreter:v1.0.2",
        description="Docker image for the sandbox container.",
    )
    timeout: int = Field(
        default=300,
        description="Sandbox lifetime in seconds.",
    )
    resource: dict[str, str] | None = Field(
        default=None,
        description='Resource limits, e.g. {"cpu": "1", "memory": "2Gi"}.',
    )

    _sandbox: Any = PrivateAttr(default=None)
    _interpreter: Any = PrivateAttr(default=None)
    _installed_libraries: set[str] = PrivateAttr(default_factory=set)
    _loop: Any = PrivateAttr(default=None)

    def _get_api_key(self) -> str | None:
        return self.opensandbox_api_key or os.environ.get("OPENSANDBOX_API_KEY")

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Return a persistent event loop for all sync invocations.

        Using a single loop ensures that async objects created in one call
        (sandbox, interpreter) remain valid for subsequent calls.
        """
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def _run(self, code: str, libraries: list[str] | None = None, **kwargs: Any) -> str:
        try:
            # Check if we're already inside a running event loop
            asyncio.get_running_loop()
            # If so, we cannot use run_until_complete; fall back to asyncio.run
            # in a new thread, but this path loses sandbox persistence.
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    self._get_loop().run_until_complete,
                    self._arun(code=code, libraries=libraries, **kwargs),
                )
                return future.result()
        except RuntimeError:
            # No running loop — safe to use our persistent loop directly
            return self._get_loop().run_until_complete(
                self._arun(code=code, libraries=libraries, **kwargs)
            )

    async def _arun(self, code: str, libraries: list[str] | None = None, **kwargs: Any) -> str:
        try:
            from code_interpreter import CodeInterpreter  # type: ignore[import-untyped]
            from code_interpreter.models.code import SupportedLanguage  # type: ignore[import-untyped]
            from opensandbox import Sandbox  # type: ignore[import-untyped]
            from opensandbox.config import ConnectionConfig  # type: ignore[import-untyped]
        except ImportError as exc:
            return (
                f"Error: Missing required packages. Install with:\n"
                f"  pip install opensandbox opensandbox-code-interpreter\n"
                f"Details: {exc}"
            )

        libraries = libraries or []

        try:
            # Create sandbox and interpreter on first call
            if self._sandbox is None or self._interpreter is None:
                await self._create_sandbox(Sandbox, CodeInterpreter, ConnectionConfig)

            # Install new libraries
            new_libs = [lib for lib in libraries if lib not in self._installed_libraries]
            if new_libs:
                install_result = await self._install_libraries(new_libs)
                if install_result:
                    return install_result

            # Execute code
            execution = await self._interpreter.codes.run(
                code, language=SupportedLanguage.PYTHON
            )

            return self._format_execution_result(execution)

        except Exception as e:
            error_type = type(e).__name__
            return f"Error ({error_type}): {e}"

    async def _create_sandbox(
        self, sandbox_cls: type, interpreter_cls: type, config_cls: type
    ) -> None:
        from datetime import timedelta

        api_key = self._get_api_key()
        config_kwargs: dict[str, Any] = {
            "domain": self.opensandbox_domain,
            "protocol": self.opensandbox_protocol,
        }
        if api_key:
            config_kwargs["api_key"] = api_key

        connection_config = config_cls(**config_kwargs)

        create_kwargs: dict[str, Any] = {
            "timeout": timedelta(seconds=self.timeout),
            "connection_config": connection_config,
        }
        if self.resource:
            create_kwargs["resource"] = self.resource

        self._sandbox = await sandbox_cls.create(
            self.opensandbox_image, **create_kwargs
        )
        self._interpreter = await interpreter_cls.create(sandbox=self._sandbox)
        self._installed_libraries = set()

    async def _install_libraries(self, libraries: list[str]) -> str | None:
        """Install libraries and return error string if failed, None on success."""
        libs_str = " ".join(libraries)
        try:
            result = await self._sandbox.commands.run(f"pip install {libs_str}")
            # Check for errors in stderr
            stderr_text = "".join(msg.text for msg in result.logs.stderr)
            if result.error:
                return f"Error installing libraries [{libs_str}]: {stderr_text}"
            self._installed_libraries.update(libraries)
            return None
        except Exception as e:
            return f"Error installing libraries [{libs_str}]: {e}"

    def _format_execution_result(self, execution: Any) -> str:
        parts: list[str] = []

        # Stdout
        stdout_text = "".join(msg.text for msg in execution.logs.stdout)
        if stdout_text:
            parts.append(f"[stdout]\n{stdout_text}")

        # Stderr
        stderr_text = "".join(msg.text for msg in execution.logs.stderr)
        if stderr_text:
            parts.append(f"[stderr]\n{stderr_text}")

        # Execution results
        for r in execution.result:
            if r.text:
                parts.append(f"[result]\n{r.text}")

        # Error
        if execution.error:
            err = execution.error
            tb = "\n".join(err.traceback) if err.traceback else ""
            error_msg = f"[error] {err.name}: {err.value}"
            if tb:
                error_msg += f"\n{tb}"
            parts.append(error_msg)

        if not parts:
            return "Code executed successfully (no output)."

        return "\n\n".join(parts)

    async def cleanup(self) -> None:
        """Kill the sandbox and release resources."""
        if self._sandbox is not None:
            try:
                await self._sandbox.kill()
                await self._sandbox.close()
            except Exception:
                pass
            finally:
                self._sandbox = None
                self._interpreter = None
                self._installed_libraries = set()
                # Note: we do NOT close self._loop here so callers can
                # still run cleanup() on it; the loop is closed in __del__.

    def __del__(self) -> None:
        if self._sandbox is not None:
            try:
                loop = self._loop
                if loop is not None and not loop.is_closed():
                    if loop.is_running():
                        loop.create_task(self.cleanup())
                    else:
                        loop.run_until_complete(self.cleanup())
            except Exception:
                pass
