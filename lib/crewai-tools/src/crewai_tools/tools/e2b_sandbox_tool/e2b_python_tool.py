from __future__ import annotations

from builtins import type as type_
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from crewai_tools.tools.e2b_sandbox_tool.e2b_base_tool import E2BBaseTool


class E2BPythonToolSchema(BaseModel):
    code: str = Field(
        ...,
        description="Python source to execute inside the sandbox.",
    )
    language: str | None = Field(
        default=None,
        description=(
            "Override the execution language (e.g. 'python', 'r', 'javascript'). "
            "Defaults to Python when omitted."
        ),
    )
    envs: dict[str, str] | None = Field(
        default=None,
        description="Optional environment variables for the run.",
    )
    timeout: float | None = Field(
        default=None,
        description="Maximum seconds to wait for the code to finish.",
    )


class E2BPythonTool(E2BBaseTool):
    """Run Python code inside an E2B code interpreter sandbox.

    Uses `e2b_code_interpreter`, which runs cells in a persistent Jupyter-style
    kernel so state (imports, variables) carries across calls when
    `persistent=True`.
    """

    name: str = "E2B Sandbox Python"
    description: str = (
        "Execute a block of Python code inside an E2B code interpreter sandbox "
        "and return captured stdout, stderr, the final expression value, and "
        "any rich results (charts, dataframes). Use this for data processing, "
        "quick scripts, or analysis that should run in an isolated environment."
    )
    args_schema: type_[BaseModel] = E2BPythonToolSchema

    package_dependencies: list[str] = Field(
        default_factory=lambda: ["e2b_code_interpreter"],
    )

    _ci_cache: ClassVar[dict[str, Any]] = {}

    @classmethod
    def _import_sandbox_class(cls) -> Any:
        cached = cls._ci_cache.get("Sandbox")
        if cached is not None:
            return cached
        try:
            from e2b_code_interpreter import Sandbox  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "The 'e2b_code_interpreter' package is required for the E2B "
                "Python tool. Install it with: "
                "uv add e2b-code-interpreter  (or) "
                "pip install e2b-code-interpreter"
            ) from exc
        cls._ci_cache["Sandbox"] = Sandbox
        return Sandbox

    def _run(
        self,
        code: str,
        language: str | None = None,
        envs: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        sandbox, should_kill = self._acquire_sandbox()
        try:
            run_kwargs: dict[str, Any] = {}
            if language is not None:
                run_kwargs["language"] = language
            if envs is not None:
                run_kwargs["envs"] = envs
            if timeout is not None:
                run_kwargs["timeout"] = timeout
            execution = sandbox.run_code(code, **run_kwargs)
            return self._serialize_execution(execution)
        finally:
            self._release_sandbox(sandbox, should_kill)

    @staticmethod
    def _serialize_execution(execution: Any) -> dict[str, Any]:
        logs = getattr(execution, "logs", None)
        error = getattr(execution, "error", None)
        results = getattr(execution, "results", None) or []
        return {
            "text": getattr(execution, "text", None),
            "stdout": list(getattr(logs, "stdout", []) or []) if logs else [],
            "stderr": list(getattr(logs, "stderr", []) or []) if logs else [],
            "error": (
                {
                    "name": getattr(error, "name", None),
                    "value": getattr(error, "value", None),
                    "traceback": getattr(error, "traceback", None),
                }
                if error
                else None
            ),
            "results": [E2BPythonTool._serialize_result(r) for r in results],
            "execution_count": getattr(execution, "execution_count", None),
        }

    @staticmethod
    def _serialize_result(result: Any) -> dict[str, Any]:
        fields = (
            "text",
            "html",
            "markdown",
            "svg",
            "png",
            "jpeg",
            "pdf",
            "latex",
            "json",
            "javascript",
            "data",
            "is_main_result",
            "extra",
        )
        return {field: getattr(result, field, None) for field in fields}
