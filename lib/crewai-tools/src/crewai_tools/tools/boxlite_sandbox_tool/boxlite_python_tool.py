from __future__ import annotations

from builtins import type as type_
from typing import Any

from pydantic import BaseModel, Field

from crewai_tools.tools.boxlite_sandbox_tool.boxlite_base_tool import BoxLiteBaseTool


class BoxLitePythonToolSchema(BaseModel):
    code: str = Field(..., description="Python source to execute inside the box.")
    envs: dict[str, str] | None = Field(
        default=None,
        description="Optional environment variables for the run.",
    )
    timeout: float | None = Field(
        default=None,
        description="Maximum seconds to wait for the code to finish.",
    )


class BoxLitePythonTool(BoxLiteBaseTool):
    """Run Python code inside a BoxLite micro-VM.

    Executes the code with the box image's ``python3`` and returns captured
    stdout, stderr, and the process exit code. Pair with ``persistent=True``
    (and a Python image) to let imports and installed packages carry across
    calls. Unlike the E2B code interpreter there is no Jupyter kernel, so rich
    results (charts, dataframes) are not returned — only text streams.
    """

    name: str = "BoxLite Sandbox Python"
    description: str = (
        "Execute a block of Python code inside a BoxLite micro-VM and return "
        "captured stdout, stderr, and the process exit code. Use this for data "
        "processing, quick scripts, or analysis that should run in an isolated "
        "environment. The box image must provide python3 (the default "
        "'python:slim' does)."
    )
    args_schema: type_[BaseModel] = BoxLitePythonToolSchema

    def _run(
        self,
        code: str,
        envs: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> Any:
        box, should_remove = self._acquire_box()
        try:
            # Run through /bin/sh so python3 resolves on PATH regardless of the
            # image's layout, and pass the source as a positional arg ($1) so no
            # shell quoting is applied to the user's code.
            result = box.exec(
                "/bin/sh",
                "-c",
                'python3 -c "$1"',
                "sh",
                code,
                env=envs,
                timeout=timeout,
            )
            return self._result_dict(result)
        finally:
            self._release_box(box, should_remove)
