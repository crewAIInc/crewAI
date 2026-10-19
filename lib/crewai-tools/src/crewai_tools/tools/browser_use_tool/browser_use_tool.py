from __future__ import annotations

import json
import os
from typing import Any

from crewai.tools import BaseTool, EnvVar, ToolFailure
from pydantic import BaseModel, Field


_INSTALL_HINT = (
    "The 'browser-use-sdk' package is required for BrowserUseTool. "
    "Install it with: uv add browser-use-sdk  (or) pip install browser-use-sdk"
)


class BrowserUseToolSchema(BaseModel):
    task: str = Field(
        ...,
        description=(
            "What to do in the browser, in plain language. Name the website "
            "and the exact information or action you need."
        ),
    )
    session_id: str | None = Field(
        default=None,
        description=(
            "Optional session_id from a previous Browser Use result. Pass it to "
            "continue in the same browser and conversation."
        ),
    )


class BrowserUseTool(BaseTool):
    """Run a browser task with the Browser Use Cloud agent.

    Each call starts a run on Browser Use Cloud (API v4), waits for it to
    finish and returns the agent's final answer. A run that does not finish
    within ``timeout`` seconds is cancelled so it stops using credits.
    """

    name: str = "Browser Use"
    description: str = (
        "Give a browser agent a task in plain language. It opens a cloud "
        "browser, navigates websites, clicks, fills forms and extracts data, "
        "then returns its final answer and a session_id for follow-up tasks."
    )
    args_schema: type[BaseModel] = BrowserUseToolSchema
    api_key: str | None = Field(
        default_factory=lambda: os.getenv("BROWSER_USE_API_KEY"),
        description="API key for Browser Use Cloud",
        json_schema_extra={"required": False},
    )
    model: str | None = Field(
        default=None,
        description="Browser Use agent model. None uses the API default.",
    )
    timeout: float = Field(
        default=900,
        description="Seconds to wait for a run before cancelling it.",
    )
    package_dependencies: list[str] = Field(default_factory=lambda: ["browser-use-sdk"])
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="BROWSER_USE_API_KEY",
                description="API key for Browser Use Cloud",
                required=True,
            ),
        ]
    )

    def _run(self, task: str, session_id: str | None = None) -> str | ToolFailure:
        sdk = _load_sdk()
        if not self.api_key:
            return _missing_key_failure()

        with sdk.BrowserUse(api_key=self.api_key) as client:
            try:
                run = client.runs.create(
                    task, model=self.model, session_id=session_id or None
                )
                summary = client.runs.wait_for_completion(run.id, timeout=self.timeout)
            except sdk.BrowserUseError as exc:
                return _api_failure(exc)
            except TimeoutError:
                try:
                    client.runs.cancel(run.id)
                except sdk.BrowserUseError:
                    pass
                return _timeout_failure(run, self.timeout)
        return _summary_output(summary)

    async def _arun(
        self, task: str, session_id: str | None = None
    ) -> str | ToolFailure:
        sdk = _load_sdk()
        if not self.api_key:
            return _missing_key_failure()

        async with sdk.AsyncBrowserUse(api_key=self.api_key) as client:
            try:
                run = await client.runs.create(
                    task, model=self.model, session_id=session_id or None
                )
                summary = await client.runs.wait_for_completion(
                    run.id, timeout=self.timeout
                )
            except sdk.BrowserUseError as exc:
                return _api_failure(exc)
            except TimeoutError:
                try:
                    await client.runs.cancel(run.id)
                except sdk.BrowserUseError:
                    pass
                return _timeout_failure(run, self.timeout)
        return _summary_output(summary)


def _load_sdk() -> Any:
    try:
        from browser_use_sdk import v4  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ImportError(_INSTALL_HINT) from exc
    return v4


def _summary_output(summary: Any) -> str | ToolFailure:
    status = summary.status.value
    if status != "completed":
        return ToolFailure(
            message=f"Browser Use run {status}: {summary.error or 'no error message'}",
            code=f"run_{status}",
            details={"run_id": str(summary.id), "session_id": str(summary.session_id)},
        )
    return json.dumps(
        {
            "result": summary.result,
            "session_id": str(summary.session_id),
            "run_id": str(summary.id),
        }
    )


def _api_failure(exc: Any) -> ToolFailure:
    status_code: int = exc.status_code
    return ToolFailure(
        message=f"Browser Use API error {status_code}: {exc.message}",
        code=str(status_code),
        retryable=status_code == 429 or status_code >= 500,
    )


def _timeout_failure(run: Any, timeout: float) -> ToolFailure:
    return ToolFailure(
        message=(
            f"Browser Use run did not finish within {timeout:g}s and was "
            "cancelled. Try a smaller task or raise the tool's timeout."
        ),
        code="timeout",
        details={"run_id": str(run.id), "session_id": str(run.session_id)},
    )


def _missing_key_failure() -> ToolFailure:
    return ToolFailure(
        message=(
            "BROWSER_USE_API_KEY is not set. Create a key at "
            "https://cloud.browser-use.com/settings?tab=api-keys"
        ),
        code="missing_api_key",
    )
