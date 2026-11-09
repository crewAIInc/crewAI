"""Shared HTTP client for the NeuralVerge tools."""

from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, ClassVar

from crewai.tools import BaseTool, EnvVar
from crewai.tools.tool_failure import ToolFailure
from pydantic import Field
import requests


NEURALVERGE_BASE_URL = "https://api.neuralverge.ai"

_ERROR_HINTS: dict[int, str] = {
    400: "the request body was rejected",
    401: "the API key is missing or invalid (it is sent in the x-api-key header)",
    402: "the plan or points limit has been reached",
    502: "the upstream source is temporarily unavailable",
}


def drop_none(data: dict[str, Any]) -> dict[str, Any]:
    """Return ``data`` without keys whose value is ``None``."""
    return {key: value for key, value in data.items() if value is not None}


class NeuralVergeBaseTool(BaseTool):
    """Base class for NeuralVerge tools.

    Subclasses set ``endpoint`` and implement ``build_payload``. Every call
    returns the NeuralVerge envelope as a JSON string (``human`` Markdown
    summary, ``machine`` structured result, ``total_points`` charged), or a
    :class:`ToolFailure` when the API answers with an error.
    """

    endpoint: ClassVar[str] = ""

    api_key: str | None = Field(
        default_factory=lambda: os.getenv("NEURALVERGE_API_KEY"),
        description="NeuralVerge API key. Defaults to the NEURALVERGE_API_KEY environment variable.",
    )
    base_url: str = Field(
        default=NEURALVERGE_BASE_URL, description="NeuralVerge API base URL."
    )
    timeout: int = Field(
        default=120, description="HTTP timeout in seconds for one request."
    )
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="NEURALVERGE_API_KEY",
                description="API key for NeuralVerge (app.neuralverge.ai, Settings, API)",
                required=True,
            ),
        ]
    )

    def build_payload(self, **kwargs: Any) -> dict[str, Any]:
        """Map the tool arguments to the NeuralVerge request body."""
        raise NotImplementedError

    def _request(
        self, method: str, path: str, **kwargs: Any
    ) -> dict[str, Any] | ToolFailure:
        if not self.api_key:
            return ToolFailure(
                message="NEURALVERGE_API_KEY is not set and no api_key was passed to the tool.",
                code="missing_api_key",
            )
        url = f"{self.base_url.rstrip('/')}/functions/v1/{path}"
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "User-Agent": "crewai-tools",
        }
        try:
            response = requests.request(
                method, url, headers=headers, timeout=self.timeout, **kwargs
            )
        except requests.RequestException as e:
            return ToolFailure(
                message=f"Could not reach the NeuralVerge API: {e}",
                code="network_error",
                retryable=True,
            )
        if response.status_code >= 400:
            try:
                detail = str(response.json().get("error") or response.text)
            except ValueError:
                detail = response.text
            hint = _ERROR_HINTS.get(response.status_code, "the request failed")
            return ToolFailure(
                message=f"NeuralVerge API returned HTTP {response.status_code}: {hint}. {detail}".strip(),
                code=str(response.status_code),
                retryable=response.status_code >= 500 or response.status_code == 429,
            )
        try:
            data = response.json()
        except ValueError:
            return ToolFailure(
                message="NeuralVerge API returned a response that is not JSON.",
                code="invalid_json",
                retryable=True,
            )
        return data if isinstance(data, dict) else {"result": data}

    def _post(self, body: dict[str, Any]) -> dict[str, Any] | ToolFailure:
        return self._request("POST", self.endpoint, json=drop_none(body))

    def _run(self, **kwargs: Any) -> str | ToolFailure:
        result = self._post(self.build_payload(**kwargs))
        if isinstance(result, ToolFailure):
            return result
        return json.dumps(result, ensure_ascii=False)

    async def _arun(self, **kwargs: Any) -> str | ToolFailure:
        return await asyncio.to_thread(self._run, **kwargs)


class NeuralVergeAsyncTaskTool(NeuralVergeBaseTool):
    """Adds session polling for asynchronous NeuralVerge tasks."""

    poll_interval: float = Field(
        default=3.0, description="Seconds between status polls."
    )
    max_wait: int = Field(
        default=600,
        description="Maximum seconds to wait for the task to complete or fail.",
    )

    def _wait(self, session_id: str) -> str | ToolFailure:
        deadline = time.monotonic() + self.max_wait
        while True:
            status = self._request(
                "GET", "get-session-status", params={"session_id": session_id}
            )
            if isinstance(status, ToolFailure):
                return status
            state = status.get("status")
            if state == "complete":
                results = status.get("results") or {}
                return json.dumps(
                    {"session_id": session_id, **results}, ensure_ascii=False
                )
            if state == "failed":
                return ToolFailure(
                    message=f"NeuralVerge research task {session_id} failed.",
                    code="research_failed",
                    details={"status": status},
                )
            if time.monotonic() >= deadline:
                return ToolFailure(
                    message=(
                        f"NeuralVerge research task {session_id} did not finish within "
                        f"{self.max_wait}s (last status: {state})."
                    ),
                    code="timeout",
                    retryable=True,
                    details={"session_id": session_id},
                )
            time.sleep(self.poll_interval)
