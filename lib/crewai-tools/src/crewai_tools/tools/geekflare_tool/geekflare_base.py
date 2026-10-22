import asyncio
import json
import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import Field
import requests

from crewai_tools.security.safe_path import validate_url


GEEKFLARE_API_URL = "https://api.geekflare.com"


class GeekflareBaseTool(BaseTool):
    """Shared auth and request handling for Geekflare API tools."""

    api_key: str | None = Field(
        default_factory=lambda: os.getenv("GEEKFLARE_API_KEY"),
        description="Geekflare API key. Defaults to the GEEKFLARE_API_KEY environment variable.",
    )
    base_url: str = Field(
        default=GEEKFLARE_API_URL, description="Geekflare API base URL."
    )
    timeout: int = Field(default=180, description="Request timeout in seconds.")
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="GEEKFLARE_API_KEY",
                description="API key for Geekflare (https://dash.geekflare.com)",
                required=True,
            ),
        ]
    )

    @staticmethod
    def _validate_url(url: str) -> str:
        """Validate an agent-supplied URL or bare domain with the shared SSRF guard."""
        if "://" not in url:
            url = f"https://{url}"
        return validate_url(url)

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """POST a JSON payload to a Geekflare endpoint and return the parsed response."""
        if not self.api_key:
            raise ValueError(
                "GEEKFLARE_API_KEY is not set. Pass api_key or set the environment variable."
            )
        body = {k: v for k, v in payload.items() if v is not None}
        response = requests.post(
            f"{self.base_url.rstrip('/')}/{path}",
            json=body,
            headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
            timeout=self.timeout,
        )
        try:
            parsed: Any = response.json()
        except ValueError:
            parsed = {"message": response.text}
        data: dict[str, Any] = parsed if isinstance(parsed, dict) else {"data": parsed}

        if response.status_code >= 400 or data.get("apiStatus") == "failure":
            raise RuntimeError(
                f"Geekflare API error ({response.status_code}) on /{path}: "
                f"{data.get('message', data)}"
            )
        return data

    def _post_json(self, path: str, payload: dict[str, Any]) -> str:
        return json.dumps(self._post(path, payload), indent=2)

    async def _arun(self, *args: Any, **kwargs: Any) -> Any:
        return await asyncio.to_thread(self._run, *args, **kwargs)
