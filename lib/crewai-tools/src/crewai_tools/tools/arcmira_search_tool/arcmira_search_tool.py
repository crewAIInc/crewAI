"""Search indexed YouTube transcript passages with Arcmira."""

import json
import os
from typing import Any

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field, SecretStr
import requests


def _api_key_from_env() -> SecretStr:
    value = os.getenv("ARCMIRA_API_KEY", "").strip()
    if not value:
        raise ValueError("Set ARCMIRA_API_KEY or pass api_key to ArcmiraSearchTool.")
    return SecretStr(value)


class ArcmiraSearchToolSchema(BaseModel):
    """Input for a transcript passage search."""

    query: str = Field(
        ...,
        min_length=2,
        description="One topic or phrase to find in indexed YouTube transcripts.",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum transcript passages to return. Each result uses account usage.",
    )


class ArcmiraSearchTool(BaseTool):
    """Search transcript passages while preserving timestamps and source links."""

    name: str = "Arcmira: YouTube Transcript Search"
    description: str = (
        "Search indexed YouTube transcript passages for one topic or phrase. "
        "Use the returned timestamps and source links when citing what was said. "
        "Report partial results and access limits. An empty result means no match "
        "in the index, not that nobody discussed the topic. Calls use Arcmira account usage."
    )
    args_schema: type[BaseModel] = ArcmiraSearchToolSchema
    api_key: SecretStr = Field(
        default_factory=_api_key_from_env, exclude=True, repr=False, min_length=1
    )
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="ARCMIRA_API_KEY",
                description="Arcmira API key, or pass api_key when creating the tool.",
                required=True,
            )
        ]
    )

    def _run(self, query: str, limit: int = 5) -> dict[str, Any]:
        """Return search evidence or raise an error without automatic retries."""
        params: dict[str, str | int] = {"q": query, "limit": limit}
        try:
            response = requests.get(
                "https://api.arcmira.com/v1/search",
                params=params,
                headers={"Authorization": f"Bearer {self.api_key.get_secret_value()}"},
                timeout=30,
                allow_redirects=False,
            )
        except requests.RequestException:
            raise RuntimeError(
                "Arcmira search could not complete the network request. "
                "No automatic retry was attempted."
            ) from None

        try:
            payload = response.json()
        except ValueError:
            raise RuntimeError(
                f"Arcmira returned a non-JSON response (HTTP {response.status_code})."
            ) from None
        if response.status_code != 200:
            raise RuntimeError(
                f"Arcmira search failed (HTTP {response.status_code}): "
                f"{json.dumps(payload, ensure_ascii=False)}"
            )
        if not isinstance(payload, dict):
            raise RuntimeError("Arcmira returned an unexpected response format.")
        return payload
