"""Minimal CrewAI tool for the Reserp Google Search API."""

from __future__ import annotations

import os
from typing import Any, cast

from crewai.tools import BaseTool, EnvVar
from pydantic import BaseModel, Field
import requests


ENDPOINT = "https://api.reserp.ai/v1/serp"


class ReserpSearchToolSchema(BaseModel):
    """Public Reserp request body."""

    url: str = Field(
        ...,
        description=(
            "Complete https://www.google.com/search URL containing a non-empty q "
            "parameter. Do not include Google's num parameter."
        ),
    )


class ReserpSearchTool(BaseTool):
    """Expose one public Reserp request as a CrewAI tool call."""

    name: str = "Search Google with Reserp"
    description: str = (
        "Send a complete Google Search URL to Reserp and return the public JSON "
        "response unchanged. The caller controls retries and all orchestration."
    )
    args_schema: type[BaseModel] = ReserpSearchToolSchema
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="RESERP_API_KEY",
                description="Bearer API key for the Reserp Google Search API",
                required=True,
            )
        ]
    )

    def _run(self, url: str, **_: Any) -> dict[str, Any]:
        response = requests.post(  # noqa: S113 -- timeout policy belongs to the caller
            ENDPOINT,
            headers={
                "Authorization": f"Bearer {os.environ['RESERP_API_KEY']}",
                "Content-Type": "application/json",
            },
            json={"url": url},
        )
        response.raise_for_status()
        return cast(dict[str, Any], response.json())
