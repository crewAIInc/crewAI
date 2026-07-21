import json
import os
from typing import Any

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr

from crewai_tools.security.safe_path import validate_url


class HauntExtractToolInput(BaseModel):
    """Input schema for HauntExtractTool."""

    url: str = Field(..., description="Mandatory public web page URL to read")
    prompt: str = Field(
        ...,
        description=(
            "Plain-language description of the data to return, for example "
            "'the product name, price and stock status'"
        ),
    )


class HauntExtractTool(BaseTool):
    name: str = "HauntExtractTool"
    description: str = (
        "Extract structured data or clean text from a public web page using the "
        "Haunt API. Takes a URL and a plain-language prompt describing the data "
        "to return. Returns JSON, or an honest error_code (access_denied, "
        "login_required, not_found) instead of made-up data when the page "
        "cannot be read."
    )
    args_schema: type[BaseModel] = HauntExtractToolInput
    base_url: str = "https://hauntapi.com"
    timeout: int = 120
    response_format: str | None = None

    _api_key: str | None = PrivateAttr(default=None)

    def __init__(self, api_key: str | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self._api_key = api_key or os.environ.get("HAUNT_API_KEY")

    def _run(self, url: str, prompt: str) -> str:
        if not self._api_key:
            raise ValueError(
                "Haunt API key missing. Pass api_key or set the HAUNT_API_KEY "
                "environment variable. Free key: https://hauntapi.com/#signup"
            )
        url = validate_url(url)
        body: dict[str, Any] = {"url": url, "prompt": prompt}
        if self.response_format is not None:
            body["response_format"] = self.response_format
        response = requests.post(
            f"{self.base_url}/v1/extract",
            headers={"X-API-Key": self._api_key, "Content-Type": "application/json"},
            json=body,
            timeout=self.timeout,
        )
        try:
            data = response.json()
        except ValueError:
            data = {}
        if not response.ok:
            raise ValueError(
                data.get("message")
                or data.get("error")
                or f"Haunt API error {response.status_code}"
            )
        if not data.get("success"):
            # Honest failure: give the agent a reason code, never invented content.
            return json.dumps(
                {
                    "error_code": data.get("error_code") or "extraction_failed",
                    "message": data.get("message")
                    or data.get("error")
                    or "extraction failed",
                }
            )
        payload = data.get("data")
        if isinstance(payload, dict) and isinstance(payload.get("markdown"), str):
            return payload["markdown"]
        return payload if isinstance(payload, str) else json.dumps(payload)
