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
    """Extract requested data from a public web page through the Haunt API."""

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
        """Return extracted content or a structured terminal failure.

        Args:
            url: Public HTTP(S) page to read.
            prompt: Plain-language description of the requested data.

        Returns:
            Extracted JSON or text on success, or a JSON failure object when
            Haunt reports that the page could not be read safely.
        """
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
            allow_redirects=False,
        )
        try:
            data = response.json()
        except ValueError:
            data = None
        if not 200 <= response.status_code < 300:
            detail = None
            if isinstance(data, dict):
                detail = data.get("message") or data.get("error")
            message = f"Haunt API error {response.status_code}"
            if detail:
                message += f": {detail}"
            raise ValueError(message)
        if not isinstance(data, dict):
            raise ValueError(
                f"Haunt API returned invalid JSON for status {response.status_code}"
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
