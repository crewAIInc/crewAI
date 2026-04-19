from typing import Any, Optional, Type

import requests
from crewai.tools import BaseTool
from pydantic import BaseModel, Field, model_validator


class DeepKeepMakeApiCallSchema(BaseModel):
    """Input schema for DeepKeepMakeApiCallTool."""

    path: str = Field(
        ...,
        description=(
            "The API path relative to deepkeep.ai/api, e.g. '/v2/firewalls'. "
            "A leading slash is optional."
        ),
    )
    method: str = Field(
        "GET",
        description="HTTP method: GET, POST, PUT, or DELETE.",
    )
    query_params: Optional[dict[str, str]] = Field(
        None,
        description="Optional dictionary of URL query-string parameters.",
    )
    extra_headers: Optional[dict[str, str]] = Field(
        None,
        description=(
            "Optional extra request headers. Auth headers are injected automatically — "
            "do not include X-API-Key here."
        ),
    )
    body: Optional[str] = Field(
        None,
        description=(
            "Optional raw request body (usually a JSON string). "
            "Leave empty for GET or DELETE requests."
        ),
    )


class DeepKeepMakeApiCallTool(BaseTool):
    """Perform any authenticated HTTP request to the DeepKeep API.

    An escape-hatch for endpoints not yet covered by the dedicated tools.
    Auth headers are injected automatically.

    Returns a JSON string with { "statusCode", "headers", "body" }, matching
    the envelope used by the Make.com and n8n DeepKeep integrations.

    Requires:
        subdomain: Your DeepKeep tenant subdomain (e.g. "acme").
        api_key:   Your DeepKeep API key.
    """

    name: str = "DeepKeep Make API Call"
    description: str = (
        "Perform any authenticated HTTP request to the DeepKeep API. "
        "Use this for endpoints not covered by the dedicated tools. "
        "Returns a JSON object with statusCode, headers, and body."
    )
    args_schema: Type[BaseModel] = DeepKeepMakeApiCallSchema

    subdomain: str = Field(
        ..., description="Your DeepKeep tenant subdomain (e.g. 'acme')."
    )
    api_key: str = Field(..., description="Your DeepKeep API key.")

    @model_validator(mode="before")
    @classmethod
    def validate_credentials(cls, values: dict[str, Any]) -> dict[str, Any]:
        subdomain = values.get("subdomain", "")
        api_key = values.get("api_key", "")
        if not subdomain:
            raise ValueError(
                "DeepKeepMakeApiCallTool requires a non-empty 'subdomain'."
            )
        if not api_key:
            raise ValueError(
                "DeepKeepMakeApiCallTool requires a non-empty 'api_key'."
            )
        return values

    def _run(
        self,
        path: str,
        method: str = "GET",
        query_params: Optional[dict[str, str]] = None,
        extra_headers: Optional[dict[str, str]] = None,
        body: Optional[str] = None,
    ) -> str:
        import json

        base_url = f"https://api.{self.subdomain}.deepkeep.ai/api"
        clean_path = path.lstrip("/")
        url = f"{base_url}/{clean_path}"

        headers: dict[str, str] = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }
        if extra_headers:
            headers.update(extra_headers)

        response = requests.request(
            method=method.upper(),
            url=url,
            headers=headers,
            params=query_params or {},
            data=body or None,
            timeout=30,
        )

        try:
            parsed_body: Any = response.json()
        except ValueError:
            parsed_body = response.text

        return json.dumps(
            {
                "statusCode": response.status_code,
                "headers": dict(response.headers),
                "body": parsed_body,
            },
            indent=2,
        )
