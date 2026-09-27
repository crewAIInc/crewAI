import os
import re
from typing import Any
from urllib.parse import urlparse

from crewai.tools import BaseTool, EnvVar
from pydantic import Field


class SearchApiBaseTool(BaseTool):
    """Base class for SearchApi functionality with shared capabilities."""

    url: str = "https://www.searchapi.io/api/v1/search"
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SEARCHAPI_API_KEY",
                description="API key for SearchApi searches",
                required=True,
            ),
        ]
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        api_key = os.getenv("SEARCHAPI_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing API key, you can get the key from https://www.searchapi.io/"
            )
        self._validate_url()

    def _validate_url(self) -> None:
        parsed = urlparse(self.url)
        if parsed.scheme.lower() != "https":
            raise ValueError(
                f"URL scheme must be HTTPS to prevent sending credentials in cleartext, got '{parsed.scheme}'."
            )
        hostname = (parsed.hostname or "").lower()
        if hostname not in {"searchapi.io", "www.searchapi.io"}:
            raise ValueError(
                f"URL host must be searchapi.io or www.searchapi.io to prevent sending credentials to unauthorized hosts, got '{hostname}'."
            )

    def _omit_fields(
        self, data: dict[str, Any] | list[Any], omit_patterns: list[str]
    ) -> None:
        if isinstance(data, dict):
            for field in list(data.keys()):
                if any(re.compile(p).match(field) for p in omit_patterns):
                    data.pop(field, None)
                else:
                    if isinstance(data[field], (dict, list)):
                        self._omit_fields(data[field], omit_patterns)
        elif isinstance(data, list):
            for item in data:
                self._omit_fields(item, omit_patterns)
