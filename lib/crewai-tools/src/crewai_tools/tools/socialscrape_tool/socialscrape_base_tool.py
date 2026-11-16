from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.parse import urlsplit

from crewai.tools import BaseTool, EnvVar
from pydantic import ConfigDict, Field, PrivateAttr


try:
    from socialscrape import (  # type: ignore[import-not-found,import-untyped,unused-ignore]
        APIError,
        SocialScrape,
        SocialScrapeConnectionError,
        SocialScrapeTimeoutError,
    )

    SOCIALSCRAPE_AVAILABLE = True
except ImportError:
    SOCIALSCRAPE_AVAILABLE = False


API_BASE_URL = "https://api.socialscrape.dev"
_ENDPOINT_ID = re.compile(r"^[a-z0-9_]+(\.[a-z0-9_]+){0,4}$")
_INSTALL_HINT = (
    "The 'socialscrape-sdk' package (Python 3.11 or newer) is required for the "
    "SocialScrape tools. Install it with: uv add socialscrape-sdk "
    "or: pip install 'crewai-tools[socialscrape]'"
)


def endpoint_path(endpoint_id: str) -> str:
    """Turn an endpoint id such as ``facebook.ad_library.search`` into its API path.

    The id is checked against a strict pattern, so the result is always a
    ``/v1/...`` path on the SocialScrape API and never a URL or a traversal.
    """
    endpoint_id = endpoint_id.strip().lower()
    if not _ENDPOINT_ID.fullmatch(endpoint_id):
        raise ValueError(
            f"Invalid endpoint id {endpoint_id!r}. Use dotted lowercase ids such as "
            "'tiktok.profile' or 'facebook.ad_library.search'."
        )
    return "/v1/" + endpoint_id.replace(".", "/").replace("_", "-")


def platform_from_url(url: str, hosts: dict[str, tuple[str, ...]]) -> str | None:
    """Return the platform whose host list matches the URL's host, if any."""
    host = (urlsplit(url if "//" in url else "https://" + url).hostname or "").lower()
    for platform, names in hosts.items():
        if any(host == name or host.endswith("." + name) for name in names):
            return platform
    return None


class SocialScrapeBaseTool(BaseTool):
    """Base class of the SocialScrape tools.

    SocialScrape is a social media data API. Every tool sends one GET request to
    https://api.socialscrape.dev with the API key and returns a JSON string that
    holds ``data``, ``credits_charged`` and ``credits_remaining``. The tools never
    fetch the URLs they are given; a URL is only passed on to the API as a
    parameter.

    Args:
        api_key: SocialScrape API key. Defaults to the SOCIALSCRAPE_API_KEY
            environment variable.
        max_age_days: Accept a cached result up to this many days old (0 to 30).
            A cached result costs 0 credits.
        trim: Ask for the smaller response, without null fields. Default True.
        max_output_chars: Longest answer returned to the agent, in characters.
            Longer answers are cut and say so. Default 50000.
        client: A configured ``socialscrape.SocialScrape`` client to use instead
            of creating one.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: str | None = Field(default=None, repr=False, exclude=True)
    max_age_days: float | None = Field(default=None, ge=0, le=30)
    trim: bool = True
    max_output_chars: int = Field(default=50_000, ge=1_000)
    client: Any = Field(default=None, exclude=True, repr=False)
    package_dependencies: list[str] = Field(
        default_factory=lambda: ["socialscrape-sdk"]
    )
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SOCIALSCRAPE_API_KEY",
                description="API key for SocialScrape (https://socialscrape.dev/app)",
                required=True,
            ),
        ]
    )
    _client: Any = PrivateAttr(default=None)

    def _get_client(self) -> Any:
        if self.client is not None:
            return self.client
        if self._client is None:
            if not SOCIALSCRAPE_AVAILABLE:
                raise ImportError(_INSTALL_HINT)
            key = self.api_key or os.environ.get("SOCIALSCRAPE_API_KEY")
            if not key:
                raise ValueError(
                    "SocialScrape API key not found. Pass api_key or set the "
                    "SOCIALSCRAPE_API_KEY environment variable. "
                    "Create a key at https://socialscrape.dev/app"
                )
            # The base URL is fixed, so SOCIALSCRAPE_BASE_URL in the environment
            # cannot send the key somewhere else.
            self._client = SocialScrape(api_key=key, base_url=API_BASE_URL)
        return self._client

    def _request(self, endpoint_id: str, params: dict[str, Any], **options: Any) -> str:
        """Call one endpoint and return the answer as a JSON string."""
        path = endpoint_path(endpoint_id)
        query = {k: v for k, v in params.items() if v is not None}
        if self.trim:
            query.setdefault("trim", True)
        if self.max_age_days is not None:
            query.setdefault("max_age_days", self.max_age_days)
        client = self._get_client()
        try:
            resp = client.request("GET", path, query=query)
        except (APIError, SocialScrapeConnectionError, SocialScrapeTimeoutError) as e:
            # A refused request is free. Hand the error to the agent so it can
            # correct the call; the message never contains the API key.
            return f"Error: {e}"
        out: dict[str, Any] = {
            "data": self._shape(resp.data, **options),
            "credits_charged": resp.credits_charged,
            "credits_remaining": resp.credits_remaining,
        }
        if resp.cached:
            out["cached"] = True
        return self._limit(json.dumps(out, ensure_ascii=False))

    def _shape(self, data: Any, **options: Any) -> Any:
        return data

    def _limit(self, text: str) -> str:
        if len(text) <= self.max_output_chars:
            return text
        return (
            text[: self.max_output_chars]
            + f"\n[Answer cut at {self.max_output_chars} characters. Ask for a "
            "smaller page, or raise max_output_chars.]"
        )
