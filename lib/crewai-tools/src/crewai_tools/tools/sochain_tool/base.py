from __future__ import annotations

import json
import os
import time
from typing import Any, Literal

from crewai.tools import BaseTool, EnvVar
from pydantic import Field
import requests


SOCHAIN_API_URL = "https://sochain.com/api/v3"

Network = Literal[
    "BTC", "LTC", "DOGE", "ZEC", "BTCTEST", "LTCTEST", "DOGETEST", "ZECTEST"
]

NETWORK_DESCRIPTION = (
    "Network code: BTC, LTC, DOGE or ZEC, or a testnet: BTCTEST (Bitcoin "
    "testnet4), LTCTEST, DOGETEST or ZECTEST."
)

# Longest Retry-After (seconds) the tool waits out before retrying once.
_MAX_RETRY_WAIT = 10.0


def _retry_after(resp: requests.Response) -> float | None:
    value = resp.headers.get("Retry-After")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


class SoChainBaseTool(BaseTool):
    """Shared request handling for the SoChain API v3 tools.

    The API key is optional: without one, lookups are limited per IP address.
    Keys come from https://sochain.com/dashboard.
    """

    api_key: str | None = Field(
        default_factory=lambda: os.getenv("SOCHAIN_API_KEY"),
        description=(
            "Optional SoChain API key, sent as the API-KEY header. "
            "Get one at https://sochain.com/dashboard."
        ),
        json_schema_extra={"required": False},
    )
    base_url: str = SOCHAIN_API_URL
    timeout: int = 30
    env_vars: list[EnvVar] = Field(
        default_factory=lambda: [
            EnvVar(
                name="SOCHAIN_API_KEY",
                description=(
                    "Optional SoChain API key for higher limits "
                    "(https://sochain.com/dashboard)"
                ),
                required=False,
            ),
        ]
    )
    package_dependencies: list[str] = Field(default_factory=lambda: ["requests"])

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json", "User-Agent": "crewai-tools"}
        if self.api_key:
            headers["API-KEY"] = self.api_key
        return headers

    def _get(self, path: str, params: dict[str, Any] | None = None) -> str:
        """GET a SoChain API path and return its ``data`` as a JSON string.

        Failures come back as a short message the agent can read.
        """
        data, error = self._get_data(path, params)
        if error is not None:
            return error
        return json.dumps(data, ensure_ascii=False)

    def _get_data(
        self, path: str, params: dict[str, Any] | None = None
    ) -> tuple[Any, str | None]:
        """GET a SoChain API path; return ``(data, None)`` or ``(None, error)``."""
        url = f"{self.base_url.rstrip('/')}/{path}"
        try:
            resp = requests.get(
                url, headers=self._headers(), params=params, timeout=self.timeout
            )
            if resp.status_code == 429:
                wait = _retry_after(resp)
                if wait is not None and wait <= _MAX_RETRY_WAIT:
                    time.sleep(wait)
                    resp = requests.get(
                        url,
                        headers=self._headers(),
                        params=params,
                        timeout=self.timeout,
                    )
        except requests.Timeout:
            return None, "SoChain API error: the request timed out."
        except requests.RequestException as exc:
            return None, f"SoChain API error: {exc}"

        if resp.status_code == 401:
            return None, (
                "SoChain API error (HTTP 401): the API key was not accepted. "
                "Check SOCHAIN_API_KEY or create a key at "
                "https://sochain.com/dashboard."
            )
        if resp.status_code == 429:
            wait = _retry_after(resp)
            after = f" Retry after {wait:g} seconds." if wait is not None else ""
            return None, (
                "SoChain API error (HTTP 429): rate limited."
                f"{after} An API key from https://sochain.com/dashboard "
                "raises the limit."
            )

        try:
            body = resp.json()
        except ValueError:
            return None, (
                f"SoChain API error (HTTP {resp.status_code}): {resp.text[:200]}"
            )

        if not isinstance(body, dict):
            return None, (
                f"SoChain API error (HTTP {resp.status_code}): unexpected response"
            )
        data = body.get("data")
        if body.get("status") != "success":
            message = (
                data.get("error_message") if isinstance(data, dict) else None
            ) or resp.text[:200]
            return None, f"SoChain API error (HTTP {resp.status_code}): {message}"
        return data, None
