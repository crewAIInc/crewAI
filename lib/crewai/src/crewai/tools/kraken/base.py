"""Base class for Kraken API tools."""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
import urllib.parse
from typing import Any

import httpx
from pydantic import Field

from crewai.tools import BaseTool


class KrakenBaseTool(BaseTool):
    """Base class for all Kraken API tools.

    Provides authentication and request methods for both public and private endpoints.

    Attributes:
        api_key: Kraken API key (required for private endpoints).
        api_secret: Kraken API secret (required for private endpoints).
        base_url: Kraken API base URL.
    """

    api_key: str = Field(default="", description="Kraken API key")
    api_secret: str = Field(default="", description="Kraken API secret")
    base_url: str = Field(
        default="https://api.kraken.com", description="Kraken API base URL"
    )

    def _get_kraken_signature(
        self, urlpath: str, data: dict[str, Any], nonce: str
    ) -> str:
        """Generate Kraken API signature for authenticated requests.

        Args:
            urlpath: The API endpoint path.
            data: The request data.
            nonce: A unique nonce for the request.

        Returns:
            The base64-encoded signature.
        """
        postdata = urllib.parse.urlencode(data)
        encoded = (nonce + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        mac = hmac.new(base64.b64decode(self.api_secret), message, hashlib.sha512)
        return base64.b64encode(mac.digest()).decode()

    def _public_request(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Make a public API request (no authentication required).

        Args:
            endpoint: The API endpoint name (e.g., "Time", "Ticker").
            params: Optional query parameters.

        Returns:
            The JSON response from the API.
        """
        url = f"{self.base_url}/0/public/{endpoint}"
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, params=params or {})
            return response.json()

    def _private_request(
        self, endpoint: str, data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Make a private API request (authentication required).

        Args:
            endpoint: The API endpoint name (e.g., "Balance", "AddOrder").
            data: Optional request data.

        Returns:
            The JSON response from the API.

        Raises:
            ValueError: If API credentials are not configured.
        """
        if not self.api_key or not self.api_secret:
            return {
                "error": ["EAPI:Invalid key - API key and secret are required for private endpoints"]
            }

        url = f"{self.base_url}/0/private/{endpoint}"
        nonce = str(int(time.time() * 1000))
        data = data or {}
        data["nonce"] = nonce

        urlpath = f"/0/private/{endpoint}"
        headers = {
            "API-Key": self.api_key,
            "API-Sign": self._get_kraken_signature(urlpath, data, nonce),
        }

        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, data=data, headers=headers)
            return response.json()
