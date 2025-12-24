"""Basis klasse voor Kraken API tools."""

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
    """Basis klasse voor alle Kraken API tools.

    Biedt authenticatie en request methodes voor zowel publieke als private endpoints.

    Attributen:
        api_key: Kraken API sleutel (vereist voor private endpoints).
        api_secret: Kraken API geheim (vereist voor private endpoints).
        base_url: Kraken API basis URL.
    """

    api_key: str = Field(default="", description="Kraken API sleutel")
    api_secret: str = Field(default="", description="Kraken API geheim")
    base_url: str = Field(
        default="https://api.kraken.com", description="Kraken API basis URL"
    )

    def _get_kraken_signature(
        self, urlpath: str, data: dict[str, Any], nonce: str
    ) -> str:
        """Genereer Kraken API handtekening voor geauthenticeerde requests.

        Args:
            urlpath: Het API endpoint pad.
            data: De request data.
            nonce: Een unieke nonce voor de request.

        Returns:
            De base64-gecodeerde handtekening.
        """
        postdata = urllib.parse.urlencode(data)
        encoded = (nonce + postdata).encode()
        message = urlpath.encode() + hashlib.sha256(encoded).digest()
        mac = hmac.new(base64.b64decode(self.api_secret), message, hashlib.sha512)
        return base64.b64encode(mac.digest()).decode()

    def _public_request(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Maak een publieke API request (geen authenticatie vereist).

        Args:
            endpoint: De API endpoint naam (bijv. "Time", "Ticker").
            params: Optionele query parameters.

        Returns:
            De JSON response van de API.
        """
        url = f"{self.base_url}/0/public/{endpoint}"
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, params=params or {})
            return response.json()

    def _private_request(
        self, endpoint: str, data: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Maak een private API request (authenticatie vereist).

        Args:
            endpoint: De API endpoint naam (bijv. "Balance", "AddOrder").
            data: Optionele request data.

        Returns:
            De JSON response van de API.

        Raises:
            ValueError: Als API credentials niet geconfigureerd zijn.
        """
        if not self.api_key or not self.api_secret:
            return {
                "error": ["EAPI:Invalid key - API sleutel en geheim zijn vereist voor private endpoints"]
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
