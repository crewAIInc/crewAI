"""Basis klasse voor Kraken Futures API tools."""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
from typing import Any

import httpx
from pydantic import Field

from crewai.tools import BaseTool


class KrakenFuturesBaseTool(BaseTool):
    """Basis klasse voor alle Kraken Futures API tools.

    Biedt authenticatie en request methodes voor zowel publieke als private endpoints.
    De Kraken Futures API gebruikt een andere authenticatie methode dan Spot.

    Attributen:
        api_key: Kraken Futures API sleutel (vereist voor private endpoints).
        api_secret: Kraken Futures API geheim (vereist voor private endpoints).
        base_url: Kraken Futures API basis URL.
    """

    api_key: str = Field(default="", description="Kraken Futures API sleutel")
    api_secret: str = Field(default="", description="Kraken Futures API geheim")
    base_url: str = Field(
        default="https://futures.kraken.com/derivatives/api/v3",
        description="Kraken Futures API basis URL",
    )

    def _get_futures_signature(self, endpoint: str, nonce: str, post_data: str = "") -> str:
        """Genereer Kraken Futures API handtekening voor geauthenticeerde requests.

        Args:
            endpoint: Het API endpoint pad.
            nonce: Een unieke nonce voor de request.
            post_data: POST data string (optioneel).

        Returns:
            De base64-gecodeerde handtekening.
        """
        # Futures signing: sha256(post_data + nonce + endpoint)
        message = post_data + nonce + endpoint
        sha256_hash = hashlib.sha256(message.encode()).digest()

        # HMAC-SHA512 met base64-decoded secret
        mac = hmac.new(
            base64.b64decode(self.api_secret),
            sha256_hash,
            hashlib.sha512
        )
        return base64.b64encode(mac.digest()).decode()

    def _public_request(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Maak een publieke API request (geen authenticatie vereist).

        Args:
            endpoint: Het API endpoint pad (bijv. "tickers").
            params: Optionele query parameters.

        Returns:
            De JSON response van de API.
        """
        url = f"{self.base_url}/{endpoint}"
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, params=params or {})
            return response.json()

    def _private_request(
        self,
        endpoint: str,
        data: dict[str, Any] | None = None,
        method: str = "POST",
    ) -> dict[str, Any]:
        """Maak een private API request (authenticatie vereist).

        Args:
            endpoint: Het API endpoint pad (bijv. "accounts").
            data: Optionele request data.
            method: HTTP methode (GET of POST).

        Returns:
            De JSON response van de API.

        Raises:
            ValueError: Als API credentials niet geconfigureerd zijn.
        """
        if not self.api_key or not self.api_secret:
            return {
                "error": "API sleutel en geheim zijn vereist voor private endpoints"
            }

        url = f"{self.base_url}/{endpoint}"
        nonce = str(int(time.time() * 1000))

        # Prepare post data
        post_data = ""
        if data:
            post_data = "&".join(f"{k}={v}" for k, v in sorted(data.items()))

        # Generate signature
        signature = self._get_futures_signature(f"/api/v3/{endpoint}", nonce, post_data)

        headers = {
            "APIKey": self.api_key,
            "Nonce": nonce,
            "Authent": signature,
            "Content-Type": "application/x-www-form-urlencoded",
        }

        with httpx.Client(timeout=30.0) as client:
            if method.upper() == "GET":
                response = client.get(url, params=data, headers=headers)
            else:
                response = client.post(url, data=data, headers=headers)
            return response.json()
