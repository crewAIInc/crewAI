"""Authentication schemes for A2A protocol agents.

Supported authentication methods:
- Bearer tokens
- OAuth2 (Client Credentials, Authorization Code)
- API Keys (header, query, cookie)
- HTTP Basic authentication
- HTTP Digest authentication
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import base64
from collections.abc import Awaitable, Callable, MutableMapping
import time
from typing import Literal
import urllib.parse

import httpx
from httpx import DigestAuth
from pydantic import BaseModel, Field, PrivateAttr


class AuthScheme(ABC, BaseModel):
    """Base class for authentication schemes."""

    @abstractmethod
    async def apply_auth(
        self, client: httpx.AsyncClient, headers: MutableMapping[str, str]
    ) -> MutableMapping[str, str]:
        """Apply authentication to request headers.

        Args:
            client: HTTP client for making auth requests.
            headers: Current request headers.

        Returns:
            Updated headers with authentication applied.
        """
        ...


class BearerTokenAuth(AuthScheme):
    """Bearer token authentication (Authorization: Bearer <token>).

    Attributes:
        token: Bearer token for authentication.
    """

    token: str = Field(description="Bearer token")

    async def apply_auth(
        self, client: httpx.AsyncClient, headers: MutableMapping[str, str]
    ) -> MutableMapping[str, str]:
        """Apply Bearer token to Authorization header.

        Args:
            client: HTTP client for making auth requests.
            headers: Current request headers.

        Returns:
            Updated headers with Bearer token in Authorization header.
        """
        headers["Authorization"] = f"Bearer {self.token}"
        return headers


class HTTPBasicAuth(AuthScheme):
    """HTTP Basic authentication.

    Attributes:
        username: Username for Basic authentication.
        password: Password for Basic authentication.
    """

    username: str = Field(description="Username")
    password: str = Field(description="Password")

    async def apply_auth(
        self, client: httpx.AsyncClient, headers: MutableMapping[str, str]
    ) -> MutableMapping[str, str]:
        """Apply HTTP Basic authentication.

        Args:
            client: HTTP client for making auth requests.
            headers: Current request headers.

        Returns:
            Updated headers with Basic auth in Authorization header.
        """
        credentials = f"{self.username}:{self.password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        headers["Authorization"] = f"Basic {encoded}"
        return headers


class HTTPDigestAuth(AuthScheme):
    """HTTP Digest authentication.

    Note: Uses httpx-auth library for digest implementation.

    Attributes:
        username: Username for Digest authentication.
        password: Password for Digest authentication.
    """

    username: str = Field(description="Username")
    password: str = Field(description="Password")

    async def apply_auth(
        self, client: httpx.AsyncClient, headers: MutableMapping[str, str]
    ) -> MutableMapping[str, str]:
        """Digest auth is handled by httpx auth flow, not headers.

        Args:
            client: HTTP client for making auth requests.
            headers: Current request headers.

        Returns:
            Unchanged headers (Digest auth handled by httpx auth flow).
        """
        return headers

    def configure_client(self, client: httpx.AsyncClient) -> None:
        """Configure client with Digest auth.

        Args:
            client: HTTP client to configure with Digest authentication.
        """
        client.auth = DigestAuth(self.username, self.password)


class APIKeyAuth(AuthScheme):
    """API Key authentication (header, query, or cookie).

    Attributes:
        api_key: API key value for authentication.
        location: Where to send the API key (header, query, or cookie).
        name: Parameter name for the API key (default: X-API-Key).
    """

    api_key: str = Field(description="API key value")
    location: Literal["header", "query", "cookie"] = Field(
        default="header", description="Where to send the API key"
    )
    name: str = Field(default="X-API-Key", description="Parameter name for the API key")

    async def apply_auth(
        self, client: httpx.AsyncClient, headers: MutableMapping[str, str]
    ) -> MutableMapping[str, str]:
        """Apply API key authentication.

        Args:
            client: HTTP client for making auth requests.
            headers: Current request headers.

        Returns:
            Updated headers with API key (for header/cookie locations).
        """
        if self.location == "header":
            headers[self.name] = self.api_key
        elif self.location == "cookie":
            headers["Cookie"] = f"{self.name}={self.api_key}"
        return headers

    def configure_client(self, client: httpx.AsyncClient) -> None:
        """Configure client for query param API keys.

        Args:
            client: HTTP client to configure with query param API key hook.
        """
        if self.location == "query":

            async def _add_api_key_param(request: httpx.Request) -> None:
                url = httpx.URL(request.url)
                request.url = url.copy_add_param(self.name, self.api_key)

            client.event_hooks["request"].append(_add_api_key_param)


class OAuth2ClientCredentials(AuthScheme):
    """OAuth2 Client Credentials flow authentication.

    Attributes:
        token_url: OAuth2 token endpoint URL.
        client_id: OAuth2 client identifier.
        client_secret: OAuth2 client secret.
        scopes: List of required OAuth2 scopes.
    """

    token_url: str = Field(description="OAuth2 token endpoint")
    client_id: str = Field(description="OAuth2 client ID")
    client_secret: str = Field(description="OAuth2 client secret")
    scopes: list[str] = Field(
        default_factory=list, description="Required OAuth2 scopes"
    )

    _access_token: str | None = PrivateAttr(default=None)
    _token_expires_at: float | None = PrivateAttr(default=None)

    async def apply_auth(
        self, client: httpx.AsyncClient, headers: MutableMapping[str, str]
    ) -> MutableMapping[str, str]:
        """Apply OAuth2 access token to Authorization header.

        Args:
            client: HTTP client for making token requests.
            headers: Current request headers.

        Returns:
            Updated headers with OAuth2 access token in Authorization header.
        """
        if (
            self._access_token is None
            or self._token_expires_at is None
            or time.time() >= self._token_expires_at
        ):
            await self._fetch_token(client)

        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"

        return headers

    async def _fetch_token(self, client: httpx.AsyncClient) -> None:
        """Fetch OAuth2 access token using client credentials flow.

        Args:
            client: HTTP client for making token request.

        Raises:
            httpx.HTTPStatusError: If token request fails.
        """
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        if self.scopes:
            data["scope"] = " ".join(self.scopes)

        response = await client.post(self.token_url, data=data)
        response.raise_for_status()

        token_data = response.json()
        self._access_token = token_data["access_token"]
        expires_in = token_data.get("expires_in", 3600)
        self._token_expires_at = time.time() + expires_in - 60


class OAuth2AuthorizationCode(AuthScheme):
    """OAuth2 Authorization Code flow authentication.

    Note: Requires interactive authorization.

    Attributes:
        authorization_url: OAuth2 authorization endpoint URL.
        token_url: OAuth2 token endpoint URL.
        client_id: OAuth2 client identifier.
        client_secret: OAuth2 client secret.
        redirect_uri: OAuth2 redirect URI for callback.
        scopes: List of required OAuth2 scopes.
    """

    authorization_url: str = Field(description="OAuth2 authorization endpoint")
    token_url: str = Field(description="OAuth2 token endpoint")
    client_id: str = Field(description="OAuth2 client ID")
    client_secret: str = Field(description="OAuth2 client secret")
    redirect_uri: str = Field(description="OAuth2 redirect URI")
    scopes: list[str] = Field(
        default_factory=list, description="Required OAuth2 scopes"
    )

    _access_token: str | None = PrivateAttr(default=None)
    _refresh_token: str | None = PrivateAttr(default=None)
    _token_expires_at: float | None = PrivateAttr(default=None)
    _authorization_callback: Callable[[str], Awaitable[str]] | None = PrivateAttr(
        default=None
    )

    def set_authorization_callback(
        self, callback: Callable[[str], Awaitable[str]] | None
    ) -> None:
        """Set callback to handle authorization URL.

        Args:
            callback: Async function that receives authorization URL and returns auth code.
        """
        self._authorization_callback = callback

    async def apply_auth(
        self, client: httpx.AsyncClient, headers: MutableMapping[str, str]
    ) -> MutableMapping[str, str]:
        """Apply OAuth2 access token to Authorization header.

        Args:
            client: HTTP client for making token requests.
            headers: Current request headers.

        Returns:
            Updated headers with OAuth2 access token in Authorization header.

        Raises:
            ValueError: If authorization callback is not set.
        """

        if self._access_token is None:
            if self._authorization_callback is None:
                msg = "Authorization callback not set. Use set_authorization_callback()"
                raise ValueError(msg)
            await self._fetch_initial_token(client)
        elif self._token_expires_at and time.time() >= self._token_expires_at:
            await self._refresh_access_token(client)

        if self._access_token:
            headers["Authorization"] = f"Bearer {self._access_token}"

        return headers

    async def _fetch_initial_token(self, client: httpx.AsyncClient) -> None:
        """Fetch initial access token using authorization code flow.

        Args:
            client: HTTP client for making token request.

        Raises:
            ValueError: If authorization callback is not set.
            httpx.HTTPStatusError: If token request fails.
        """
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(self.scopes),
        }
        auth_url = f"{self.authorization_url}?{urllib.parse.urlencode(params)}"

        if self._authorization_callback is None:
            msg = "Authorization callback not set"
            raise ValueError(msg)
        auth_code = await self._authorization_callback(auth_url)

        data = {
            "grant_type": "authorization_code",
            "code": auth_code,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
            "redirect_uri": self.redirect_uri,
        }

        response = await client.post(self.token_url, data=data)
        response.raise_for_status()

        token_data = response.json()
        self._access_token = token_data["access_token"]
        self._refresh_token = token_data.get("refresh_token")

        expires_in = token_data.get("expires_in", 3600)
        self._token_expires_at = time.time() + expires_in - 60

    async def _refresh_access_token(self, client: httpx.AsyncClient) -> None:
        """Refresh the access token using refresh token.

        Args:
            client: HTTP client for making token request.

        Raises:
            httpx.HTTPStatusError: If token refresh request fails.
        """
        if not self._refresh_token:
            await self._fetch_initial_token(client)
            return

        data = {
            "grant_type": "refresh_token",
            "refresh_token": self._refresh_token,
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }

        response = await client.post(self.token_url, data=data)
        response.raise_for_status()

        token_data = response.json()
        self._access_token = token_data["access_token"]
        if "refresh_token" in token_data:
            self._refresh_token = token_data["refresh_token"]

        expires_in = token_data.get("expires_in", 3600)
        self._token_expires_at = time.time() + expires_in - 60
