"""Authentication schemes for A2A protocol agents.

This module provides support for various authentication methods:
- Bearer tokens (existing)
- OAuth2 (Client Credentials, Authorization Code)
- API Keys (header, query, cookie)
- HTTP Basic authentication
- HTTP Digest authentication
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import base64
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal

import httpx
from pydantic import BaseModel, Field


if TYPE_CHECKING:
    from a2a.types import AgentCard


class AuthScheme(ABC, BaseModel):
    """Base class for authentication schemes."""

    @abstractmethod
    async def apply_auth(
        self, client: httpx.AsyncClient, headers: dict[str, str]
    ) -> dict[str, str]:
        """Apply authentication to request headers.

        Args:
            client: HTTP client for making auth requests.
            headers: Current request headers.

        Returns:
            Updated headers with authentication applied.
        """
        ...

    @abstractmethod
    def configure_client(self, client: httpx.AsyncClient) -> None:
        """Configure the HTTP client for this auth scheme.

        Args:
            client: HTTP client to configure.
        """
        ...


class BearerTokenAuth(AuthScheme):
    """Bearer token authentication (Authorization: Bearer <token>)."""

    token: str = Field(description="Bearer token")

    async def apply_auth(
        self, client: httpx.AsyncClient, headers: dict[str, str]
    ) -> dict[str, str]:
        """Apply Bearer token to Authorization header."""
        headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def configure_client(self, client: httpx.AsyncClient) -> None:
        """No client configuration needed for Bearer tokens."""


class HTTPBasicAuth(AuthScheme):
    """HTTP Basic authentication."""

    username: str = Field(description="Username")
    password: str = Field(description="Password")

    async def apply_auth(
        self, client: httpx.AsyncClient, headers: dict[str, str]
    ) -> dict[str, str]:
        """Apply HTTP Basic authentication."""
        credentials = f"{self.username}:{self.password}"
        encoded = base64.b64encode(credentials.encode()).decode()
        headers["Authorization"] = f"Basic {encoded}"
        return headers

    def configure_client(self, client: httpx.AsyncClient) -> None:
        """No client configuration needed for Basic auth."""


class HTTPDigestAuth(AuthScheme):
    """HTTP Digest authentication.

    Note: Uses httpx-auth library for proper digest implementation.
    """

    username: str = Field(description="Username")
    password: str = Field(description="Password")

    async def apply_auth(
        self, client: httpx.AsyncClient, headers: dict[str, str]
    ) -> dict[str, str]:
        """Digest auth is handled by httpx auth flow, not headers."""
        return headers

    def configure_client(self, client: httpx.AsyncClient) -> None:
        """Configure client with Digest auth."""
        try:
            from httpx_auth import DigestAuth  # type: ignore[import-not-found]

            client.auth = DigestAuth(self.username, self.password)  # type: ignore[import-not-found]
        except ImportError as e:
            msg = "httpx-auth required for Digest authentication. Install with: pip install httpx-auth"
            raise ImportError(msg) from e


class APIKeyAuth(AuthScheme):
    """API Key authentication (header, query, or cookie)."""

    api_key: str = Field(description="API key value")
    location: Literal["header", "query", "cookie"] = Field(
        default="header", description="Where to send the API key"
    )
    name: str = Field(default="X-API-Key", description="Parameter name for the API key")

    async def apply_auth(
        self, client: httpx.AsyncClient, headers: dict[str, str]
    ) -> dict[str, str]:
        """Apply API key authentication."""
        if self.location == "header":
            headers[self.name] = self.api_key
        elif self.location == "cookie":
            headers["Cookie"] = f"{self.name}={self.api_key}"
        # Query params are handled in configure_client via event hooks
        return headers

    def configure_client(self, client: httpx.AsyncClient) -> None:
        """Configure client for query param API keys."""
        if self.location == "query":
            # Add API key to all requests via event hook
            async def add_api_key_param(request: httpx.Request) -> None:
                url = httpx.URL(request.url)
                request.url = url.copy_add_param(self.name, self.api_key)

            client.event_hooks["request"].append(add_api_key_param)


class OAuth2ClientCredentials(AuthScheme):
    """OAuth2 Client Credentials flow authentication."""

    token_url: str = Field(description="OAuth2 token endpoint")
    client_id: str = Field(description="OAuth2 client ID")
    client_secret: str = Field(description="OAuth2 client secret")
    scopes: list[str] = Field(
        default_factory=list, description="Required OAuth2 scopes"
    )

    _access_token: str | None = None
    _token_expires_at: float | None = None

    async def apply_auth(
        self, client: httpx.AsyncClient, headers: dict[str, str]
    ) -> dict[str, str]:
        """Apply OAuth2 access token to Authorization header."""
        # Get or refresh token if needed
        import time

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
        """Fetch OAuth2 access token using client credentials flow."""
        import time

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

        # Calculate expiration time (default to 3600 seconds if not provided)
        expires_in = token_data.get("expires_in", 3600)
        self._token_expires_at = time.time() + expires_in - 60  # 60s buffer

    def configure_client(self, client: httpx.AsyncClient) -> None:
        """No client configuration needed for OAuth2."""


class OAuth2AuthorizationCode(AuthScheme):
    """OAuth2 Authorization Code flow authentication.

    Note: This requires interactive authorization and is typically used
    for user-facing applications. For server-to-server, use ClientCredentials.
    """

    authorization_url: str = Field(description="OAuth2 authorization endpoint")
    token_url: str = Field(description="OAuth2 token endpoint")
    client_id: str = Field(description="OAuth2 client ID")
    client_secret: str = Field(description="OAuth2 client secret")
    redirect_uri: str = Field(description="OAuth2 redirect URI")
    scopes: list[str] = Field(
        default_factory=list, description="Required OAuth2 scopes"
    )

    _access_token: str | None = None
    _refresh_token: str | None = None
    _token_expires_at: float | None = None
    _authorization_callback: Callable[[str], Awaitable[str]] | None = None

    def set_authorization_callback(
        self, callback: Callable[[str], Awaitable[str]] | None
    ) -> None:
        """Set callback to handle authorization URL.

        The callback receives the authorization URL and should return
        the authorization code after user completes the flow.
        """
        self._authorization_callback = callback

    async def apply_auth(
        self, client: httpx.AsyncClient, headers: dict[str, str]
    ) -> dict[str, str]:
        """Apply OAuth2 access token to Authorization header."""
        import time

        # Get or refresh token if needed
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
        """Fetch initial access token using authorization code flow."""
        import time
        import urllib.parse

        # Build authorization URL
        params = {
            "response_type": "code",
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "scope": " ".join(self.scopes),
        }
        auth_url = f"{self.authorization_url}?{urllib.parse.urlencode(params)}"

        # Get authorization code from callback
        if self._authorization_callback is None:
            msg = "Authorization callback not set"
            raise ValueError(msg)
        auth_code = await self._authorization_callback(auth_url)

        # Exchange code for token
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
        """Refresh the access token using refresh token."""
        import time

        if not self._refresh_token:
            # Re-authorize if no refresh token
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

    def configure_client(self, client: httpx.AsyncClient) -> None:
        """No client configuration needed for OAuth2."""


def create_auth_from_agent_card(
    agent_card: AgentCard, credentials: dict[str, Any]
) -> AuthScheme | None:
    """Create an appropriate authentication scheme from AgentCard security config.

    Args:
        agent_card: The A2A AgentCard containing security requirements.
        credentials: User-provided credentials (passwords, tokens, keys, etc.).

    Returns:
        Configured AuthScheme, or None if no authentication required.

    Example:
        ```python
        # For OAuth2
        credentials = {
            "client_id": "my-app",
            "client_secret": "secret123",
        }
        auth = create_auth_from_agent_card(agent_card, credentials)

        # For API Key
        credentials = {"api_key": "key-12345"}
        auth = create_auth_from_agent_card(agent_card, credentials)

        # For HTTP Basic
        credentials = {"username": "user", "password": "pass"}
        auth = create_auth_from_agent_card(agent_card, credentials)
        ```
    """
    if not agent_card.security or not agent_card.security_schemes:
        return None

    # Get the first required security scheme
    first_security_req = agent_card.security[0] if agent_card.security else {}

    for scheme_name, _scopes in first_security_req.items():
        security_scheme_obj = agent_card.security_schemes.get(scheme_name)
        if not security_scheme_obj:
            continue

        # SecurityScheme is a dict-like object
        security_scheme = dict(security_scheme_obj)  # type: ignore[arg-type]
        scheme_type = str(security_scheme.get("type", "")).lower()

        # OAuth2
        if scheme_type == "oauth2":
            flows = security_scheme.get("flows", {})

            if "clientCredentials" in flows:
                flow = flows["clientCredentials"]
                return OAuth2ClientCredentials(
                    token_url=str(flow["tokenUrl"]),
                    client_id=str(credentials.get("client_id", "")),
                    client_secret=str(credentials.get("client_secret", "")),
                    scopes=list(flow.get("scopes", {}).keys()),
                )

            if "authorizationCode" in flows:
                flow = flows["authorizationCode"]
                return OAuth2AuthorizationCode(
                    authorization_url=str(flow["authorizationUrl"]),
                    token_url=str(flow["tokenUrl"]),
                    client_id=str(credentials.get("client_id", "")),
                    client_secret=str(credentials.get("client_secret", "")),
                    redirect_uri=str(credentials.get("redirect_uri", "")),
                    scopes=list(flow.get("scopes", {}).keys()),
                )

        # API Key
        elif scheme_type == "apikey":
            location = str(security_scheme.get("in", "header"))
            name = str(security_scheme.get("name", "X-API-Key"))
            return APIKeyAuth(
                api_key=str(credentials.get("api_key", "")),
                location=location,  # type: ignore[arg-type]
                name=name,
            )

        # HTTP Auth
        elif scheme_type == "http":
            http_scheme = str(security_scheme.get("scheme", "")).lower()

            if http_scheme == "basic":
                return HTTPBasicAuth(
                    username=str(credentials.get("username", "")),
                    password=str(credentials.get("password", "")),
                )

            if http_scheme == "digest":
                return HTTPDigestAuth(
                    username=str(credentials.get("username", "")),
                    password=str(credentials.get("password", "")),
                )

            if http_scheme == "bearer":
                return BearerTokenAuth(token=str(credentials.get("token", "")))

    return None
